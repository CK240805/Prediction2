#!/usr/bin/env python3
"""
LottOracle-Advanced (Fixed RMT, Deprecation‑free)
==================================================
Reads 6/49 lottery CSV (draw_no,date,n1..n6,additional).
Enhancements:
  - Dirichlet-Multinomial likelihood
  - RMT denoising of ball counts (corrected)
  - Trend (momentum) feature
  - Monte Carlo set sampling (most frequent combination)
Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize
from scipy.special import gammaln
from collections import Counter
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"
MIN_HISTORY_FOR_CAL = 100        # draws needed before starting cross‑validation
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6
BALL_MASS_MEAN = 3.3e-3          # kg
BALL_MASS_TOL  = 0.1e-3          # kg
EWMA_DECAY = 0.95
TREND_WINDOW = 20

# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_draws_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.sort_values("draw_no").reset_index(drop=True)
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1
    dates = df["date"]
    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws, dates

# =============================================================================
# 2. ENVIRONMENTAL MODEL
# =============================================================================
def get_env_from_date(date):
    month = date.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. CORRECTED RMT DENOISER
# =============================================================================
def marchenko_pastur_threshold(q):
    """MP upper bound for eigenvalue distribution with aspect ratio q = p/T."""
    return (1 + np.sqrt(q))**2

def rmt_denoise_counts(counts_history):
    """
    counts_history: numpy array of shape (T, NUM_BALLS)
    Returns a denoised count vector (NUM_BALLS,) for the latest time step,
    obtained by projecting onto RMT‑filtered eigenvectors.
    """
    T = len(counts_history)
    if T < NUM_BALLS:
        return counts_history[-1].copy()

    X = counts_history  # (T, 49)
    # Standardize each ball (column-wise)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_stdz = (X - X_mean) / X_std

    # Correlation matrix of balls (49x49)
    corr = np.corrcoef(X_stdz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)

    # Marchenko‑Pastur threshold
    p = NUM_BALLS
    q = p / T
    lambda_plus = marchenko_pastur_threshold(q)
    significant = eigvals > lambda_plus

    # Project data onto significant eigenvectors
    V_sig = eigvecs[:, significant]                 # (49, k)
    X_clean = X_stdz @ V_sig @ V_sig.T              # (T, 49) @ (49,k) @ (k,49) -> (T,49)
    # Recover original scale
    X_denoised = X_clean * X_std + X_mean
    latest_denoised = X_denoised[-1, :].clip(min=0)
    return latest_denoised

# =============================================================================
# 4. EWMA COUNTS & MASS ESTIMATION
# =============================================================================
def update_ewma_counts(prev_counts, draw, decay=EWMA_DECAY):
    new_counts = decay * prev_counts
    for ball in draw:
        new_counts[ball] += (1 - decay) * (NUM_BALLS / DRAW_SIZE)
    return new_counts

def masses_from_counts(counts, mean_mass=BALL_MASS_MEAN, tol=BALL_MASS_TOL):
    counts = np.maximum(counts, 1e-8)
    freq = counts / counts.sum()
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * mean_mass
    return np.clip(raw_mass, mean_mass - tol, mean_mass + tol)

# =============================================================================
# 5. TREND (MOMENTUM) FEATURE
# =============================================================================
def compute_trend_slopes(counts_series, window=TREND_WINDOW):
    """counts_series: list/array of count vectors (length NUM_BALLS)."""
    if len(counts_series) < window:
        return np.zeros(NUM_BALLS)
    recent = np.array(counts_series[-window:])   # (window, 49)
    x = np.arange(window).reshape(-1, 1)
    slopes = np.zeros(NUM_BALLS)
    model = LinearRegression()
    for i in range(NUM_BALLS):
        y = recent[:, i]
        model.fit(x, y)
        slopes[i] = model.coef_[0]
    # Normalize to a reasonable scale
    slopes = slopes / (np.std(slopes) + 1e-6) * 0.1
    return slopes

# =============================================================================
# 6. PROBABILITY MODEL
# =============================================================================
def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0,
                          trend_slopes=None, delta=0.0):
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    if trend_slopes is not None and delta != 0.0:
        trend_factor = np.exp(delta * trend_slopes)
        weight = weight * trend_factor
    prob = weight / weight.sum()
    return prob

# =============================================================================
# 7. DIRICHLET-MULTINOMIAL LOG‑LIKELIHOOD
# =============================================================================
def dirichlet_multinomial_loglike(prob_vec, draw, concentration):
    """Log‑likelihood of drawing a set of 6 balls without replacement."""
    alpha = concentration * prob_vec
    A0 = alpha.sum()
    x = np.bincount(draw, minlength=NUM_BALLS)
    ll = gammaln(A0) - gammaln(DRAW_SIZE + A0)
    for i in range(NUM_BALLS):
        ll += gammaln(x[i] + alpha[i]) - gammaln(alpha[i])
    return ll

# =============================================================================
# 8. CROSS‑VALIDATION CALIBRATION
# =============================================================================
def cv_log_likelihood(params, draws_cal, dates_cal, min_history=MIN_HISTORY_FOR_CAL):
    alpha, beta, gamma, delta, log_concentration = params
    concentration = np.exp(log_concentration)
    n_draws = len(draws_cal)
    if n_draws <= min_history:
        return -1e12

    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    counts_history = []

    total_ll = 0.0
    for i in range(n_draws):
        draw = draws_cal[i]
        if i >= min_history:
            # RMT denoise using the counts history built so far
            denoised = rmt_denoise_counts(np.array(counts_history))
            masses = masses_from_counts(denoised)
            temp, hum = get_env_from_date(dates_cal[i])
            slopes = compute_trend_slopes(counts_history, window=TREND_WINDOW)
            prob = physics_probabilities(masses, alpha, temp, hum, beta, gamma,
                                        trend_slopes=slopes, delta=delta)
            total_ll += dirichlet_multinomial_loglike(prob, draw, concentration)

        counts = update_ewma_counts(counts, draw)
        counts_history.append(counts.copy())

    return total_ll

def calibrate_parameters(draws_cal, dates_cal):
    x0 = np.array([2.0, 0.0, 0.0, 0.0, np.log(10.0)])
    bounds = [(0.5, 5.0), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (np.log(1), np.log(1000))]

    def objective(params):
        return -cv_log_likelihood(params, draws_cal, dates_cal)

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300})
    if not result.success:
        print("Warning: optimisation did not converge:", result.message)

    alpha, beta, gamma, delta, log_conc = result.x
    print(f"\nCalibrated parameters (cross‑validated):")
    print(f"  α (mass sensitivity) = {alpha:.3f}")
    print(f"  β (temperature)      = {beta:.3f}")
    print(f"  γ (humidity)         = {gamma:.3f}")
    print(f"  δ (trend)            = {delta:.3f}")
    print(f"  concentration        = {np.exp(log_conc):.1f}")
    print(f"  CV log‑likelihood    = {-result.fun:.2f}")
    return alpha, beta, gamma, delta, np.exp(log_conc)

# =============================================================================
# 9. MONTE CARLO SET SAMPLING
# =============================================================================
def sample_best_set(prob_vector, num_draws=500_000, top_n=3):
    set_counter = Counter()
    p = prob_vector.astype(np.float64)
    batch_size = 50000
    for batch_start in range(0, num_draws, batch_size):
        actual = min(batch_size, num_draws - batch_start)
        draws = np.array([
            np.random.choice(NUM_BALLS, size=DRAW_SIZE, replace=False, p=p)
            for _ in range(actual)
        ])
        for draw in draws:
            set_counter[tuple(np.sort(draw))] += 1
    most_common = set_counter.most_common(top_n)
    return [(np.array(combo) + 1, count) for combo, count in most_common]

# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================
def main():
    draws, dates = load_draws_from_csv(CSV_FILE)
    if len(draws) < MIN_HISTORY_FOR_CAL + 1:
        print(f"Need at least {MIN_HISTORY_FOR_CAL+1} draws, got {len(draws)}.")
        sys.exit(1)

    calibration_draws = draws[:-1]
    calibration_dates = dates[:-1]
    last_draw = draws[-1]
    last_date = dates.iloc[-1]
    print(f"Using {len(calibration_draws)} draws for calibration.")
    print(f"Most recent known draw: {last_draw + 1}")

    # Calibrate
    alpha, beta, gamma, delta, concentration = calibrate_parameters(calibration_draws, calibration_dates)

    # Final state (all calibration draws)
    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    counts_history = []
    for draw in calibration_draws:
        counts = update_ewma_counts(counts, draw)
        counts_history.append(counts.copy())

    denoised_counts = rmt_denoise_counts(np.array(counts_history))
    final_masses = masses_from_counts(denoised_counts)
    print("\nFinal estimated masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {final_masses[i]*1000:.3f} g")

    temp_next, hum_next = get_env_from_date(last_date)
    final_slopes = compute_trend_slopes(counts_history[-TREND_WINDOW:])

    prob_vec = physics_probabilities(final_masses, alpha, temp_next, hum_next,
                                    beta, gamma, final_slopes, delta)
    print("\nFinal physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    # Set sampling
    print("\n--- Monte Carlo set sampling (500k draws) ---")
    top_sets = sample_best_set(prob_vec, num_draws=500_000, top_n=5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {combo}  (occurred {count} times)")

    print("\nTop 10 marginal probabilities:")
    for rank, idx in enumerate(np.argsort(prob_vec)[-10:][::-1]):
        print(f"  {rank+1}. Ball {idx+1:02d}: {prob_vec[idx]:.4f}")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
