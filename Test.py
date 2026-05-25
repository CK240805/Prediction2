#!/usr/bin/env python3
"""
LottOracle-Advanced (Full Physics + RMT + Trend + Dirichlet-Multinomial + Set Sampling)
Reads 6/49 lottery CSV (draw_no,date,n1..n6,additional).
Enhancements:
  - Dirichlet-Multinomial likelihood for draws without replacement
  - RMT denoising of ball occurrence counts
  - Trend (momentum) feature in probability model
  - Final set selection via Monte Carlo set sampling (most frequent combination)
Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize
from scipy.special import gammaln
from collections import Counter
from sklearn.linear_model import LinearRegression  # for trend computation

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results1.csv"
MIN_HISTORY_FOR_CAL = 100        # draws needed before starting cross‑validation
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6
BALL_MASS_MEAN = 3.3e-3          # kg
BALL_MASS_TOL  = 0.1e-3          # kg

# EWMA decay for counts
EWMA_DECAY = 0.95
# Window for trend estimation
TREND_WINDOW = 20

# =============================================================================
# 1. DATA LOADING (parse dates)
# =============================================================================
def load_draws_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.sort_values("draw_no").reset_index(drop=True)
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1  # 0-indexed
    dates = df["date"]
    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws, dates

# =============================================================================
# 2. ENVIRONMENTAL MODEL (seasonal temp/humidity)
# =============================================================================
def get_env_from_date(date):
    month = date.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. RMT DENOISER OF COUNT VECTORS
# =============================================================================
def marchenko_pastur_threshold(q, sigma=1.0):
    """Marchenko-Pastur upper bound for correlation matrix with aspect ratio q."""
    return sigma**2 * (1 + np.sqrt(q))**2

def rmt_denoise_counts(counts_history):
    """
    counts_history: numpy array of shape (T, NUM_BALLS) where each row is
                    the EWMA count vector after observing draws up to time t.
    Returns denoised count vector for the latest time step.
    """
    T = len(counts_history)
    if T < NUM_BALLS:
        # not enough data for RMT, return raw latest counts
        return counts_history[-1].copy()
    # Standardize each ball's time series to zero mean, unit variance
    X = counts_history.T  # shape (NUM_BALLS, T)
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_stdz = (X - X_mean) / X_std
    # Correlation matrix (NUM_BALLS x NUM_BALLS)
    corr = np.corrcoef(X_stdz)
    eigvals, eigvecs = np.linalg.eigh(corr)
    q = T / NUM_BALLS
    lambda_plus = marchenko_pastur_threshold(q)
    # Keep eigenvectors with eigenvalues > lambda_plus
    significant = eigvals > lambda_plus
    # Project the standardized data onto significant eigenvectors (denoised)
    X_clean = X_stdz @ eigvecs[:, significant] @ eigvecs[:, significant].T
    # Recover original scale and mean
    X_denoised = X_clean * X_std + X_mean
    # Latest column is the denoised count vector
    latest_counts = X_denoised[:, -1].clip(min=0)
    return latest_counts

# =============================================================================
# 4. EWMA COUNT UPDATER & MASS ESTIMATION
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
# 5. TREND FEATURE (slope of recent EWMA counts)
# =============================================================================
def compute_trend_slopes(counts_series, window=TREND_WINDOW):
    """
    counts_series: list/array of count vectors (time steps). Returns slopes (49,).
    """
    if len(counts_series) < window:
        return np.zeros(NUM_BALLS)
    recent = np.array(counts_series[-window:])  # (window, 49)
    x = np.arange(window).reshape(-1, 1)
    slopes = np.zeros(NUM_BALLS)
    model = LinearRegression()
    for i in range(NUM_BALLS):
        y = recent[:, i]
        model.fit(x, y)
        slopes[i] = model.coef_[0]
    # Normalize slopes to a reasonable range
    slopes = slopes / (np.std(slopes) + 1e-6) * 0.1
    return slopes

# =============================================================================
# 6. PROBABILITY MODEL (with physics, env, trend, Dirichlet)
# =============================================================================
def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0,
                          trend_slopes=None, delta=0.0):
    """P(ball) ∝ (1/mass)^α * exp(βΔT + γΔH + δ*trend)"""
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    if trend_slopes is not None and delta != 0.0:
        trend_factor = np.exp(delta * trend_slopes)
        weight = weight * trend_factor
    prob = weight / weight.sum()
    return prob

# =============================================================================
# 7. DIRICHLET-MULTINOMIAL LOG-LIKELIHOOD
# =============================================================================
def dirichlet_multinomial_loglike(prob_vec, draw, concentration):
    """
    Log-likelihood of drawing a set (without replacement) under Dirichlet-Multinomial.
    prob_vec: marginal probabilities (49,) of each ball.
    concentration: scalar > 0, where alpha_i = concentration * prob_i.
    draw: array of 6 ball indices (0-indexed).
    """
    alpha = concentration * prob_vec
    A0 = alpha.sum()
    # Counts of each ball in the draw (0 or 1)
    x = np.bincount(draw, minlength=NUM_BALLS)
    # DM log-likelihood = gammaln(A0) - gammaln(DRAW_SIZE + A0) +
    #                     sum_i[ gammaln(x_i + alpha_i) - gammaln(alpha_i) ]
    ll = gammaln(A0) - gammaln(DRAW_SIZE + A0)
    for i in range(NUM_BALLS):
        ll += gammaln(x[i] + alpha[i]) - gammaln(alpha[i])
    return ll

# =============================================================================
# 8. CROSS-VALIDATED CALIBRATION (includes concentration & trend)
# =============================================================================
def cv_log_likelihood(params, draws_cal, dates_cal, min_history=MIN_HISTORY_FOR_CAL):
    alpha, beta, gamma, delta, log_concentration = params
    concentration = np.exp(log_concentration)  # ensure positive
    n_draws = len(draws_cal)
    if n_draws <= min_history:
        return -1e12

    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    # Store historical count vectors for trend and RMT
    counts_history = []

    total_ll = 0.0
    for i in range(n_draws):
        draw = draws_cal[i]
        # Only evaluate likelihood after we have enough history
        if i >= min_history:
            # Get denoised counts via RMT using the counts_history so far
            denoised_counts = rmt_denoise_counts(np.array(counts_history))
            masses = masses_from_counts(denoised_counts)
            temp, hum = get_env_from_date(dates_cal[i])
            # Compute trend slopes from recent counts history (raw, not denoised)
            slopes = compute_trend_slopes(counts_history, window=TREND_WINDOW)
            prob = physics_probabilities(masses, alpha, temp, hum, beta, gamma,
                                        trend_slopes=slopes, delta=delta)
            # Dirichlet-multinomial log-likelihood
            ll = dirichlet_multinomial_loglike(prob, draw, concentration)
            total_ll += ll

        # Update EWMA counts for next step
        counts = update_ewma_counts(counts, draw)
        counts_history.append(counts.copy())

    return total_ll

def calibrate_parameters(draws_cal, dates_cal):
    # Initial guess: α=2.0, β=0, γ=0, δ=0, log_concentration=log(10)
    x0 = np.array([2.0, 0.0, 0.0, 0.0, np.log(10.0)])
    bounds = [(0.5, 5.0),   # α
              (-0.5, 0.5),  # β
              (-0.5, 0.5),  # γ
              (-0.5, 0.5),  # δ (trend sensitivity)
              (np.log(1), np.log(1000))]  # concentration

    def objective(params):
        return -cv_log_likelihood(params, draws_cal, dates_cal)

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'disp': False})
    if not result.success:
        print("Warning: optimisation did not converge:", result.message)

    best_params = result.x
    alpha, beta, gamma, delta, log_conc = best_params
    print(f"\nCalibrated parameters (cross‑validated):")
    print(f"  α (mass sensitivity)   = {alpha:.3f}")
    print(f"  β (temperature)        = {beta:.3f}")
    print(f"  γ (humidity)           = {gamma:.3f}")
    print(f"  δ (trend)              = {delta:.3f}")
    print(f"  concentration          = {np.exp(log_conc):.1f}")
    print(f"  Cross‑validated log‑likelihood = {-result.fun:.2f}")
    return alpha, beta, gamma, delta, np.exp(log_conc)

# =============================================================================
# 9. MONTE CARLO SET SAMPLING (most frequent combination)
# =============================================================================
def sample_best_set(prob_vector, num_draws=500_000, top_n=3):
    """
    Samples many draws without replacement from prob_vector, counts full
    combinations, and returns the most frequent combination(s).
    """
    set_counter = Counter()
    # Convert probabilities to float64 for speed
    p = prob_vector.astype(np.float64)
    # We'll sample in batches
    batch_size = 50000
    for batch_start in range(0, num_draws, batch_size):
        actual_batch = min(batch_size, num_draws - batch_start)
        draws = np.array([
            np.random.choice(NUM_BALLS, size=DRAW_SIZE, replace=False, p=p)
            for _ in range(actual_batch)
        ])
        # Convert each draw to a tuple (sorted)
        for draw in draws:
            set_counter[tuple(np.sort(draw))] += 1

    # Get top_n most common sets
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

    # --- Calibration ---
    alpha, beta, gamma, delta, concentration = calibrate_parameters(calibration_draws, calibration_dates)

    # --- Final state estimation (using all calibration draws) ---
    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    counts_history = []
    for draw in calibration_draws:
        counts = update_ewma_counts(counts, draw)
        counts_history.append(counts.copy())

    # RMT denoising of final state
    denoised_counts = rmt_denoise_counts(np.array(counts_history))
    final_masses = masses_from_counts(denoised_counts)
    print("\nFinal estimated masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {final_masses[i]*1000:.3f} g")

    # Environment for next draw
    temp_next, hum_next = get_env_from_date(last_date)

    # Trend slopes from recent history
    final_slopes = compute_trend_slopes(counts_history[-TREND_WINDOW:])

    # Final probability vector
    prob_vec = physics_probabilities(final_masses, alpha, temp_next, hum_next,
                                    beta, gamma, final_slopes, delta)
    print("\nFinal physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    # --- Monte Carlo set sampling to find best combination ---
    print("\n--- Monte Carlo set sampling (500k draws) ---")
    top_sets = sample_best_set(prob_vec, num_draws=500_000, top_n=5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {combo}  (occurred {count} times)")

    # Also print marginal probabilities for reference
    print("\nTop 10 marginal probabilities:")
    for rank, idx in enumerate(np.argsort(prob_vec)[-10:][::-1]):
        print(f"  {rank+1}. Ball {idx+1:02d}: {prob_vec[idx]:.4f}")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
