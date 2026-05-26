#!/usr/bin/env python3
"""
LottOracle – Unknown Ball Sets (3.3–4.0 g, 1% tolerance within each set)
=========================================================================
The lottery uses multiple ball sets. Each set has a nominal mass between
3.3 g and 4.0 g, and within a set every ball is within 1 % of that nominal.
Because the set used for each draw is unknown, we cannot assume a fixed
mean.  Instead, we allow the per‑ball mass estimate to range over the
entire 3.3–4.0 g interval.  The EWMA frequency estimator will converge
to the average mass of each ball over all the sets that have been used.

Physics (air‑mix), EWMA state estimation, Dirichlet‑Multinomial likelihood,
cross‑validated α calibration, Monte Carlo set sampling.
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from collections import Counter
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"
MIN_HISTORY_FOR_CAL = 100
NUM_BALLS = 49
DRAW_SIZE = 6

# Because sets vary, we use the midpoint as a starting point and allow the
# estimate to move anywhere within the full possible mass range.
BALL_MASS_MEAN = 3.65e-3            # kg  (midpoint of 3.3 – 4.0 g)
BALL_MASS_TOL  = 0.35e-3            # kg  (lower bound 3.3, upper bound 4.0)

EWMA_DECAY = 0.98
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
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1  # 0‑indexed
    dates = df["date"]
    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws, dates

# =============================================================================
# 2. ENVIRONMENTAL MODEL (seasonal)
# =============================================================================
def get_env_from_date(date):
    month = date.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. EWMA COUNTS & MASS ESTIMATION
# =============================================================================
def update_ewma_counts(prev_counts, draw, decay=EWMA_DECAY):
    new_counts = decay * prev_counts
    for ball in draw:
        new_counts[ball] += (1 - decay) * (NUM_BALLS / DRAW_SIZE)
    return new_counts

def masses_from_counts(counts, mean_mass=BALL_MASS_MEAN, tol=BALL_MASS_TOL):
    """Convert EWMA counts to ball masses, clipping to the allowed full range."""
    counts = np.maximum(counts, 1e-8)
    freq = counts / counts.sum()
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * mean_mass
    # Clip to [3.3 g, 4.0 g]
    return np.clip(raw_mass, mean_mass - tol, mean_mass + tol)

# =============================================================================
# 4. TREND FEATURE (optional)
# =============================================================================
def compute_trend_slopes(counts_series, window=TREND_WINDOW):
    if len(counts_series) < window:
        return np.zeros(NUM_BALLS)
    recent = np.array(counts_series[-window:])
    x = np.arange(window).reshape(-1, 1)
    slopes = np.zeros(NUM_BALLS)
    model = LinearRegression()
    for i in range(NUM_BALLS):
        y = recent[:, i]
        model.fit(x, y)
        slopes[i] = model.coef_[0]
    slopes = slopes / (np.std(slopes) + 1e-6) * 0.1
    return slopes

# =============================================================================
# 5. PHYSICS PROBABILITY (air‑mix)
# =============================================================================
def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0,
                          trend_slopes=None, delta=0.0):
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    if trend_slopes is not None and delta != 0.0:
        weight = weight * np.exp(delta * trend_slopes)
    return weight / weight.sum()

# =============================================================================
# 6. DIRICHLET‑MULTINOMIAL LOG‑LIKELIHOOD
# =============================================================================
def dirichlet_multinomial_loglike(prob_vec, draw, concentration):
    alpha = concentration * prob_vec
    A0 = alpha.sum()
    x = np.bincount(draw, minlength=NUM_BALLS)
    ll = gammaln(A0) - gammaln(DRAW_SIZE + A0)
    for i in range(NUM_BALLS):
        ll += gammaln(x[i] + alpha[i]) - gammaln(alpha[i])
    return ll

# =============================================================================
# 7. CROSS‑VALIDATION CALIBRATION (α and concentration)
# =============================================================================
def cv_log_likelihood(params, draws_cal, dates_cal, min_history=MIN_HISTORY_FOR_CAL):
    alpha, log_concentration = params
    concentration = np.exp(log_concentration)
    n_draws = len(draws_cal)
    if n_draws <= min_history:
        return -1e12

    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    total_ll = 0.0
    for i in range(n_draws):
        draw = draws_cal[i]
        if i >= min_history:
            masses = masses_from_counts(counts)
            temp, hum = get_env_from_date(dates_cal[i])
            prob = physics_probabilities(masses, alpha, temp, hum)
            total_ll += dirichlet_multinomial_loglike(prob, draw, concentration)
        counts = update_ewma_counts(counts, draw)
    return total_ll

def calibrate_parameters(draws_cal, dates_cal):
    x0 = np.array([2.0, np.log(10.0)])
    bounds = [(0.5, 5.0), (np.log(1), np.log(1000))]
    def objective(params):
        return -cv_log_likelihood(params, draws_cal, dates_cal)
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300})
    if not result.success:
        print("Warning: optimisation did not converge:", result.message)
    alpha, log_conc = result.x
    concentration = np.exp(log_conc)
    print(f"\nCalibrated parameters:")
    print(f"  α (mass sensitivity) = {alpha:.3f}")
    print(f"  concentration        = {concentration:.1f}")
    return alpha, concentration

# =============================================================================
# 8. MONTE CARLO SET SAMPLING
# =============================================================================
def sample_best_set(prob_vector, num_draws=500_000, top_n=3):
    set_counter = Counter()
    p = prob_vector.astype(np.float64)
    p = p / p.sum()                     # ensure exact normalisation
    batch_sz = 50000
    for batch_start in range(0, num_draws, batch_sz):
        actual = min(batch_sz, num_draws - batch_start)
        draws = np.array([
            np.random.choice(NUM_BALLS, size=DRAW_SIZE, replace=False, p=p)
            for _ in range(actual)
        ])
        for d in draws:
            set_counter[tuple(np.sort(d))] += 1
    most_common = set_counter.most_common(top_n)
    return [(np.array(combo) + 1, count) for combo, count in most_common]

# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
def main():
    draws, dates = load_draws_from_csv(CSV_FILE)
    if len(draws) < MIN_HISTORY_FOR_CAL + 1:
        print(f"Need at least {MIN_HISTORY_FOR_CAL+1} draws, got {len(draws)}.")
        sys.exit(1)

    cal_draws = draws[:-1]
    cal_dates = dates[:-1]
    last_draw = draws[-1]
    last_date = dates.iloc[-1]
    print(f"Using {len(cal_draws)} draws for calibration.")

    # Calibrate α and concentration
    alpha, concentration = calibrate_parameters(cal_draws, cal_dates)

    # Final state estimation (EWMA over all calibration draws)
    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    for draw in cal_draws:
        counts = update_ewma_counts(counts, draw)
    final_masses = masses_from_counts(counts)
    print("\nFinal estimated masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {final_masses[i]*1000:.3f} g")

    temp_next, hum_next = get_env_from_date(last_date)
    prob_vec = physics_probabilities(final_masses, alpha, temp_next, hum_next)
    print("\nFinal physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    print("\n--- Monte Carlo set sampling (500k draws) ---")
    top_sets = sample_best_set(prob_vec, num_draws=500_000, top_n=5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {combo}  (occurred {count} times)")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
