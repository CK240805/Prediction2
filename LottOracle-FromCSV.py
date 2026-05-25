#!/usr/bin/env python3
"""
LottOracle-FromCSV-Calibrated
==============================
Reads historical 6/49 draws from CSV.
1. Estimates ball masses from draw frequencies.
2. Simulates the historical draws to find the best air‑mix sensitivity α
   (the parameter that best reproduces the past).
3. Uses the calibrated model to simulate future draws until convergence
   and predicts the next set of numbers.

Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm diameter.
"""

import numpy as np
import pandas as pd
import sys
from scipy.special import logsumexp

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "historical_draws.csv"   # <-- your file
HISTORY_LEN = 1189                    # draws used for mass estimation
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 14_000_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6

BALL_MASS_MEAN = 3.3e-3            # kg (3.3 g)
BALL_MASS_TOL  = 0.1e-3            # kg (±0.1 g)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_draws_from_csv(filepath):
    df = pd.read_csv(filepath)
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.sort_values("draw_no").reset_index(drop=True)
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1  # 0-indexed
    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws

# =============================================================================
# 2. MASS ESTIMATION (clipped to ±0.1 g)
# =============================================================================
def estimate_ball_masses(draws_history):
    H = len(draws_history)
    counts = np.zeros(NUM_BALLS)
    for draw in draws_history:
        for ball in draw:
            counts[ball] += 1
    counts = counts + 1e-6
    freq = counts / (H * DRAW_SIZE)
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * BALL_MASS_MEAN
    raw_mass = np.clip(raw_mass,
                       BALL_MASS_MEAN - BALL_MASS_TOL,
                       BALL_MASS_MEAN + BALL_MASS_TOL)
    return raw_mass

# =============================================================================
# 3. PHYSICS PROBABILITY GIVEN MASSES AND α
# =============================================================================
def physics_probabilities(masses, alpha):
    """P(ball) ∝ (1/mass)^α"""
    weight = (1.0 / masses) ** alpha
    prob = weight / np.sum(weight)
    return prob

# =============================================================================
# 4. LIKELIHOOD OF HISTORICAL DRAWS UNDER GIVEN α
# =============================================================================
def historical_log_likelihood(draws, masses, alpha):
    """
    Computes the total log-likelihood of all draws, assuming each draw
    consists of 6 independent picks from the physics probabilities.
    (This ignores the without-replacement constraint, which is a minor
    approximation for calibration purposes.)
    """
    prob = physics_probabilities(masses, alpha)
    # Each draw's log-likelihood = sum of log-prob of each drawn ball
    log_like = 0.0
    for draw in draws:
        log_like += np.sum(np.log(prob[draw]))
    return log_like

# =============================================================================
# 5. CALIBRATE α BY MAXIMISING HISTORICAL LIKELIHOOD
# =============================================================================
def calibrate_alpha(draws_cal, masses):
    """
    Searches for the α that best reproduces the historical draws.
    We use a simple grid search over [0.5, 5.0] in steps of 0.01.
    """
    best_alpha = None
    best_loglike = -np.inf
    for alpha in np.arange(0.5, 5.01, 0.01):
        ll = historical_log_likelihood(draws_cal, masses, alpha)
        if ll > best_loglike:
            best_loglike = ll
            best_alpha = alpha
    print(f"\nCalibrated α = {best_alpha:.3f} (max log-likelihood = {best_loglike:.2f})")
    return best_alpha

# =============================================================================
# 6. CONTINUOUS MONTE CARLO SIMULATION (FORWARD)
# =============================================================================
def simulate_until_convergence(prob_vector, threshold=CONVERGENCE_THRESHOLD,
                               max_iters=MAX_SIM_DRAWS, batch_size=BATCH_SIZE):
    counts = np.zeros(NUM_BALLS, dtype=np.int64)
    prev_probs = None
    iteration = 0
    delta = None

    while iteration < max_iters:
        batch_draws = np.array([
            np.random.choice(NUM_BALLS, size=DRAW_SIZE, replace=False, p=prob_vector)
            for _ in range(batch_size)
        ])
        for ball in batch_draws.flat:
            counts[ball] += 1

        total_draws = (iteration + 1) * batch_size
        empirical_probs = counts / (total_draws * DRAW_SIZE)

        if prev_probs is not None:
            delta = np.max(np.abs(empirical_probs - prev_probs))
            if delta < threshold:
                break
        prev_probs = empirical_probs.copy()
        iteration += 1

        if iteration % 10 == 0:
            msg = f"  Simulated {total_draws} draws"
            if delta is not None:
                msg += f", max delta = {delta:.6f}"
            print(msg)

    final_total = (iteration + 1) * batch_size
    if iteration >= max_iters:
        print("Warning: max iterations reached without full convergence.")
    else:
        print(f"Converged after {final_total} virtual draws. Final delta = {delta:.6f}")
    return empirical_probs, final_total

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================
def main():
    draws = load_draws_from_csv(CSV_FILE)
    if len(draws) < HISTORY_LEN + 1:
        print(f"Need at least {HISTORY_LEN+1} draws, only {len(draws)} available.")
        sys.exit(1)

    # Separate calibration set (all draws except the very last one)
    calibration_draws = draws[:-1]
    last_known_draw = draws[-1]   # this is the most recent draw, NOT the target

    print(f"\nUsing {len(calibration_draws)} draws for calibration.")
    print(f"Most recent draw (for reference only): {last_known_draw + 1}")

    # --- Step 1: Estimate masses from recent history ---
    # We use the last HISTORY_LEN draws *before the last one* for mass estimation.
    recent_for_mass = calibration_draws[-HISTORY_LEN:]
    masses = estimate_ball_masses(recent_for_mass)
    print("\nEstimated ball masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {masses[i]*1000:.3f} g")

    # --- Step 2: Calibrate α using ALL calibration draws ---
    best_alpha = calibrate_alpha(calibration_draws, masses)

    # --- Step 3: Compute final physics probability vector ---
    prob_vec = physics_probabilities(masses, best_alpha)
    print("\nCalibrated physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    # --- Step 4: Forward simulation until convergence ---
    print("\n--- Forward Monte Carlo simulation for next draw ---")
    final_probs, total_sim = simulate_until_convergence(prob_vec)

    # --- Step 5: Predict the next set ---
    top_idx = np.argsort(final_probs)[-DRAW_SIZE:][::-1]
    predicted = top_idx + 1
    predicted.sort()
    print(f"\nPredicted next set: {predicted}")

    print("\nTop 10 final probabilities:")
    for rank, idx in enumerate(np.argsort(final_probs)[-10:][::-1]):
        print(f"  {rank+1}. Ball {idx+1:02d}: {final_probs[idx]:.4f}")

    print(f"\nTotal virtual draws: {total_sim}")
    print("Disclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
