#!/usr/bin/env python3
"""
LottOracle-Advanced
====================
Reads historical 6/49 draws from CSV (draw_no,date,n1..n6,additional),
now with three major enhancements:
  1. Bayesian mass drift tracker (EWMA over historical counts)
  2. Environmental coupling using a seasonal temperature/humidity model
  3. Cross‑validated calibration of α, β, γ via walk‑forward likelihood
Then runs forward Monte Carlo simulation to convergence and predicts the next set.

Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm diameter.
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"      # your file
HISTORY_LEN = 1188                   # minimum draws before starting cross‑validation
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 14_000_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6

BALL_MASS_MEAN = 3.3e-3            # kg (3.3 g)
BALL_MASS_TOL  = 0.1e-3            # kg (±0.1 g)

# EWMA decay factor for state estimation (1 = keep all, <1 forget past)
EWMA_DECAY = 0.95

# =============================================================================
# 1. LOAD DATA (including date parsing for seasonal env)
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
# 2. ENVIRONMENTAL MODEL: synthetic seasonal temp/humidity from date
# =============================================================================
def get_env_from_date(date):
    """Return (temperature, humidity) based on month (southern hemisphere style)."""
    month = date.month
    # temperature: peak ~30°C in Jan, trough ~20°C in Jul (approx)
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    # humidity: higher in winter (simplified)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. MASS ESTIMATION WITH EWMA (Bayesian drift tracker)
# =============================================================================
def update_ewma_counts(prev_counts, draw, decay=EWMA_DECAY):
    """
    Updates the EWMA of per‑ball occurrence counts.
    prev_counts: (49,) current running counts.
    draw: array of 6 ball indices (0‑indexed).
    Returns updated counts.
    """
    new_counts = decay * prev_counts
    for ball in draw:
        new_counts[ball] += (1 - decay) * (NUM_BALLS / DRAW_SIZE)  # weighted contribution
    return new_counts

def masses_from_counts(counts, mean_mass=BALL_MASS_MEAN, tol=BALL_MASS_TOL):
    """
    Convert EWMA counts to estimated ball masses.
    Higher count → lighter ball.
    """
    counts = np.maximum(counts, 1e-8)
    freq = counts / counts.sum()   # approx frequency (scaled)
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * mean_mass
    raw_mass = np.clip(raw_mass, mean_mass - tol, mean_mass + tol)
    return raw_mass

# =============================================================================
# 4. PHYSICS PROBABILITY WITH ENVIRONMENTAL COUPLING
# =============================================================================
def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0):
    """
    Base probability ∝ (1/mass)^α
    multiplied by environmental factor exp(β·(T-25) + γ·(H-60)).
    """
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    prob = weight / weight.sum()
    return prob

# =============================================================================
# 5. CROSS‑VALIDATION LIKELIHOOD (walk‑forward)
# =============================================================================
def cv_log_likelihood(params, draws_cal, dates_cal, history_len=HISTORY_LEN):
    """
    Walk‑forward cross‑validation log‑likelihood.
    For each draw i from history_len to end-1, we use all past draws up to i-1
    (EWMA) to estimate masses, then compute log‑likelihood of draw i.
    Returns the total log‑likelihood (higher is better).
    """
    alpha, beta, gamma = params
    n_draws = len(draws_cal)
    if n_draws <= history_len:
        return -1e12  # insufficient data

    # Initialise EWMA counts uniformly
    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)

    total_ll = 0.0

    for i in range(n_draws):
        draw = draws_cal[i]
        if i >= history_len:
            # estimate masses using counts (data up to i-1)
            masses = masses_from_counts(counts)
            # get env for the draw we are predicting (draw i)
            temp, hum = get_env_from_date(dates_cal[i])
            prob = physics_probabilities(masses, alpha, temp, hum, beta, gamma)
            # log-likelihood of this draw (without replacement, approximated as sum)
            ll = np.sum(np.log(prob[draw]))
            total_ll += ll

        # update counts with the current draw for future steps
        counts = update_ewma_counts(counts, draw)

    return total_ll

# =============================================================================
# 6. CALIBRATE α, β, γ BY MAXIMISING CV LOG‑LIKELIHOOD
# =============================================================================
def calibrate_parameters(draws_cal, dates_cal):
    """
    Use L‑BFGS‑B to find α, β, γ that maximise the walk‑forward log‑likelihood.
    """
    # initial guess: α=2.0, β=0, γ=0
    x0 = np.array([2.0, 0.0, 0.0])
    bounds = [(0.5, 5.0),   # α
              (-0.5, 0.5),  # β (temperature sensitivity)
              (-0.5, 0.5)]  # γ (humidity sensitivity)

    # negative log‑likelihood (minimise)
    def objective(params):
        return -cv_log_likelihood(params, draws_cal, dates_cal)

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'disp': False})
    if not result.success:
        print("Warning: optimisation did not converge:", result.message)

    best_params = result.x
    best_alpha, best_beta, best_gamma = best_params
    print(f"\nCalibrated parameters (cross‑validated):")
    print(f"  α (mass sensitivity) = {best_alpha:.3f}")
    print(f"  β (temperature)      = {best_beta:.3f}")
    print(f"  γ (humidity)         = {best_gamma:.3f}")
    print(f"  Cross‑validated log‑likelihood = {-result.fun:.2f}")
    return best_alpha, best_beta, best_gamma

# =============================================================================
# 7. FORWARD MONTE CARLO SIMULATION
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
# 8. MAIN PIPELINE
# =============================================================================
def main():
    draws, dates = load_draws_from_csv(CSV_FILE)
    if len(draws) < HISTORY_LEN + 1:
        print(f"Need at least {HISTORY_LEN+1} draws, only {len(draws)} available.")
        sys.exit(1)

    # Separate calibration set (all draws except the very last one)
    calibration_draws = draws[:-1]
    calibration_dates = dates[:-1]
    last_draw = draws[-1]
    last_date = dates.iloc[-1]
    print(f"Using {len(calibration_draws)} draws for calibration.")
    print(f"Most recent known draw: {last_draw + 1}")

    # --- 1. Calibrate hyperparameters via cross‑validation ---
    print("\n--- Cross‑validated calibration ---")
    best_alpha, best_beta, best_gamma = calibrate_parameters(calibration_draws, calibration_dates)

    # --- 2. Final state estimation using the whole calibration set (EWMA) ---
    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    for draw in calibration_draws:
        counts = update_ewma_counts(counts, draw)

    final_masses = masses_from_counts(counts)
    print("\nFinal estimated masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {final_masses[i]*1000:.3f} g")

    # --- 3. Get environment for the next draw (last known date + 1 day?) ---
    # We'll use the last known date's month as a rough estimate.
    temp_next, hum_next = get_env_from_date(last_date)

    # --- 4. Compute final physics probability vector ---
    prob_vec = physics_probabilities(final_masses, best_alpha,
                                     temp_next, hum_next,
                                     best_beta, best_gamma)
    print("\nFinal physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    # --- 5. Forward simulation ---
    print("\n--- Forward Monte Carlo simulation for next draw ---")
    final_probs, total_sim = simulate_until_convergence(prob_vec)

    # --- 6. Prediction ---
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
