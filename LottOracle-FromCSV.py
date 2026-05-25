#!/usr/bin/env python3
"""
LottOracle-FromCSV
===================
Reads historical 6/49 draw results from a CSV file (draw_no,date,n1,...,n6,additional),
estimates the physical bias of the air-mix machine, and runs a continuous Monte Carlo
simulation of the drawing process until the predicted probabilities converge.
Then outputs the predicted set for the next draw.

Ball specs: 3.3 g, 40 mm diameter.
Physics model: top-selection air-mix → P(ball) ∝ (1/mass)^α.
"""

import numpy as np
import pandas as pd
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_draws.csv"   # <-- change to your filename
HISTORY_LEN = 1189                    # number of past draws used for estimation
ALPHA = 2.0                         # air-mix sensitivity exponent
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6
BALL_MASS_MEAN = 3.3e-3             # kg
BALL_MASS_TOLERANCE = 0.165e-3      # ±5%

# Default environmental conditions (if not available)
DEFAULT_TEMP = 20.0       # °C
DEFAULT_HUMIDITY = 50.0   # %
DEFAULT_PRESSURE = 1013.0 # hPa

# =============================================================================
# 1. LOAD HISTORICAL DRAWS FROM CSV
# =============================================================================
def load_draws_from_csv(filepath):
    """
    Reads CSV with columns: draw_no, date, n1, n2, n3, n4, n5, n6, additional
    Returns: draws (N, 6) as 0-indexed integers, sorted by draw_no/date.
    """
    df = pd.read_csv(filepath)
    # Ensure columns exist
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by draw_no or date (both work)
    df = df.sort_values("draw_no").reset_index(drop=True)

    # Extract numbers (convert to 0-indexed)
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1  # 0..48

    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws

# =============================================================================
# 2. STATE ESTIMATION: INFER BALL MASSES FROM DRAW FREQUENCIES
# =============================================================================
def estimate_ball_masses(draws_history):
    """
    Given an array of past draws (H, 6), 0-indexed,
    estimate the mass of each ball using the inverse frequency rule.
    Lighter balls appear more often in top-selection air-mix.
    Returns: masses (49,) in kg.
    """
    H = len(draws_history)
    counts = np.zeros(NUM_BALLS)
    for draw in draws_history:
        for ball in draw:
            counts[ball] += 1
    # Avoid zero counts
    counts = counts + 1e-6
    freq = counts / (H * DRAW_SIZE)

    # Mass ∝ 1 / freq (lighter → higher freq)
    raw_mass = 1.0 / (freq + 1e-8)
    # Normalize to mean = BALL_MASS_MEAN
    raw_mass = raw_mass / np.mean(raw_mass) * BALL_MASS_MEAN

    # Clip to ±5% tolerance around the mean
    raw_mass = np.clip(raw_mass,
                       BALL_MASS_MEAN - BALL_MASS_TOLERANCE,
                       BALL_MASS_MEAN + BALL_MASS_TOLERANCE)
    return raw_mass

# =============================================================================
# 3. PHYSICS ENGINE: ANALYTICAL AIR-MIX MODEL
# =============================================================================
def compute_physics_probabilities(masses, env):
    """
    Given current ball masses (49,) and environment dict,
    compute the draw probability vector using the air-mix bias.
    P(ball) ∝ (1/mass)^α
    """
    # α is a fixed constant; in a real system it would be calibrated.
    alpha = ALPHA
    weight = (1.0 / masses) ** alpha
    prob = weight / np.sum(weight)
    return prob

# =============================================================================
# 4. CONTINUOUS MONTE CARLO SIMULATION UNTIL CONVERGENCE
# =============================================================================
def simulate_until_convergence(prob_vector,
                               threshold=CONVERGENCE_THRESHOLD,
                               max_iters=MAX_SIM_DRAWS,
                               batch_size=BATCH_SIZE):
    """
    Samples virtual draws from prob_vector (using np.random.choice without
    replacement per draw) until the empirical distribution stabilises.
    Returns: final empirical probabilities, total draws performed.
    """
    counts = np.zeros(NUM_BALLS, dtype=np.int64)
    prev_probs = None
    iteration = 0
    delta = None

    while iteration < max_iters:
        # Sample batch of draws
        batch_draws = []
        for _ in range(batch_size):
            draw = np.random.choice(NUM_BALLS, size=DRAW_SIZE,
                                    replace=False, p=prob_vector)
            batch_draws.append(draw)
        batch_draws = np.array(batch_draws)  # (batch_size, 6)
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
            print(f"  Simulated {total_draws} draws, max delta = {delta:.6f}" if delta else
                  f"  Simulated {total_draws} draws...")

    final_total = (iteration + 1) * batch_size
    if iteration >= max_iters:
        print("Warning: maximum iterations reached without full convergence.")
    else:
        print(f"Converged after {final_total} virtual draws. Final max delta = {delta:.6f}")
    return empirical_probs, final_total

# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================
def main():
    # Load data
    draws = load_draws_from_csv(CSV_FILE)

    if len(draws) < HISTORY_LEN:
        print(f"Error: Need at least {HISTORY_LEN} draws, but only {len(draws)} available.")
        sys.exit(1)

    # Use the last HISTORY_LEN draws for state estimation
    recent_draws = draws[-HISTORY_LEN:]   # shape (HISTORY_LEN, 6)

    # Estimate current ball masses
    estimated_masses = estimate_ball_masses(recent_draws)
    print("\nEstimated ball masses (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {estimated_masses[i]*1000:.3f} g")

    # Environmental conditions (if you have real data, replace these)
    env = {
        "temperature": DEFAULT_TEMP,
        "humidity": DEFAULT_HUMIDITY,
        "pressure": DEFAULT_PRESSURE
    }
    # (Environment affects air density, but we keep it simple for now)

    # Compute physical probability vector
    prob_vec = compute_physics_probabilities(estimated_masses, env)
    print("\nInitial physics-based probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    # Run continuous simulation to refine probabilities
    print("\n--- Starting continuous Monte Carlo simulation ---")
    final_probs, total_simulated = simulate_until_convergence(prob_vec)

    # Predict the next set: top 6 most probable balls
    top_indices = np.argsort(final_probs)[-DRAW_SIZE:][::-1]
    predicted_balls = top_indices + 1  # back to 1-indexed
    predicted_balls.sort()
    print(f"\nPredicted set for next draw: {predicted_balls}")

    print("\nFinal probability distribution (top 10):")
    for rank, idx in enumerate(np.argsort(final_probs)[-10:][::-1]):
        print(f"  {rank+1}. Ball {idx+1:02d}: {final_probs[idx]:.4f}")

    print(f"\nTotal virtual draws performed: {total_simulated}")
    print("Disclaimer: This is an educational demonstration based on a simplified")
    print("physical model. Real lottery outcomes are unpredictable without illegal")
    print("access to the machine's internal state.")

if __name__ == "__main__":
    main()
