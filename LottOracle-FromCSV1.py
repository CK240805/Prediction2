#!/usr/bin/env python3
"""
LottOracle-FromCSV (Refined)
=============================
Reads historical 6/49 draws from CSV (draw_no,date,n1,...,n6,additional),
estimates the physical bias of the air-mix machine with tight ball tolerances,
and runs a Monte Carlo simulation until probabilities converge.
Outputs the predicted next set of numbers.

Ball specs (user‑confirmed):
  - Mass:   3.3 g ± 0.1 g
  - Diameter: 40 mm ± 0.1 mm

Model: top‑selection air‑mix → P(ball) ∝ (1/mass)^α
"""

import numpy as np
import pandas as pd
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"   # <-- your file
HISTORY_LEN = 50
ALPHA = 2.0                         # turbulence sensitivity
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6

# --- NEW BALL SPECS (tight) ---
BALL_MASS_MEAN = 3.3e-3            # kg (3.3 g)
BALL_MASS_TOL  = 0.1e-3            # kg (±0.1 g)
BALL_DIAMETER  = 40.0e-3           # m
BALL_DIAM_TOL  = 0.1e-3            # m (±0.1 mm) – noted for reference

# Environment defaults (if not available)
DEFAULT_TEMP = 20.0
DEFAULT_HUMIDITY = 50.0
DEFAULT_PRESSURE = 1013.0

# =============================================================================
# 1. LOAD HISTORICAL DRAWS
# =============================================================================
def load_draws_from_csv(filepath):
    df = pd.read_csv(filepath)
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.sort_values("draw_no").reset_index(drop=True)
    draws = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values - 1
    print(f"Loaded {len(draws)} draws from {filepath}")
    return draws

# =============================================================================
# 2. MASS ESTIMATION FROM FREQUENCIES (clipped to new tolerance)
# =============================================================================
def estimate_ball_masses(draws_history):
    """
    Infer ball masses from occurrence frequencies.
    Heavier → less likely in top‑selection air‑mix.
    """
    H = len(draws_history)
    counts = np.zeros(NUM_BALLS)
    for draw in draws_history:
        for ball in draw:
            counts[ball] += 1
    counts = counts + 1e-6
    freq = counts / (H * DRAW_SIZE)

    # Inverse mapping: mass ∝ 1/freq
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * BALL_MASS_MEAN

    # Clip to user‑specified ±0.1 g
    raw_mass = np.clip(raw_mass,
                       BALL_MASS_MEAN - BALL_MASS_TOL,
                       BALL_MASS_MEAN + BALL_MASS_TOL)
    return raw_mass

# =============================================================================
# 3. PHYSICS ENGINE: ANALYTICAL AIR‑MIX PROBABILITY
# =============================================================================
def compute_physics_probabilities(masses, env):
    """
    P(ball) ∝ (1/mass)^α.
    env is passed for future use (air density, charge), but not yet implemented.
    """
    alpha = ALPHA
    weight = (1.0 / masses) ** alpha
    prob = weight / np.sum(weight)
    return prob

# =============================================================================
# 4. CONTINUOUS MONTE CARLO SIMULATION
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
            print(f"  Simulated {total_draws} draws, max delta = {delta:.6f}" if delta else
                  f"  Simulated {total_draws} draws...")

    final_total = (iteration + 1) * batch_size
    if iteration >= max_iters:
        print("Warning: max iterations reached without full convergence.")
    else:
        print(f"Converged after {final_total} virtual draws. Final delta = {delta:.6f}")
    return empirical_probs, final_total

# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================
def main():
    draws = load_draws_from_csv(CSV_FILE)
    if len(draws) < HISTORY_LEN:
        print(f"Need at least {HISTORY_LEN} draws, only {len(draws)} available.")
        sys.exit(1)

    recent_draws = draws[-HISTORY_LEN:]

    # Estimate masses (now clipped to ±0.1 g)
    masses = estimate_ball_masses(recent_draws)
    print("\nEstimated ball masses (first 10, in grams):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {masses[i]*1000:.3f} g")

    env = {"temperature": DEFAULT_TEMP, "humidity": DEFAULT_HUMIDITY,
           "pressure": DEFAULT_PRESSURE}

    prob_vec = compute_physics_probabilities(masses, env)
    print("\nInitial physics probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_vec[i]:.4f}")

    print("\n--- Continuous Monte Carlo simulation ---")
    final_probs, total_sim = simulate_until_convergence(prob_vec)

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
