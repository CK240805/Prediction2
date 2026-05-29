#!/usr/bin/env python3
"""
LottOracle – Unknown Ball Sets (colour changes every draw) [FINAL FIX]
=======================================================================
Automatically clusters historical draws into ball set groups.
For each cluster a separate physics model (EWMA + α calibration) is built.
Final prediction = weighted average of per‑cluster probabilities.

Compatible with pandas Timestamp and numpy.datetime64.
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results2.csv"
MIN_HISTORY_FOR_CAL = 104
NUM_BALLS = 49
DRAW_SIZE = 6

BALL_MASS_NOMINAL = 1.0
BALL_MASS_TOL_REL = 0.01

EWMA_DECAY = 0.98
TREND_WINDOW = 20
MAX_K = 5

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
# 2. CLUSTERING DRAWS INTO BALL SETS
# =============================================================================
def cluster_draws(draws, max_k=MAX_K):
    n = len(draws)
    X = np.zeros((n, NUM_BALLS))
    for i, d in enumerate(draws):
        X[i, d] = 1.0
    X = X / X.sum(axis=1, keepdims=True)

    best_bic = np.inf
    best_gmm = None
    best_k = 2
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        print(f"  k = {k} : BIC = {bic:.1f}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k
    labels = best_gmm.predict(X)
    print(f"Selected {best_k} clusters based on BIC.")
    return labels, best_gmm

# =============================================================================
# 3. ENVIRONMENTAL MODEL (NOW HANDLES numpy.datetime64)
# =============================================================================
def get_env_from_date(date):
    # Convert any date type (datetime, Timestamp, numpy.datetime64) to pd.Timestamp
    ts = pd.Timestamp(date)
    month = ts.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 4. EWMA & MASS ESTIMATION
# =============================================================================
def update_ewma_counts(prev_counts, draw, decay=EWMA_DECAY):
    new_counts = decay * prev_counts
    for ball in draw:
        new_counts[ball] += (1 - decay) * (NUM_BALLS / DRAW_SIZE)
    return new_counts

def masses_from_counts(counts, nominal=BALL_MASS_NOMINAL, tol_rel=BALL_MASS_TOL_REL):
    counts = np.maximum(counts, 1e-8)
    freq = counts / counts.sum()
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * nominal
    return np.clip(raw_mass, nominal - tol_rel, nominal + tol_rel)

# =============================================================================
# 5. TREND (optional, not used actively)
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
# 6. PHYSICS PROBABILITY
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
# 7. DIRICHLET‑MULTINOMIAL LOG‑LIKELIHOOD
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
# 8. CROSS‑VALIDATION CALIBRATION (per cluster)
# =============================================================================
def cv_log_likelihood(params, draws_subset, dates_subset, min_history):
    alpha, log_concentration = params
    concentration = np.exp(log_concentration)
    n = len(draws_subset)
    if n <= min_history:
        return -1e12

    counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    total_ll = 0.0
    for i in range(n):
        draw = draws_subset[i]
        if i >= min_history:
            masses = masses_from_counts(counts)
            temp, hum = get_env_from_date(dates_subset[i])
            prob = physics_probabilities(masses, alpha, temp, hum)
            total_ll += dirichlet_multinomial_loglike(prob, draw, concentration)
        counts = update_ewma_counts(counts, draw)
    return total_ll

def calibrate_parameters(draws_subset, dates_subset, min_history=MIN_HISTORY_FOR_CAL):
    if len(draws_subset) < min_history + 1:
        return 2.0, 10.0
    x0 = np.array([2.0, np.log(10.0)])
    bounds = [(0.5, 5.0), (np.log(1), np.log(1000))]
    def objective(params):
        return -cv_log_likelihood(params, draws_subset, dates_subset, min_history)
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 300})
    if not res.success:
        print("    Warning: optimisation did not converge:", res.message)
    alpha, log_conc = res.x
    return alpha, np.exp(log_conc)

# =============================================================================
# 9. MONTE CARLO SET SAMPLING
# =============================================================================
def sample_best_set(prob_vector, num_draws=500_000, top_n=3):
    set_counter = Counter()
    p = prob_vector.astype(np.float64)
    p = p / p.sum()
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
# 10. MAIN PIPELINE
# =============================================================================
def main():
    draws, dates = load_draws_from_csv(CSV_FILE)
    if len(draws) < MIN_HISTORY_FOR_CAL + 1:
        print(f"Need at least {MIN_HISTORY_FOR_CAL+1} draws, got {len(draws)}.")
        sys.exit(1)

    cal_draws = draws[:-1]
    cal_dates = dates[:-1]          # pandas Series
    last_draw = draws[-1]
    last_date = dates.iloc[-1]
    print(f"Using {len(cal_draws)} draws for calibration.")

    print("Clustering draws into ball sets (colours) ...")
    labels, gmm = cluster_draws(cal_draws, max_k=MAX_K)
    n_clusters = gmm.n_components
    cluster_sizes = np.bincount(labels)
    print(f"Cluster sizes: {cluster_sizes}")

    cluster_models = []
    for k in range(n_clusters):
        mask = labels == k
        if cluster_sizes[k] < MIN_HISTORY_FOR_CAL:
            print(f"Cluster {k} too small ({cluster_sizes[k]} draws) – skipping.")
            continue

        draws_k = cal_draws[mask]
        dates_k = cal_dates[mask].values   # NumPy array of Timestamps

        print(f"\n--- Cluster {k} ({cluster_sizes[k]} draws) ---")

        alpha_k, conc_k = calibrate_parameters(draws_k, dates_k)
        print(f"  α = {alpha_k:.3f}, concentration = {conc_k:.1f}")

        counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
        for draw in draws_k:
            counts = update_ewma_counts(counts, draw)
        masses_k = masses_from_counts(counts)
        print("  Masses (first 5):", np.round(masses_k[:5], 4))

        temp_next, hum_next = get_env_from_date(last_date)
        prob_k = physics_probabilities(masses_k, alpha_k, temp_next, hum_next)
        cluster_models.append({
            'prob': prob_k,
            'weight': cluster_sizes[k] / len(cal_draws)
        })

    if not cluster_models:
        print("No cluster has enough data. Falling back to single‑set model.")
        counts = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
        for draw in cal_draws:
            counts = update_ewma_counts(counts, draw)
        masses = masses_from_counts(counts)
        alpha, conc = calibrate_parameters(cal_draws, cal_dates.values)
        temp_next, hum_next = get_env_from_date(last_date)
        prob = physics_probabilities(masses, alpha, temp_next, hum_next)
        cluster_models = [{'prob': prob, 'weight': 1.0}]

    # Weighted average of per‑cluster probabilities
    comb_prob = np.zeros(NUM_BALLS)
    for m in cluster_models:
        comb_prob += m['weight'] * m['prob']
    comb_prob = comb_prob / comb_prob.sum()

    print("\nCombined prediction probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {comb_prob[i]:.4f}")

    print("\n--- Monte Carlo set sampling (500k draws) ---")
    top_sets = sample_best_set(comb_prob, num_draws=500_000, top_n=5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {combo}  (occurred {count} times)")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
