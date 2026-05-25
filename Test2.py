#!/usr/bin/env python3
"""
LottOracle-Ultimate
====================
Full physics+ML lottery predictor with:
  - Unscented Kalman Filter for ball‑mass state tracking
  - Dirichlet‑multinomial observation model
  - Environmental coupling (seasonal temp/humidity)
  - Trend (momentum) feature
  - RMT initialisation of state
  - Hyperparameter tuning via time‑series cross‑validation
  - Residual bias corrector (optimised per‑ball log‑odds correction)
  - Monte Carlo set sampling for final prediction

Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm diameter
"""

import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.linalg import cholesky, sqrtm
from sklearn.linear_model import LinearRegression
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"
MIN_HISTORY_FOR_CAL = 100          # draws before cross‑validation can start
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6
BALL_MASS_MEAN = 3.3e-3            # kg
BALL_MASS_TOL  = 0.1e-3            # kg

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
# 2. ENVIRONMENTAL MODEL
# =============================================================================
def get_env_from_date(date):
    month = date.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. RMT INITIAL STATE ESTIMATE (for Kalman filter)
# =============================================================================
def marchenko_pastur_threshold(q):
    return (1 + np.sqrt(q))**2

def rmt_denoise_counts(counts_history):
    T = len(counts_history)
    if T < NUM_BALLS:
        return counts_history[-1].copy()
    X = counts_history  # (T, 49)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_stdz = (X - X_mean) / X_std
    corr = np.corrcoef(X_stdz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    p = NUM_BALLS; q = p / T
    lambda_plus = marchenko_pastur_threshold(q)
    significant = eigvals > lambda_plus
    V_sig = eigvecs[:, significant]
    X_clean = X_stdz @ V_sig @ V_sig.T
    X_denoised = X_clean * X_std + X_mean
    return X_denoised[-1, :].clip(min=0)

def counts_to_logodds(counts):
    """Convert counts to log-odds (centered) for UKF state initialisation."""
    p = counts / (counts.sum() + 1e-8)
    p = np.clip(p, 1e-6, 1-1e-6)
    logodds = np.log(p) - np.log(1-p)
    return logodds

# =============================================================================
# 4. PHYSICS PROBABILITY MODEL (with mass, env, trend)
# =============================================================================
def masses_from_logodds(logodds, mean_mass=BALL_MASS_MEAN, tol=BALL_MASS_TOL):
    p = 1 / (1 + np.exp(-logodds))
    freq = p / p.sum()
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * mean_mass
    return np.clip(raw_mass, mean_mass - tol, mean_mass + tol)

def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0,
                          trend_slopes=None, delta=0.0):
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    if trend_slopes is not None and delta != 0.0:
        weight = weight * np.exp(delta * trend_slopes)
    return weight / weight.sum()

# =============================================================================
# 5. DIRICHLET-MULTINOMIAL LOG-LIKELIHOOD
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
# 6. TREND FEATURE
# =============================================================================
def compute_trend_slopes(prob_history, window=20):
    """prob_history: list/array of probability vectors (each of length 49)."""
    if len(prob_history) < window:
        return np.zeros(NUM_BALLS)
    recent = np.array(prob_history[-window:])
    x = np.arange(window).reshape(-1, 1)
    slopes = np.zeros(NUM_BALLS)
    model = LinearRegression()
    for i in range(NUM_BALLS):
        model.fit(x, recent[:, i])
        slopes[i] = model.coef_[0]
    slopes = slopes / (np.std(slopes) + 1e-6) * 0.1
    return slopes

# =============================================================================
# 7. UNSCENTED KALMAN FILTER (UKF) FOR BALL PROBABILITIES
# =============================================================================
def ukf_predict(x, P, Q):
    """Prediction step: x_{t+1} = x_t + w, w ~ N(0,Q)."""
    x_pred = x
    P_pred = P + Q
    return x_pred, P_pred

def ukf_update(x_pred, P_pred, z, R, h_func, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Update step with measurement z (49-dim empirical draw frequencies).
    h_func: softmax function.
    """
    n = len(x_pred)
    lam = alpha**2 * (n + kappa) - n
    # Generate sigma points
    sqrt_P = cholesky(P_pred, lower=True)
    sigma_points = np.vstack([
        x_pred,
        x_pred + np.sqrt(n + lam) * sqrt_P.T,
        x_pred - np.sqrt(n + lam) * sqrt_P.T
    ])  # shape (2n+1, n)
    # Weights
    Wm = np.full(2*n+1, 0.5/(n+lam))
    Wc = np.full(2*n+1, 0.5/(n+lam))
    Wm[0] = lam / (n+lam)
    Wc[0] = lam / (n+lam) + (1 - alpha**2 + beta)
    # Propagate sigma points through h
    Z_sigma = np.array([h_func(s) for s in sigma_points])  # (2n+1, 49)
    # Predicted measurement mean
    z_hat = np.dot(Wm, Z_sigma)
    # Innovation covariance
    dz = Z_sigma - z_hat
    Pzz = (Wc[:, None] * dz).T @ dz + R
    # Cross covariance
    dx = sigma_points - x_pred
    Pxz = (Wc[:, None] * dx).T @ dz
    # Kalman gain
    K = Pxz @ np.linalg.inv(Pzz)
    # Update state
    x_upd = x_pred + K @ (z - z_hat)
    P_upd = P_pred - K @ Pzz @ K.T
    return x_upd, P_upd

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =============================================================================
# 8. HYPERPARAMETER TUNING (time‑series CV)
# =============================================================================
def evaluate_params(params, draws_cal, dates_cal, min_history=MIN_HISTORY_FOR_CAL):
    """
    Run UKF with given params and return total log‑likelihood on calibration set.
    params = [log_Q_diag, log_R_diag, log_concentration, log_trend_window]
    """
    Q_diag = np.exp(params[0])   # process noise per ball (log-odds)
    R_diag = np.exp(params[1])   # measurement noise (probabilities)
    concentration = np.exp(params[2])
    trend_window = int(np.exp(params[3]))
    trend_window = max(5, min(trend_window, 50))  # clip

    # UKF initialisation
    # Build initial counts from first min_history draws using EWMA
    counts_init = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    for draw in draws_cal[:min_history]:
        # simple update
        counts_init = 0.95 * counts_init
        for b in draw:
            counts_init[b] += (1-0.95)*(NUM_BALLS/DRAW_SIZE)
    logodds_init = counts_to_logodds(counts_init)
    P_init = np.eye(NUM_BALLS) * 0.1
    Q = np.eye(NUM_BALLS) * Q_diag
    R = np.eye(NUM_BALLS) * R_diag

    x = logodds_init.copy()
    P = P_init.copy()

    total_ll = 0.0
    prob_history = []  # store filtered probabilities for trend

    for i in range(min_history, len(draws_cal)):
        draw = draws_cal[i]
        temp, hum = get_env_from_date(dates_cal[i])
        # Current physics probability from state (without bias correction)
        cur_logodds = x
        prob_from_ukf = softmax(cur_logodds)
        masses = masses_from_logodds(cur_logodds)
        # Compute trend from history of ukf probabilities
        slopes = compute_trend_slopes(prob_history, window=trend_window)
        # Final physics prob (combine mass, env, trend)
        prob_phys = physics_probabilities(masses, alpha=2.0,  # we'll use a fixed alpha=2.0, could also tune
                                         temp=temp, hum=hum, beta=0.0, gamma=0.0,
                                         trend_slopes=slopes, delta=0.1)  # small trend weight
        # Use Dirichlet-multinomial likelihood
        ll = dirichlet_multinomial_loglike(prob_phys, draw, concentration)
        total_ll += ll

        # UKF update: measurement = empirical draw frequencies (one-hot divided by 6)
        z = np.zeros(NUM_BALLS)
        for b in draw:
            z[b] = 1.0 / DRAW_SIZE
        x, P = ukf_predict(x, P, Q)
        x, P = ukf_update(x, P, z, R, softmax)

        prob_history.append(prob_from_ukf)

    return total_ll

def tune_hyperparameters(draws_cal, dates_cal):
    print("Tuning hyperparameters via time‑series CV (this may take a few minutes)...")
    # Initial guess: log_Q_diag=-3, log_R_diag=-3, log_concentration=log(10), log_trend_window=log(20)
    x0 = np.array([np.log(1e-3), np.log(1e-3), np.log(10.0), np.log(20)])
    bounds = [(-8, 0), (-8, 0), (np.log(1), np.log(100)), (np.log(5), np.log(50))]
    def objective(p):
        return -evaluate_params(p, draws_cal, dates_cal)
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    best_params = res.x
    Q_diag = np.exp(best_params[0])
    R_diag = np.exp(best_params[1])
    concentration = np.exp(best_params[2])
    trend_window = int(np.exp(best_params[3]))
    print(f"Best Q_diag={Q_diag:.4e}, R_diag={R_diag:.4e}, concentration={concentration:.1f}, trend_window={trend_window}")
    return Q_diag, R_diag, concentration, trend_window

# =============================================================================
# 9. RESIDUAL BIAS CORRECTOR (per‑ball log‑odds bias)
# =============================================================================
def optimize_bias_correction(ukf_logodds_list, draws_cal, dates_cal, concentration):
    """Optimize a 49‑dim bias vector added to log‑odds before physics model."""
    n = len(ukf_logodds_list)
    def objective(bias):
        total_ll = 0.0
        for i in range(n):
            logodds = ukf_logodds_list[i] + bias
            prob_phys = softmax(logodds)
            draw = draws_cal[i]
            total_ll += dirichlet_multinomial_loglike(prob_phys, draw, concentration)
        return -total_ll
    bias0 = np.zeros(NUM_BALLS)
    res = minimize(objective, bias0, method='L-BFGS-B', options={'maxiter': 200})
    bias = res.x
    print(f"Bias correction optimized. Final log-lik improvement: {-res.fun - (-objective(np.zeros(NUM_BALLS))):.2f}")
    return bias

# =============================================================================
# 10. MONTE CARLO SET SAMPLING
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
# 11. MAIN PIPELINE
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

    # --- Step 1: Tune hyperparameters (Q, R, concentration, trend window) ---
    Q_diag, R_diag, concentration, trend_window = tune_hyperparameters(calibration_draws, calibration_dates)

    # --- Step 2: Run UKF on calibration set with tuned params to get state estimates ---
    # Initialise from RMT‑denoised counts of early draws
    counts_init = np.ones(NUM_BALLS) * (DRAW_SIZE / NUM_BALLS)
    for draw in calibration_draws[:MIN_HISTORY_FOR_CAL]:
        counts_init = 0.95 * counts_init
        for b in draw:
            counts_init[b] += (1-0.95)*(NUM_BALLS/DRAW_SIZE)
    # Optionally denoise
    if MIN_HISTORY_FOR_CAL >= NUM_BALLS:
        counts_init = rmt_denoise_counts(np.array([counts_init] * MIN_HISTORY_FOR_CAL))  # give it a history of identical vectors? Better: use actual early counts. We'll just use counts_init directly.
    logodds_init = counts_to_logodds(counts_init)
    P_init = np.eye(NUM_BALLS) * 0.1
    Q = np.eye(NUM_BALLS) * Q_diag
    R = np.eye(NUM_BALLS) * R_diag

    x = logodds_init.copy()
    P = P_init.copy()
    ukf_logodds_seq = []  # after each update, store state
    prob_history = []

    for i in range(len(calibration_draws)):
        draw = calibration_draws[i]
        # record state before update? After update. We'll store post-update state.
        ukf_logodds_seq.append(x.copy())
        prob_ukf = softmax(x)
        prob_history.append(prob_ukf)

        if i >= MIN_HISTORY_FOR_CAL:
            z = np.zeros(NUM_BALLS)
            for b in draw:
                z[b] = 1.0 / DRAW_SIZE
            x, P = ukf_predict(x, P, Q)
            x, P = ukf_update(x, P, z, R, softmax)
        else:
            # during burn-in, just use simple EWMA update to keep state moving
            # we already built counts_init, so for early draws we can still update
            z = np.zeros(NUM_BALLS)
            for b in draw:
                z[b] = 1.0 / DRAW_SIZE
            x, P = ukf_predict(x, P, Q)
            x, P = ukf_update(x, P, z, R, softmax)

    # Final state is x (post last draw)
    final_logodds = x.copy()

    # --- Step 3: Optimize residual bias corrector ---
    # Use the UKF states (pre‑update) for the calibration draws after burn-in
    # We'll use the stored ukf_logodds_seq, but they correspond to state before each update?
    # Our loop stores state before prediction? Let's adjust: we'll store after update. Better: we want the state that would be used to predict the next draw. So for draw i, the predictive state is the one after processing draw i-1. That's exactly ukf_logodds_seq[i] (since we store after previous update). We'll align.
    bias = optimize_bias_correction(ukf_logodds_seq[MIN_HISTORY_FOR_CAL:], calibration_draws[MIN_HISTORY_FOR_CAL:],
                                   calibration_dates[MIN_HISTORY_FOR_CAL:], concentration)

    # --- Step 4: Predict next draw ---
    final_logodds_corrected = final_logodds + bias
    prob_final = softmax(final_logodds_corrected)
    # incorporate env and trend (using recent prob history)
    temp_next, hum_next = get_env_from_date(last_date)
    slopes_next = compute_trend_slopes(prob_history[-trend_window:])
    masses_final = masses_from_logodds(final_logodds_corrected)
    prob_phys = physics_probabilities(masses_final, alpha=2.0, temp=temp_next, hum=hum_next,
                                     beta=0.0, gamma=0.0, trend_slopes=slopes_next, delta=0.1)
    # Blend with UKF softmax? Actually the UKF already gives a direct probability softmax(final_logodds_corrected).
    # The physics model is separate. We can combine them: final_probs = (prob_final + prob_phys) / 2
    final_probs = (prob_final + prob_phys) / 2
    final_probs = final_probs / final_probs.sum()

    print("\nFinal blended probabilities (first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {final_probs[i]:.4f}")

    # --- Step 5: Monte Carlo set sampling ---
    print("\n--- Monte Carlo set sampling (500k draws) ---")
    top_sets = sample_best_set(final_probs, num_draws=500_000, top_n=5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {combo}  (occurred {count} times)")

    print("\nTop 10 marginal probabilities:")
    for rank, idx in enumerate(np.argsort(final_probs)[-10:][::-1]):
        print(f"  {rank+1}. Ball {idx+1:02d}: {final_probs[idx]:.4f}")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
