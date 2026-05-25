#!/usr/bin/env python3
"""
LottOracle-Kalman-Pair-Neural
===============================
Advanced 6/49 lottery prediction with:
- Kalman filter for dynamic ball mass tracking
- Pairwise co‑occurrence bias matrix (symmetric)
- Neural residual corrector (trained on historical residuals)
- All previous features retained (RMT, trend, DM likelihood, set sampling)

Ball specs: 3.3 g ± 0.1 g, 40 mm ± 0.1 mm
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
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = "toto_results.csv"
MIN_HISTORY_FOR_CAL = 100
CONVERGENCE_THRESHOLD = 1e-4
MAX_SIM_DRAWS = 200_000
BATCH_SIZE = 10_000
NUM_BALLS = 49
DRAW_SIZE = 6
BALL_MASS_MEAN = 3.3e-3          # kg
BALL_MASS_TOL  = 0.1e-3          # kg
TREND_WINDOW = 20

# Kalman filter parameters
Q_FACTOR = 1e-12         # process noise variance per draw for mass drift
R_FACTOR = 1e-4          # measurement noise variance (count observations)

# Neural corrector config
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 20

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
# 2. ENVIRONMENT
# =============================================================================
def get_env_from_date(date):
    month = date.month
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * (month - 1) / 12)
    hum = 60.0 + 15.0 * np.cos(2 * np.pi * (month - 1) / 12)
    return temp, hum

# =============================================================================
# 3. KALMAN FILTER FOR BALL MASS TRACKING
# =============================================================================
class KalmanMassTracker:
    """Tracks the mass of 49 balls with a simple Kalman filter."""
    def __init__(self, initial_mass=BALL_MASS_MEAN, mass_tol=BALL_MASS_TOL,
                 Q_factor=Q_FACTOR, R_factor=R_FACTOR):
        self.n = NUM_BALLS
        self.mass_tol = mass_tol
        # State: estimated masses (49,)
        self.masses = np.full(self.n, initial_mass)
        self.cov = np.eye(self.n) * (mass_tol / 3.0)**2   # initial uncertainty
        self.Q = np.eye(self.n) * Q_factor                # process noise
        self.R = np.eye(self.n) * R_factor                # measurement noise (counts)

    def update(self, draw):
        """
        draw: array of 6 ball indices (0‑indexed).
        Observation: we observe a frequency estimate from this single draw
        (very noisy). We use a pseudo‑measurement: count vector / 6.
        """
        # Pseudo measurement: m_meas ∝ 1/(counts+1)
        counts = np.bincount(draw, minlength=self.n) + 1e-6
        freq = counts / counts.sum()
        # Convert to mass estimate (same inverse mapping used in masses_from_counts)
        obs_mass = 1.0 / (freq + 1e-8)
        obs_mass = obs_mass / np.mean(obs_mass) * BALL_MASS_MEAN
        obs_mass = np.clip(obs_mass, BALL_MASS_MEAN - self.mass_tol, BALL_MASS_MEAN + self.mass_tol)

        # Kalman update
        # Predict (process model: mass stays same, but add Q)
        x_pred = self.masses
        P_pred = self.cov + self.Q
        # Measurement update
        y = obs_mass - x_pred
        S = P_pred + self.R
        K = P_pred @ np.linalg.inv(S)
        self.masses = x_pred + K @ y
        self.cov = (np.eye(self.n) - K) @ P_pred
        # Clip to tolerance
        self.masses = np.clip(self.masses, BALL_MASS_MEAN - self.mass_tol,
                              BALL_MASS_MEAN + self.mass_tol)

    def get_masses(self):
        return self.masses.copy()

# =============================================================================
# 4. MASS FROM COUNTS (for initialisation outside Kalman)
# =============================================================================
def masses_from_counts(counts, mean_mass=BALL_MASS_MEAN, tol=BALL_MASS_TOL):
    counts = np.maximum(counts, 1e-8)
    freq = counts / counts.sum()
    raw_mass = 1.0 / (freq + 1e-8)
    raw_mass = raw_mass / np.mean(raw_mass) * mean_mass
    return np.clip(raw_mass, mean_mass - tol, mean_mass + tol)

# =============================================================================
# 5. RMT DENOISER (same as before)
# =============================================================================
def marchenko_pastur_threshold(q):
    return (1 + np.sqrt(q))**2

def rmt_denoise_counts(counts_history):
    T = len(counts_history)
    if T < NUM_BALLS:
        return counts_history[-1].copy()
    X = counts_history
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_stdz = (X - X_mean) / X_std
    corr = np.corrcoef(X_stdz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    p = NUM_BALLS
    q = p / T
    lambda_plus = marchenko_pastur_threshold(q)
    significant = eigvals > lambda_plus
    V_sig = eigvecs[:, significant]
    X_clean = X_stdz @ V_sig @ V_sig.T
    X_denoised = X_clean * X_std + X_mean
    return X_denoised[-1, :].clip(min=0)

# =============================================================================
# 6. TREND
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
# 7. PAIRWISE BIAS MATRIX
# =============================================================================
class PairwiseBias:
    """Symmetric 49x49 matrix of log‑bias for co‑occurrence."""
    def __init__(self):
        # We store a lower‑triangular matrix and enforce symmetry
        self.params = np.zeros((NUM_BALLS, NUM_BALLS))

    def set_zero(self):
        self.params[:] = 0.0

    def update_from_draw(self, draw, learning_rate=0.01):
        """Hebbian‑like update: reinforce pairs that occur together, suppress others."""
        # draw: array of 6 indices
        for i in range(DRAW_SIZE):
            for j in range(i+1, DRAW_SIZE):
                a, b = draw[i], draw[j]
                if a != b:
                    self.params[a, b] += learning_rate
                    self.params[b, a] = self.params[a, b]  # symmetry

    def get_log_bias(self):
        return self.params  # we use this as log‑bias directly

# =============================================================================
# 8. PROBABILITY MODEL (with pairwise bias)
# =============================================================================
def physics_probabilities(masses, alpha, temp, hum, beta=0.0, gamma=0.0,
                          trend_slopes=None, delta=0.0, pair_log_bias=None, pair_weight=0.0):
    weight = (1.0 / masses) ** alpha
    env_factor = np.exp(beta * (temp - 25.0) + gamma * (hum - 60.0))
    weight = weight * env_factor
    if trend_slopes is not None and delta != 0.0:
        weight = weight * np.exp(delta * trend_slopes)
    prob = weight / weight.sum()
    return prob

# We'll incorporate pairwise bias into the final probability vector via an adjustment
# applied after the physics probabilities: prob_adj = softmax(log(prob) + pair_weight * pair_log_bias_row)
# For a set of 6, the joint probability is complex, so we use it at the marginal level.

def apply_pairwise_bias(prob, pair_log_bias, pair_weight, drawn_set=None):
    """Modify marginal probabilities based on average pairwise bias with balls already drawn."""
    # We use the expected bias from all possible partners (mean of log_bias over all other balls)
    if pair_log_bias is None or pair_weight == 0.0:
        return prob
    log_prob = np.log(prob + 1e-10)
    # Average bias for each ball: mean of its row in pair_log_bias
    bias_avg = pair_log_bias.mean(axis=1)  # (49,)
    log_prob = log_prob + pair_weight * bias_avg
    prob_adj = np.exp(log_prob - logsumexp(log_prob))
    return prob_adj

# =============================================================================
# 9. DIRICHLET‑MULTINOMIAL LIKELIHOOD
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
# 10. NEURAL CORRECTOR (trained on residuals)
# =============================================================================
class ResidualCorrector(nn.Module):
    def __init__(self, input_dim=49, hidden=HIDDEN_DIM, output_dim=49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        # x: raw physics logits or probabilities
        return self.net(x)

def train_corrector(physics_logits_list, target_counts_list):
    """
    physics_logits_list: list of log probability vectors (49,) for each historical draw
    target_counts_list: list of bincount vectors (49,) for the actual draws
    Returns trained model.
    """
    device = torch.device("cpu")
    model = ResidualCorrector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    # Convert lists to tensors
    X = torch.tensor(np.array(physics_logits_list), dtype=torch.float32)
    Y = torch.tensor(np.array(target_counts_list), dtype=torch.float32)  # counts
    # We want to predict the residual between target counts and physics probabilities (scaled)
    # We'll train to output a correction to the logits such that softmax matches target counts.
    # Simpler: train to directly output probabilities that match target frequencies.
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X)  # logits
        prob_output = torch.softmax(output, dim=1)
        target_prob = Y / Y.sum(dim=1, keepdim=True)
        loss = criterion(prob_output, target_prob)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    model.eval()
    return model

# =============================================================================
# 11. CROSS‑VALIDATION LOG‑LIKELIHOOD (integrated)
# =============================================================================
def cv_log_likelihood(params, draws_cal, dates_cal, min_history=MIN_HISTORY_FOR_CAL,
                      pair_weight=0.0, use_neural_corrector=False, corrector=None):
    alpha, beta, gamma, delta, log_concentration = params
    concentration = np.exp(log_concentration)
    n_draws = len(draws_cal)
    if n_draws <= min_history:
        return -1e12

    # Initialize Kalman tracker
    tracker = KalmanMassTracker()
    # For pairwise bias, we'll learn it online but not include in calibration likelihood for simplicity
    # (pairwise bias would need a separate meta‑parameter; we'll skip for now and add after calibration)
    # For neural corrector, we'll train it after calibration using residuals.

    total_ll = 0.0
    masses_history = []  # store mass vectors for training corrector later

    for i in range(n_draws):
        draw = draws_cal[i]
        if i < 10:
            # Initial warm‑up: update tracker with draws before calculating
            tracker.update(draw)
            masses_history.append(tracker.get_masses())
            continue

        masses = tracker.get_masses()
        masses_history.append(masses)
        temp, hum = get_env_from_date(dates_cal[i])
        prob = physics_probabilities(masses, alpha, temp, hum, beta, gamma,
                                     trend_slopes=None, delta=delta)
        total_ll += dirichlet_multinomial_loglike(prob, draw, concentration)
        # Update tracker with this draw
        tracker.update(draw)

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
    print(f"  α = {alpha:.3f}, β = {beta:.3f}, γ = {gamma:.3f}, δ = {delta:.3f}, conc = {np.exp(log_conc):.1f}")
    return alpha, beta, gamma, delta, np.exp(log_conc)

# =============================================================================
# 12. FULL PIPELINE
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

    # --- 1. Calibrate core parameters ---
    alpha, beta, gamma, delta, concentration = calibrate_parameters(cal_draws, cal_dates)

    # --- 2. Build Kalman state from all calibration draws and collect mass histories ---
    tracker = KalmanMassTracker()
    mass_history = []  # store mass vectors
    for draw in cal_draws:
        tracker.update(draw)
        mass_history.append(tracker.get_masses())

    # Final masses from Kalman
    final_masses = tracker.get_masses()

    # --- 3. Train pairwise bias matrix from calibration data ---
    pair_bias = PairwiseBias()
    for draw in cal_draws:
        pair_bias.update_from_draw(draw, learning_rate=0.001)
    pair_log_bias = pair_bias.get_log_bias()

    # --- 4. Train neural corrector on residuals ---
    # We'll use mass_history and actual draws to compute physics probabilities and targets
    physics_logits_list = []
    target_counts_list = []
    for i, masses_i in enumerate(mass_history):
        draw_i = cal_draws[i]
        temp_i, hum_i = get_env_from_date(cal_dates[i])
        prob_i = physics_probabilities(masses_i, alpha, temp_i, hum_i, beta, gamma,
                                       trend_slopes=None, delta=delta)
        # Also apply pairwise bias (with a small weight) to get a better baseline
        prob_i_adj = apply_pairwise_bias(prob_i, pair_log_bias, pair_weight=0.01)
        physics_logits_list.append(np.log(prob_i_adj + 1e-10))
        target_counts_list.append(np.bincount(draw_i, minlength=NUM_BALLS).astype(float))
    corrector_model = train_corrector(physics_logits_list, target_counts_list)

    # --- 5. Final prediction for next draw ---
    # Use Kalman masses (final), with environment from last_date
    temp_next, hum_next = get_env_from_date(last_date)
    prob_physics = physics_probabilities(final_masses, alpha, temp_next, hum_next,
                                        beta, gamma, trend_slopes=None, delta=delta)
    # Apply pairwise bias
    prob_w_pair = apply_pairwise_bias(prob_physics, pair_log_bias, pair_weight=0.01)

    # Apply neural corrector
    corrector_model.eval()
    with torch.no_grad():
        logits_in = torch.tensor(np.log(prob_w_pair + 1e-10), dtype=torch.float32).unsqueeze(0)
        corrected_logits = corrector_model(logits_in).squeeze().numpy()
    prob_corrected = np.exp(corrected_logits - logsumexp(corrected_logits))

    print("\nFinal probabilities (with pair bias + neural corrector, first 10):")
    for i in range(10):
        print(f"  Ball {i+1:02d}: {prob_corrected[i]:.4f}")

    # --- 6. Monte Carlo set sampling ---
    print("\n--- Monte Carlo set sampling (500k draws) ---")
    set_counter = Counter()
    p = prob_corrected.astype(np.float64)
    batch_sz = 50000
    for batch_start in range(0, 500_000, batch_sz):
        actual = min(batch_sz, 500_000 - batch_start)
        draws_batch = np.array([np.random.choice(NUM_BALLS, size=DRAW_SIZE, replace=False, p=p)
                                for _ in range(actual)])
        for d in draws_batch:
            set_counter[tuple(np.sort(d))] += 1
    top_sets = set_counter.most_common(5)
    print("Predicted next set (most frequent combination):")
    for combo, count in top_sets:
        print(f"  {np.array(combo)+1}  (occurred {count} times)")

    print("\nDisclaimer: educational demo. Real lottery outcomes remain unpredictable.")

if __name__ == "__main__":
    main()
