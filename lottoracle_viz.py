#!/usr/bin/env python3
"""
LottOracle-Inferred-Viz: Reads CSV, infers ball masses via air‑mix physics,
outputs charts for GitHub rendering.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# =============================================================================
# Physical constants (3.3 g, 40 mm)
# =============================================================================
BALL_MASS_MEAN  = 3.3e-3
BALL_MASS_TOL   = 0.165e-3
BALL_DIAMETER   = 40.0e-3
BALL_RADIUS     = BALL_DIAMETER / 2.0
BALL_CROSS_SEC  = np.pi * BALL_RADIUS**2
AIR_DENSITY     = 1.225
DRAG_COEFF      = 0.50
GRAVITY         = 9.81

CONFIG = {
    "num_balls": 49,
    "draw_size": 6,
    "history_len": 50,
    "rmt_window": 30,
    "d_model": 256,
    "nhead": 8,
    "num_transformer_layers": 6,
    "physics_state_dim": 128,
    "env_feat_dim": 8,
    "num_gaussians": 5,
    "batch_size": 64,
    "epochs": 5,
    "lr": 3e-4,
    "rmt_out_dim": 64,
    "gp_drift_dim": 32,
    "chaos_dim": 1,
}

# =============================================================================
# Dataset (unchanged)
# =============================================================================
class CSVDataset(Dataset):
    def __init__(self, csv_path, num_draws=None):
        df = pd.read_csv(csv_path)
        df = df.sort_values("draw_no", ascending=True).reset_index(drop=True)
        if num_draws is not None:
            df = df.tail(num_draws)
        draws = df[["n1","n2","n3","n4","n5","n6"]].values - 1
        self.draws = draws.astype(np.int64)

    def __len__(self):
        return len(self.draws) - CONFIG["history_len"]

    def __getitem__(self, idx):
        hist_draws = self.draws[idx : idx + CONFIG["history_len"]]
        next_draw  = self.draws[idx + CONFIG["history_len"]]
        hist_tokens = torch.tensor(hist_draws.flatten() + 1, dtype=torch.long)
        return hist_tokens, torch.tensor(next_draw, dtype=torch.long)

# =============================================================================
# Physics engine & model (unchanged)
# =============================================================================
class AirMixPhysicsEngine(nn.Module):
    def __init__(self, num_balls=49):
        super().__init__()
        self.raw_masses = nn.Parameter(torch.zeros(num_balls))
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self):
        masses = BALL_MASS_MEAN + BALL_MASS_TOL * torch.tanh(self.raw_masses)
        alpha = F.softplus(self.alpha) + 0.1
        weight = 1.0 / (masses ** alpha + 1e-12)
        probs = weight / weight.sum()
        return masses, probs

# (include all model modules: RMT, GP, Chaos, SetTransformer, LottOracleInverted)
# For brevity, I'll just import the complete model from our previous version.
# In practice, copy the full model class from the previous message.

class RMTCollectiveModes(nn.Module):
    def __init__(self, num_balls=49, rmt_window=30, out_dim=64):
        super().__init__()
        self.num_balls = num_balls
        self.rmt_window = rmt_window
        self.out_dim = out_dim
        self.proj = nn.Linear(num_balls * out_dim, out_dim)

    def forward(self, hist_tokens):
        B = hist_tokens.size(0)
        draws = hist_tokens.view(B, CONFIG["history_len"], CONFIG["draw_size"]) - 1
        draws = draws[:, -self.rmt_window:, :]
        multi_hot = torch.zeros(B, self.rmt_window, self.num_balls, device=hist_tokens.device)
        for w in range(self.rmt_window):
            for s in range(CONFIG["draw_size"]):
                multi_hot[:, w, draws[:, w, s]] = 1.0
        X = multi_hot.permute(0, 2, 1)
        X_mean = X.mean(dim=-1, keepdim=True)
        Xc = X - X_mean
        C = torch.bmm(Xc, Xc.transpose(1,2)) / (self.rmt_window - 1)
        _, eigenvectors = torch.linalg.eigh(C)
        q = self.rmt_window / self.num_balls
        lambda_plus = (1 + np.sqrt(q))**2
        top_k = min(self.out_dim, self.num_balls)
        sel_eigvecs = eigenvectors[:, :, -top_k:]
        flat = sel_eigvecs.reshape(B, -1)
        return self.proj(flat)

class HierarchicalDriftTracker(nn.Module):
    def __init__(self, phys_dim=98, latent_dim=32):
        super().__init__()
        self.rnn = nn.GRU(phys_dim, 128, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(128, latent_dim)

    def forward(self, phys_seq):
        _, h_n = self.rnn(phys_seq)
        return self.output_proj(h_n[-1])

class ChaosPredictabilityGauge(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(CONFIG["num_balls"]+1, d_model, padding_idx=0)
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, hist_tokens):
        B = hist_tokens.size(0)
        tokens = hist_tokens.view(B, CONFIG["history_len"], CONFIG["draw_size"])
        emb = self.embed(tokens)
        draw_vecs = emb.mean(dim=2)
        attn_out, _ = self.attn(draw_vecs, draw_vecs, draw_vecs)
        return self.fc(attn_out.mean(dim=1))

class SetTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class LottOracleInverted(nn.Module):
    def __init__(self):
        super().__init__()
        self.physics = AirMixPhysicsEngine(CONFIG["num_balls"])
        self.physics_encoder = nn.Sequential(
            nn.Linear(CONFIG["num_balls"]*2, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["physics_state_dim"]),
            nn.LayerNorm(CONFIG["physics_state_dim"])
        )
        self.env_proj = nn.Linear(CONFIG["env_feat_dim"], CONFIG["d_model"])
        self.ball_embed = nn.Embedding(CONFIG["num_balls"]+1, CONFIG["d_model"], padding_idx=0)
        self.draw_encoder = SetTransformerEncoder(CONFIG["d_model"], CONFIG["nhead"])
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
                                       batch_first=True, dim_feedforward=512),
            num_layers=CONFIG["num_transformer_layers"]
        )
        self.rmt_extractor = RMTCollectiveModes(CONFIG["num_balls"], CONFIG["rmt_window"],
                                                CONFIG["rmt_out_dim"])
        self.gp_drift = HierarchicalDriftTracker(98, CONFIG["gp_drift_dim"])
        self.chaos_gauge = ChaosPredictabilityGauge(CONFIG["d_model"])
        total_dim = (CONFIG["d_model"] + CONFIG["physics_state_dim"] +
                     CONFIG["rmt_out_dim"] + CONFIG["gp_drift_dim"] + 1)
        self.fusion_proj = nn.Linear(total_dim, CONFIG["d_model"])
        self.mdn_proj = nn.Linear(CONFIG["d_model"],
                                  CONFIG["num_gaussians"] * (CONFIG["num_balls"] + 2))

    def forward(self, hist_tokens):
        B = hist_tokens.size(0)
        masses, probs = self.physics()
        phys_state = torch.cat([masses, probs], dim=-1)
        phys_latent = self.physics_encoder(phys_state.unsqueeze(0).expand(B, -1))

        tokens = hist_tokens.view(B, CONFIG["history_len"], CONFIG["draw_size"])
        emb = self.ball_embed(tokens)
        H = CONFIG["history_len"]
        emb_set = emb.view(B * H, CONFIG["draw_size"], CONFIG["d_model"])
        draw_enc = self.draw_encoder(emb_set)
        draw_emb = draw_enc.mean(dim=1).view(B, H, -1)
        temp_out = self.temporal_transformer(draw_emb)
        seq_summary = temp_out[:, -1, :]

        rmt_feat = self.rmt_extractor(hist_tokens)
        phys_seq = phys_state.unsqueeze(0).unsqueeze(1).repeat(B, H, 1)
        drift_feat = self.gp_drift(phys_seq)
        chaos_idx = self.chaos_gauge(hist_tokens)

        env_latent = self.env_proj(torch.zeros(B, CONFIG["env_feat_dim"], device=hist_tokens.device))
        combined = torch.cat([seq_summary, phys_latent, env_latent, rmt_feat, drift_feat, chaos_idx], dim=1)
        fused = F.relu(self.fusion_proj(combined))

        raw = self.mdn_proj(fused).view(B, CONFIG["num_gaussians"], -1)
        mix_logits = raw[:, :, 0]
        mix_coeffs = F.softmax(mix_logits, dim=-1)
        comp_params = raw[:, :, 1:]
        alpha_base = F.softplus(comp_params[:, :, :CONFIG["num_balls"]]) + 1e-6
        conc = F.softplus(comp_params[:, :, -1:]) + 1e-6
        alphas = alpha_base * conc
        return mix_coeffs, alphas

# =============================================================================
# Training & visualization
# =============================================================================
def train_and_visualize(csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CSVDataset(csv_path)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    model = LottOracleInverted().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print("=== Training LottOracle-Inferred ===")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for hist, tgt in loader:
            hist, tgt = hist.to(device), tgt.to(device)
            mix, alphas = model(hist)
            loss = -torch.logsumexp(torch.log(mix + 1e-10) +
                                    (F.one_hot(tgt, CONFIG["num_balls"]).float().unsqueeze(1) *
                                     torch.log(alphas/alphas.sum(-1,keepdim=True).unsqueeze(2) + 1e-10)).sum(-1).sum(-1),
                                    dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg loss = {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        masses, _ = model.physics()
        masses_g = masses.cpu().numpy() * 1000.0  # convert to grams

        # --- Plot inferred masses ---
        plt.figure(figsize=(10,5))
        plt.bar(range(1,50), masses_g, color='royalblue')
        plt.axhline(y=BALL_MASS_MEAN*1000, color='gray', linestyle='--', label='Nominal 3.3 g')
        plt.xlabel('Ball number')
        plt.ylabel('Mass (g)')
        plt.title('Inferred Ball Masses (Air‑Mix Model)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('inferred_masses.png', dpi=150)
        plt.show()

        # --- Predicted probabilities for next draw ---
        last_hist, _ = dataset[-1]
        hist = last_hist.unsqueeze(0).to(device)
        mix, alphas = model(hist)
        alpha_sums = alphas.sum(dim=-1, keepdim=True)
        probs = (mix.unsqueeze(-1) * alphas / alpha_sums).sum(dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(10,5))
        plt.bar(range(1,50), probs, color='darkorange')
        uniform = 1.0/49
        plt.axhline(y=uniform, color='gray', linestyle='--', label='Uniform (1/49)')
        plt.xlabel('Ball number')
        plt.ylabel('Probability')
        plt.title('Predicted Probability Distribution – Next Draw')
        plt.legend()
        plt.tight_layout()
        plt.savefig('predicted_probs.png', dpi=150)
        plt.show()

        print("\nPredicted set (top 6 weighted sample):")
        pred_set = np.random.choice(CONFIG["num_balls"], size=6, replace=False, p=probs)
        print(np.sort(pred_set)+1)

if __name__ == "__main__":
    train_and_visualize("lottery_results.csv")
