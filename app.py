import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ---------- Physical constants ----------
BALL_MASS_MEAN  = 3.3e-3
BALL_MASS_TOL   = 0.165e-3
BALL_DIAMETER   = 40.0e-3

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

# ---------- Dataset ----------
class CSVDataset(Dataset):
    def __init__(self, df):
        df = df.sort_values("draw_no", ascending=True).reset_index(drop=True)
        draws = df[["n1","n2","n3","n4","n5","n6"]].values - 1
        self.draws = draws.astype(np.int64)

    def __len__(self):
        return len(self.draws) - CONFIG["history_len"]

    def __getitem__(self, idx):
        hist = self.draws[idx : idx + CONFIG["history_len"]]
        nxt  = self.draws[idx + CONFIG["history_len"]]
        return torch.tensor(hist.flatten() + 1, dtype=torch.long), torch.tensor(nxt, dtype=torch.long)

# ---------- Physics engine ----------
class AirMixPhysicsEngine(nn.Module):
    def __init__(self, num_balls=49):
        super().__init__()
        self.raw_masses = nn.Parameter(torch.zeros(num_balls))
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self):
        masses = BALL_MASS_MEAN + BALL_MASS_TOL * torch.tanh(self.raw_masses)
        alpha = F.softplus(self.alpha) + 0.1
        weight = 1.0 / (masses ** alpha + 1e-12)
        return masses, weight / weight.sum()

# ---------- RMT ----------
class RMTCollectiveModes(nn.Module):
    def __init__(self, num_balls=49, rmt_window=30, out_dim=64):
        super().__init__()
        self.num_balls = num_balls
        self.rmt_window = rmt_window
        self.out_dim = out_dim
        self.top_k = min(out_dim, num_balls)
        self.proj = nn.Linear(num_balls * self.top_k, out_dim)

    def forward(self, hist_tokens):
        B = hist_tokens.size(0)
        draws = hist_tokens.view(B, CONFIG["history_len"], CONFIG["draw_size"]) - 1
        draws = draws[:, -self.rmt_window:, :]
        multi_hot = torch.zeros(B, self.rmt_window, self.num_balls, device=hist_tokens.device)
        for w in range(self.rmt_window):
            for s in range(CONFIG["draw_size"]):
                multi_hot[:, w, draws[:, w, s]] = 1.0
        X = multi_hot.permute(0, 2, 1)
        Xc = X - X.mean(dim=-1, keepdim=True)
        C = torch.bmm(Xc, Xc.transpose(1,2)) / (self.rmt_window - 1)
        _, eigvecs = torch.linalg.eigh(C)
        sel = eigvecs[:, :, -self.top_k:]
        flat = sel.reshape(B, -1)
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
        total_dim = (CONFIG["d_model"] + CONFIG["physics_state_dim"] + CONFIG["d_model"] +
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
        combined = torch.cat([
            seq_summary, phys_latent, env_latent,
            rmt_feat, drift_feat, chaos_idx
        ], dim=1)
        fused = F.relu(self.fusion_proj(combined))

        raw = self.mdn_proj(fused).view(B, CONFIG["num_gaussians"], -1)
        mix_logits = raw[:, :, 0]
        mix_coeffs = F.softmax(mix_logits, dim=-1)
        comp_params = raw[:, :, 1:]
        alpha_base = F.softplus(comp_params[:, :, :CONFIG["num_balls"]]) + 1e-6
        conc = F.softplus(comp_params[:, :, -1:]) + 1e-6
        alphas = alpha_base * conc
        return mix_coeffs, alphas

# ---------- Training ----------
def train_model(df, epochs, device):
    dataset = CSVDataset(df)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    model = LottOracleInverted().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for hist, tgt in loader:
            hist, tgt = hist.to(device), tgt.to(device)
            mix, alphas = model(hist)
            alpha_sums = alphas.sum(dim=-1, keepdim=True)
            probs = alphas / alpha_sums
            tgt_oh = F.one_hot(tgt, CONFIG["num_balls"]).float()
            log_like = (tgt_oh.unsqueeze(1) * torch.log(probs.unsqueeze(2) + 1e-10)).sum(-1).sum(-1)
            loss = -torch.logsumexp(torch.log(mix + 1e-10) + log_like, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
    return model, dataset

# ---------- Streamlit UI ----------
st.set_page_config(page_title="LottOracle", layout="wide")
st.title("🎱 LottOracle – Air‑Mix Physics Inverter")
st.markdown("Upload your **6/49 lottery CSV** or use the default file from the repository.")

# Check for default CSV in the repo
DEFAULT_CSV = "toto_results.csv"
use_default = False
if os.path.exists(DEFAULT_CSV):
    use_default = st.checkbox("Use default 'toto_results.csv' from repository", value=True)

uploaded_file = None
if not use_default or not os.path.exists(DEFAULT_CSV):
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

epochs = st.slider("Training epochs", 1, 20, 5)

df = None
if use_default and os.path.exists(DEFAULT_CSV):
    df = pd.read_csv(DEFAULT_CSV)
    st.success(f"Loaded {len(df)} draws from '{DEFAULT_CSV}'.")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} draws from uploaded file.")
else:
    st.info("Please upload a CSV file or ensure the default CSV exists in the repository.")

if df is not None:
    if st.button("🚀 Train & Predict"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner("Training the digital twin... (this may take a minute)"):
            model, dataset = train_model(df, epochs, device)

        model.eval()
        with torch.no_grad():
            masses, _ = model.physics()
            masses_g = masses.cpu().numpy() * 1000.0

            last_hist, _ = dataset[-1]
            hist = last_hist.unsqueeze(0).to(device)
            mix, alphas = model(hist)
            alpha_sums = alphas.sum(dim=-1, keepdim=True)
            probs = (mix.unsqueeze(-1) * alphas / alpha_sums).sum(dim=1).squeeze().cpu().numpy()

        fig1, ax1 = plt.subplots(figsize=(10,4))
        ax1.bar(range(1,50), masses_g, color='royalblue')
        ax1.axhline(y=BALL_MASS_MEAN*1000, color='gray', linestyle='--', label='Nominal 3.3 g')
        ax1.set_xlabel('Ball number'); ax1.set_ylabel('Mass (g)')
        ax1.set_title('Inferred Ball Masses (Air‑Mix Physics)')
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.bar(range(1,50), probs, color='darkorange')
        ax2.axhline(y=1/49, color='gray', linestyle='--', label='Uniform (1/49)')
        ax2.set_xlabel('Ball number'); ax2.set_ylabel('Probability')
        ax2.set_title('Predicted Probabilities – Next Draw')
        ax2.legend()
        st.pyplot(fig2)

        pred_set = np.random.choice(CONFIG["num_balls"], size=6, replace=False, p=probs)
        st.subheader("🎯 Sampled Predicted Set")
        st.markdown(" ".join([f"**{n+1:02d}**" for n in np.sort(pred_set)]))

        edges = probs - (1/49)
        top_edges = np.argsort(edges)[-10:][::-1]
        st.subheader("🔝 Top 10 Over‑Weighted Numbers")
        for idx in top_edges:
            st.write(f"Ball **{idx+1:02d}** – probability: {probs[idx]:.4f} (edge: {edges[idx]:.5f})")

        st.caption("⚠️ Educational simulation only. No real‑world lottery prediction is intended or possible on fair draws.")
