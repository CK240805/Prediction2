"""
Full Lottery 6/49 + Bonus Predictor (Deterministic, Chronological)
Uses all historical data, oldest first.
If 'additional' column is present, it learns and predicts all 7 numbers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Lottery Predictor", layout="wide")
st.title("🎰 6/49 + Bonus Predictor (Delta System, Chronological)")
st.markdown("Trains from oldest → newest. Automatically includes the 7th number if present.")

# ------------------------------
# LOAD & SORT DATA
# ------------------------------
@st.cache_data
def load_data(filepath='lottery_data.csv'):
    df = pd.read_csv(filepath, parse_dates=['date'])
    required = ['draw_no','date','n1','n2','n3','n4','n5','n6']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()
    # Sort chronologically (oldest first)
    df = df.sort_values(by=['date','draw_no'], ascending=[True, True]).reset_index(drop=True)
    return df

df = load_data()
st.write(f"📅 {len(df)} draws, {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

# Check if additional number exists and is valid
use_additional = 'additional' in df.columns and df['additional'].notna().any()
if use_additional:
    # Ensure additional is integer, drop draws where it's missing to avoid gaps
    df['additional'] = df['additional'].astype(int)
    df = df.dropna(subset=['additional']).reset_index(drop=True)
    st.success("✅ Using 7th (additional) number.")
else:
    st.info("ℹ️ No additional column – predicting only 6 numbers.")

# ------------------------------
# COMPUTE DELTAS (6 or 7 dims)
# ------------------------------
numbers = df[['n1','n2','n3','n4','n5','n6']].values
num_dims = 7 if use_additional else 6
deltas = np.zeros((len(df), num_dims), dtype=int)
deltas[:, 0] = numbers[:, 0]                     # n1
for i in range(1, 6):
    deltas[:, i] = numbers[:, i] - numbers[:, i-1]  # n2-n1, ..., n6-n5
if use_additional:
    deltas[:, 6] = df['additional'].values - numbers[:, 5]  # additional - n6

# ------------------------------
# PREPARE SEQUENCES (chronological)
# ------------------------------
def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i+seq_len].flatten())
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(deltas, seq_len=5)
st.caption(f"Training samples: {X.shape[0]}, each {X.shape[1]} features "
           f"({5} draws × {num_dims} deltas)")

# ------------------------------
# TRAIN RANDOM FOREST (deterministic)
# ------------------------------
@st.cache_resource
def train_model(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

rf = train_model(X, y)

# ------------------------------
# HELPER: Make valid delta set (6 or 7 numbers)
# ------------------------------
def make_valid_deltas(delta_vec, with_additional=False):
    """
    Sorts, clips, and adjusts a delta vector so that:
    - all deltas >= 1
    - cumulative sum yields unique numbers ≤ 49
    - if with_additional, the 7th number (additional) is not among the first 6
    """
    d = np.sort(np.maximum(np.round(delta_vec).astype(int), 1))
    n = len(d)
    # Ensure sum of all deltas <= 49
    while d.sum() > 49:
        d[-1] -= 1
        d = np.maximum(d, 1)
        d.sort()

    nums = np.cumsum(d)

    # Ensure all numbers unique (and ≤49)
    while len(set(nums)) < n or nums[-1] > 49:
        # find first duplicate
        for i in range(n-1):
            if nums[i] == nums[i+1]:
                d[i+1] += 1
                break
        else:
            if nums[-1] > 49:
                d[-1] -= 1
        d = np.maximum(d, 1)
        d.sort()
        while d.sum() > 49:
            d[-1] -= 1
            d = np.maximum(d, 1)
            d.sort()
        nums = np.cumsum(d)

    # Additional specific check: if we have 7 numbers, ensure the bonus is not in main set
    if with_additional and n == 7:
        main_set = set(nums[:6])
        if nums[6] in main_set:
            # increase delta7 until bonus is unique
            while nums[6] in main_set:
                d[6] += 1
                d = np.maximum(d, 1)
                # sum may exceed 49, adjust again
                while d.sum() > 49:
                    d[-1] -= 1
                    d = np.maximum(d, 1)
                d.sort()
                nums = np.cumsum(d)
    return d, nums

# ------------------------------
# GENERATE PREDICTION (button)
# ------------------------------
if st.button("🔮 Generate Prediction"):
    with st.spinner("Training on all data (oldest → newest)..."):
        last_sequence = deltas[-5:].flatten().reshape(1, -1)
        raw_pred = rf.predict(last_sequence)[0]
        valid_d, pred_nums = make_valid_deltas(raw_pred, with_additional=use_additional)

        # Baseline: most frequent delta for each position (all history)
        mode_deltas = np.array([np.bincount(deltas[:, j]).argmax() for j in range(num_dims)])
        mode_deltas.sort()
        while mode_deltas.sum() > 49:
            mode_deltas[-1] -= 1
            mode_deltas = np.maximum(mode_deltas, 1)
            mode_deltas.sort()
        mode_nums = np.cumsum(mode_deltas)
        if use_additional and len(mode_nums) == 7:
            # ensure mode bonus is unique
            while mode_nums[6] in set(mode_nums[:6]):
                mode_deltas[6] += 1
                while mode_deltas.sum() > 49:
                    mode_deltas[-1] -= 1
                mode_deltas = np.maximum(mode_deltas, 1)
                mode_deltas.sort()
                mode_nums = np.cumsum(mode_deltas)

    if use_additional:
        st.success("### 🎯 Predicted Numbers (6 + Bonus)")
        st.markdown(f"**Main:** {', '.join(map(str, pred_nums[:6]))}")
        st.markdown(f"**Bonus:** {pred_nums[6]}")
        st.info(f"**Baseline (mode):** Main: {', '.join(map(str, mode_nums[:6]))} | Bonus: {mode_nums[6]}")
    else:
        st.success("### 🎯 Predicted Numbers")
        st.markdown(f"# {', '.join(map(str, pred_nums))}")
        st.info(f"**Baseline (mode):** {', '.join(map(str, mode_nums))}")

    st.warning("⚠️ Prediction is deterministic and purely the average delta pattern. No edge.")

    # Delta distributions
    fig, axes = plt.subplots(2, 4 if use_additional else 3, figsize=(16,8))
    axes = axes.flatten()
    for i in range(num_dims):
        ax = axes[i]
        ax.hist(deltas[:, i], bins=range(1, 35), density=True, alpha=0.7, color='steelblue')
        ax.axvline(mode_deltas[i], color='red', linestyle='--', label=f'Mode={mode_deltas[i]}')
        ax.set_title(f'Delta {i+1}')
        ax.legend()
    for j in range(num_dims, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Last draws
    with st.expander("📋 Last 10 Historical Draws"):
        display_cols = ['draw_no','date','n1','n2','n3','n4','n5','n6']
        if use_additional:
            display_cols.append('additional')
        st.dataframe(df[display_cols].tail(10))

st.markdown("---")
st.markdown("Built with Streamlit • Training: oldest → newest")
