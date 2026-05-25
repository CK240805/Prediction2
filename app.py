"""
Full Lottery 6/49 Predictor (Deterministic, Chronological Training)
Input CSV: draw_no, date, n1, n2, n3, n4, n5, n6, additional
Training: from oldest draw to newest, no future leakage.
Output: a fixed predicted 6-number set for the next draw.
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
st.set_page_config(page_title="Lottery 6/49 Predictor", layout="wide")
st.title("🎰 6/49 Lottery Predictor (Deterministic, Oldest→Newest)")
st.markdown(
    "This app trains a Random Forest on **all historical draws in chronological order** "
    "(oldest first), then predicts the next 6 numbers using the delta system. "
    "It’s for education only—a fair lottery cannot be predicted."
)

# ------------------------------
# LOAD DATA & SORT CHRONOLOGICALLY
# ------------------------------
@st.cache_data
def load_data(filepath='lottery_data.csv'):
    df = pd.read_csv(filepath, parse_dates=['date'])
    required_cols = ['draw_no','date','n1','n2','n3','n4','n5','n6','additional']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()
    # Sort by date ascending (oldest first). Use draw_no if dates are identical.
    df = df.sort_values(by=['date', 'draw_no'], ascending=[True, True]).reset_index(drop=True)
    return df

df = load_data()
st.write(f"📅 Loaded {len(df)} draws, sorted oldest → newest.")
st.write(f"First draw: {df['date'].iloc[0].date()}, Last draw: {df['date'].iloc[-1].date()}")

# ------------------------------
# COMPUTE DELTAS (on sorted data)
# ------------------------------
numbers = df[['n1','n2','n3','n4','n5','n6']].values
deltas = np.zeros_like(numbers)
deltas[:, 0] = numbers[:, 0]                # first number = first delta
for i in range(1, 6):
    deltas[:, i] = numbers[:, i] - numbers[:, i-1]

# ------------------------------
# PREPARE SEQUENCES (chronological, no shuffling)
# ------------------------------
def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        # Use draws i...i+seq_len-1 to predict draw i+seq_len
        X.append(data[i : i+seq_len].flatten())   # last seq_len delta vectors
        y.append(data[i+seq_len])                 # next delta vector
    return np.array(X), np.array(y)

X, y = create_sequences(deltas, seq_len=5)
st.caption(f"Training samples: {X.shape[0]} (each from {X.shape[1]} features: 5 draws × 6 deltas)")

# ------------------------------
# TRAIN RANDOM FOREST (deterministic, on all sequences)
# ------------------------------
@st.cache_resource
def train_model(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)   # no shuffling needed; tree-based models treat order as arbitrary
    return rf

rf = train_model(X, y)

# ------------------------------
# HELPER: Make a valid set of 6 numbers from any delta vector
# ------------------------------
def make_valid_deltas(delta_vec):
    """
    Sorts, rounds, clips, and adjusts deltas to guarantee:
    - all deltas >= 1
    - sum <= 49
    - cumulative sum gives 6 unique numbers all <= 49
    Returns (valid_deltas, numbers)
    """
    d = np.sort(np.maximum(np.round(delta_vec).astype(int), 1))

    # Ensure sum <= 49
    while d.sum() > 49:
        d[-1] -= 1
        d = np.maximum(d, 1)
        d.sort()

    nums = np.cumsum(d)
    # Resolve duplicate cumulative values (when a delta is 1)
    while len(set(nums)) < 6 or nums[-1] > 49:
        for i in range(5):
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
    return d, nums

# ------------------------------
# GENERATE PREDICTION (button)
# ------------------------------
if st.button("🔮 Generate Deterministic Prediction"):
    with st.spinner("Training on all past data (oldest → newest)..."):
        # Input: the last 5 delta vectors (from the most recent draws)
        last_sequence = deltas[-5:].flatten().reshape(1, -1)
        raw_pred_deltas = rf.predict(last_sequence)[0]
        valid_deltas, final_numbers = make_valid_deltas(raw_pred_deltas)

        # Most-frequent-delta baseline (mode of each delta position, over all history)
        mode_deltas = np.array([np.bincount(deltas[:, j]).argmax() for j in range(6)])
        mode_deltas.sort()
        while mode_deltas.sum() > 49:
            mode_deltas[-1] -= 1
            mode_deltas = np.maximum(mode_deltas, 1)
            mode_deltas.sort()
        mode_numbers = np.cumsum(mode_deltas)

    col1, col2 = st.columns(2)
    with col1:
        st.success("### 🎯 RF Predicted Numbers")
        st.markdown(f"# {', '.join(map(str, final_numbers))}")
        st.caption("Random Forest trained on all chronological sequences.")

    with col2:
        st.info("### 📊 Most-Frequent-Delta Baseline")
        st.markdown(f"# {', '.join(map(str, mode_numbers))}")
        st.caption("The single most common delta at each position (all history).")

    st.warning(
        "⚠️ This prediction is **deterministic** and will never change "
        "for the same data. It represents the average delta pattern, "
        "not a genuine edge. The lottery remains a game of pure chance."
    )

    # Show delta distributions
    st.subheader("Historical Delta Distributions (Oldest → Newest)")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.hist(deltas[:, i], bins=range(1, 35), density=True, alpha=0.7, color='steelblue')
        ax.axvline(mode_deltas[i], color='red', linestyle='--', label=f'Mode = {mode_deltas[i]}')
        ax.set_title(f'Delta {i+1}')
        ax.set_xlabel('Value')
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Show last 10 draws
    with st.expander("📋 Last 10 Historical Draws (Newest at Bottom)"):
        st.dataframe(df[['draw_no','date','n1','n2','n3','n4','n5','n6']].tail(10))

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Built with Streamlit • Training direction: oldest → newest")
