"""
Full Lottery 6/49 Predictor (Deterministic, All Historical Deltas)
Input CSV: draw_no, date, n1, n2, n3, n4, n5, n6, additional
Output: A fixed predicted 6-number set, reproducible for the same CSV.
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
st.title("🎰 6/49 Lottery Predictor (Deterministic ML + Delta System)")
st.markdown(
    "This app uses **all historical draws**, the **delta system**, and a "
    "Random Forest to produce a single, fixed prediction. "
    "It is purely educational – a fair lottery cannot be predicted."
)

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data(filepath='lottery_data.csv'):
    df = pd.read_csv(filepath, parse_dates=['date'])
    required_cols = ['draw_no','date','n1','n2','n3','n4','n5','n6','additional']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()
    return df

df = load_data()
st.write(f"📅 Loaded {len(df)} draws from {df['date'].min().date()} to {df['date'].max().date()}")

# ------------------------------
# COMPUTE DELTAS
# ------------------------------
numbers = df[['n1','n2','n3','n4','n5','n6']].values
deltas = np.zeros_like(numbers)
deltas[:, 0] = numbers[:, 0]                # first number = first delta
for i in range(1, 6):
    deltas[:, i] = numbers[:, i] - numbers[:, i-1]

# Add deltas to DataFrame for optional display
for i in range(6):
    df[f'delta{i+1}'] = deltas[:, i]

# ------------------------------
# PREPARE SEQUENCES FOR ML
# ------------------------------
def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len].flatten())   # last seq_len delta vectors
        y.append(data[i+seq_len])               # next delta vector
    return np.array(X), np.array(y)

X, y = create_sequences(deltas, seq_len=5)
st.caption(f"Training samples: {X.shape[0]}, features: {X.shape[1]} (5 draws × 6 deltas)")

# ------------------------------
# TRAIN RANDOM FOREST (deterministic, fixed seed)
# ------------------------------
@st.cache_resource
def train_model(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

rf = train_model(X, y)

# ------------------------------
# HELPER: Make a valid set of 6 numbers from any delta vector
# ------------------------------
def make_valid_deltas(delta_vec):
    """
    Takes a raw 6-element array, sorts it, and adjusts so that:
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
    # If duplicates exist (deltas can be 1 and cause same cumulative)
    while len(set(nums)) < 6 or nums[-1] > 49:
        # Find the first delta that can be increased to break tie
        for i in range(5):
            if nums[i] == nums[i+1]:
                d[i+1] += 1
                break
        else:
            # if no duplicate but last >49, reduce largest delta
            if nums[-1] > 49:
                d[-1] -= 1
        d = np.maximum(d, 1)
        d.sort()
        # Re-clamp sum
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
    with st.spinner("Analyzing all historical data..."):
        # Use the last 5 delta vectors as input (same every time)
        last_sequence = deltas[-5:].flatten().reshape(1, -1)
        raw_pred_deltas = rf.predict(last_sequence)[0]
        valid_deltas, final_numbers = make_valid_deltas(raw_pred_deltas)

        # Most-frequent-delta baseline (mode of each delta position)
        mode_deltas = np.array([np.bincount(deltas[:, j]).argmax() for j in range(6)])
        mode_deltas.sort()
        if mode_deltas.sum() > 49:
            while mode_deltas.sum() > 49:
                mode_deltas[-1] -= 1
                mode_deltas = np.maximum(mode_deltas, 1)
                mode_deltas.sort()
        mode_numbers = np.cumsum(mode_deltas)

    col1, col2 = st.columns(2)
    with col1:
        st.success("### 🎯 RF Predicted Numbers")
        st.markdown(f"# {', '.join(map(str, final_numbers))}")
        st.caption("Using Random Forest trained on all delta sequences.")

    with col2:
        st.info("### 📊 Most-Frequent-Delta Baseline")
        st.markdown(f"# {', '.join(map(str, mode_numbers))}")
        st.caption("The single most common delta for each position (all history).")

    st.warning(
        "⚠️ This prediction is **deterministic and fixed** for the given data. "
        "It represents the average delta pattern – not a real edge. "
        "The lottery remains completely random."
    )

    # Show delta distributions for context
    st.subheader("Historical Delta Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.hist(deltas[:, i], bins=range(1, 35), density=True, alpha=0.7, color='steelblue')
        ax.axvline(mode_deltas[i], color='red', linestyle='--', label=f'Mode = {mode_deltas[i]}')
        ax.set_title(f'Delta {i+1}')
        ax.set_xlabel('Value')
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Optional: Show last 10 draws for reference
    with st.expander("📋 Last 10 Historical Draws"):
        st.dataframe(df[['draw_no','date','n1','n2','n3','n4','n5','n6']].tail(10))

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Built with Streamlit • [GitHub Repository](https://github.com)")
