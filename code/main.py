!pip install hmmlearn pandas_datareader -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from hmmlearn.hmm import GaussianHMM

# -----------------------------
# 1. Download public data
# -----------------------------

start = "2018-01-01"
end = "2025-12-31"

sofr = web.DataReader("SOFR", "fred", start, end)
tbill = web.DataReader("DTB3", "fred", start, end)
vix = web.DataReader("VIXCLS", "fred", start, end)

df = pd.concat([sofr, tbill, vix], axis=1)
df.columns = ["SOFR", "TBill_3M", "VIX"]
df = df.dropna()

# -----------------------------
# 2. Build simple indicators
# -----------------------------

df["funding_spread"] = df["SOFR"] - df["TBill_3M"]

# Smooth inputs slightly to reduce noisy regime switching
df["funding_spread_smooth"] = df["funding_spread"].rolling(5).mean()
df["vix_smooth"] = df["VIX"].rolling(5).mean()

model_df = df.dropna().copy()

X = model_df[["funding_spread_smooth", "vix_smooth"]].values

# -----------------------------
# 3. Fit 2 state HMM
# -----------------------------

model = GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

model.fit(X)

model_df["state"] = model.predict(X)
probs = model.predict_proba(X)

# Stress state = higher average VIX
state_summary = model_df.groupby("state")[["funding_spread", "VIX"]].mean()
stress_state = state_summary["VIX"].idxmax()

model_df["stress_probability"] = probs[:, stress_state]

# Label regimes clearly
state_summary["Regime"] = [
    "Stress" if state == stress_state else "Calm"
    for state in state_summary.index
]

state_summary = state_summary[["Regime", "funding_spread", "VIX"]]
state_summary.columns = ["Regime", "Avg Funding Spread", "Avg VIX"]

# -----------------------------
# 4. Clean chart 1
# -----------------------------

plt.figure(figsize=(13, 5))
plt.plot(model_df.index, model_df["stress_probability"], linewidth=1.4)

plt.axvline(pd.Timestamp("2020-03-01"), linestyle="--", linewidth=1)
plt.axvline(pd.Timestamp("2023-03-01"), linestyle="--", linewidth=1)

plt.text(pd.Timestamp("2020-03-15"), 0.9, "COVID stress", fontsize=9)
plt.text(pd.Timestamp("2023-03-15"), 0.8, "Regional banking stress", fontsize=9)

plt.title("HMM Estimated Probability of Liquidity Stress Regime")
plt.ylabel("Stress Probability")
plt.xlabel("Date")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Clean chart 2
# -----------------------------

plt.figure(figsize=(13, 5))
plt.plot(model_df.index, model_df["funding_spread"], linewidth=1.2)
plt.axhline(0, linestyle="--", linewidth=1)

plt.axvline(pd.Timestamp("2020-03-01"), linestyle="--", linewidth=1)
plt.axvline(pd.Timestamp("2023-03-01"), linestyle="--", linewidth=1)

plt.title("SOFR minus 3 Month Treasury Bill Spread")
plt.ylabel("Spread, percentage points")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Clean summary outputs
# -----------------------------

print("REGIME SUMMARY")
print("=" * 50)
print(state_summary.round(3).to_string())

print("\nINTERPRETATION")
print("=" * 50)
print(
    "The HMM separates the sample into a calm regime and a stress regime. "
    "The stress regime is defined as the state with the higher average VIX. "
    "This makes the model easy to interpret: it identifies periods where market "
    "volatility and funding spread behaviour suggest tighter liquidity conditions."
)

transition_matrix = pd.DataFrame(
    model.transmat_,
    columns=["To State 0", "To State 1"],
    index=["From State 0", "From State 1"]
)

print("\nTRANSITION MATRIX")
print("=" * 50)
print(transition_matrix.round(3).to_string())

# -----------------------------
# 7. Save outputs
# -----------------------------

model_df.to_csv("funding_stress_hmm_results.csv")
state_summary.to_csv("regime_summary.csv")
transition_matrix.to_csv("transition_matrix.csv")

print("\nSaved files:")
print("- funding_stress_hmm_results.csv")
print("- regime_summary.csv")
print("- transition_matrix.csv")
