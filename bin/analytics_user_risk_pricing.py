#!/usr/bin/env python3
# /src/analytics_user_risk.py
#
# Modern Matplotlib viz + hover pop-ups (if mplcursors installed).
# Pricing is computed from the user's OVERALL risk (configurable aggregator).

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import seaborn as sns

# Optional hover library
try:
    import mplcursors  # pip install mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

# -------- CONFIG -------- #
USER_ID = 4
DB_URI = "mongodb://localhost:27017/"
DB_NAME = "telematics"
COLLECTION = "risk_scores"

RISK_THRESHOLD = 0.75

# Visual theme (built-in Matplotlib style). Examples:
# 'seaborn-v0_8-whitegrid', 'seaborn-v0_8-darkgrid', 'fivethirtyeight', 'ggplot', 'bmh'
STYLE_NAME = "seaborn-v0_8-whitegrid"

# Overall score aggregation mode:
AGGREGATION_MODE = "exposure_cred"   # "exposure_cred" | "ewma" | "mean"
EWM_ALPHA = 0.30                     # used if AGGREGATION_MODE == "ewma"
CREDIBILITY_K = 20                   # higher = slower to trust user's data
EXPOSURE_FEATURE = "trip_len_km"     # exposure proxy (km); falls back to *_x if needed

# Pricing (overall-risk → monthly premium via logistic curve)
BASE_MONTHLY = 120.0   # $
MIN_FACTOR   = 0.85    # best driver factor
MAX_FACTOR   = 1.25    # riskiest driver factor
LOGISTIC_A   = 6.0     # slope
LOGISTIC_B   = 0.55    # midpoint
# ------------------------ #

# --- Style setup (uses built-in style sheets) ---
try:
    plt.style.use(STYLE_NAME)
except Exception:
    # Fallback without erroring out
    pass

# Additional rc tweaks for a cleaner look
mpl.rcParams.update({
    "figure.figsize": (11, 5.8),
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#e5e7eb",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.5,
})

# ---- DB ----
client = pymongo.MongoClient(DB_URI)
collection = client[DB_NAME][COLLECTION]

records = list(collection.find({"user_id": USER_ID}))
if not records:
    print(f"❌ No records found for user_id={USER_ID}")
    raise SystemExit(0)

df = pd.DataFrame(records)

# Timestamp
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    df["timestamp"] = df["_id"].apply(lambda oid: oid.generation_time)
df = df.sort_values("timestamp").reset_index(drop=True)

# Series
risk = df["risk_score"].astype(float).clip(0, 1)
ts   = pd.to_datetime(df["timestamp"])

def _get_exposure_row(feat: dict) -> float:
    if not isinstance(feat, dict):
        return 0.0
    val = feat.get(EXPOSURE_FEATURE, feat.get(f"{EXPOSURE_FEATURE}_x", 0.0))
    try:
        return float(val or 0.0)
    except Exception:
        return 0.0

exposure = df["features"].apply(_get_exposure_row).astype(float).clip(lower=0.0)

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def behavior_factor_logistic(r: float,
                             min_f: float = MIN_FACTOR,
                             max_f: float = MAX_FACTOR,
                             a: float = LOGISTIC_A,
                             b: float = LOGISTIC_B) -> float:
    r = clamp01(r)
    sig = 1.0 / (1.0 + np.exp(-a * (r - b)))
    return min_f + (max_f - min_f) * sig

def price_from_overall(r_overall: float, base_monthly: float = BASE_MONTHLY) -> float:
    return round(base_monthly * behavior_factor_logistic(r_overall), 2)

# ---- Rolling overall series (so hover can show rolling premium) ----
if AGGREGATION_MODE == "ewma":
    overall_series = risk.ewm(alpha=EWM_ALPHA).mean()
elif AGGREGATION_MODE == "mean":
    overall_series = risk.expanding().mean()
elif AGGREGATION_MODE == "exposure_cred":
    # rolling exposure-weighted mean
    wsum = exposure.cumsum() + 1e-9
    wrsum = (exposure * risk).cumsum()
    wmean = wrsum / wsum
    # credibility towards portfolio mean
    agg = list(collection.aggregate([{"$group": {"_id": None, "m": {"$avg": "$risk_score"}}}]))
    portfolio_mean = float(agg[0]["m"]) if agg else 0.5
    n = pd.Series(np.arange(1, len(risk) + 1), index=risk.index, dtype=float)
    cred_w = n / (n + float(CREDIBILITY_K))
    overall_series = cred_w * wmean + (1.0 - cred_w) * portfolio_mean
else:
    raise ValueError("AGGREGATION_MODE must be 'exposure_cred' | 'ewma' | 'mean'")

overall_series = overall_series.clip(lower=0.0, upper=1.0)
price_series = overall_series.apply(price_from_overall)

overall_final = float(overall_series.iloc[-1])
premium_final = float(price_series.iloc[-1])

# ----- ANALYTICS -----
summary = {
    "User ID": USER_ID,
    "Aggregation Mode": AGGREGATION_MODE,
    "Total Trips": int(len(df)),
    "Overall Risk (pricing)": round(overall_final, 4),
    "Est. Monthly Premium ($)": premium_final,
    "Avg Trip Risk": round(float(risk.mean()), 4),
    "Min Trip Risk": round(float(risk.min()), 4),
    "Max Trip Risk": round(float(risk.max()), 4),
    "Std Dev": round(float(risk.std()), 4),
    "Recent Trend (Δ)": round(float(risk.diff().mean()), 4),
    "Trips > Threshold": int((risk > RISK_THRESHOLD).sum()),
    "% > Threshold": round(float((risk > RISK_THRESHOLD).mean() * 100), 2),
}

print("\n📊 Risk Summary (Pricing from Overall User Score)")
for k, v in summary.items():
    print(f"{k:>30}: {v}")

# ----- VISUALIZATION -----
fig, ax = plt.subplots()
try:
    fig.canvas.manager.set_window_title(f"User {USER_ID} — Risk & Pricing")
except Exception:
    pass

# Trip risk line + markers
line_risk, = ax.plot(
    ts, risk, marker='o', linestyle='-', linewidth=2,
    color="#2563eb", markersize=6, markerfacecolor="#60a5fa",
    markeredgecolor="#ffffff", markeredgewidth=0.8, label="Trip Risk"
)

# Rolling overall (dotted)
ax.plot(
    ts, overall_series, linestyle='--', linewidth=2,
    color="#0ea5e9", label="Overall Risk (rolling)"
)

# Threshold
ax.axhline(RISK_THRESHOLD, color='#ef4444', linestyle='--', linewidth=1.3,
           label=f"Threshold {RISK_THRESHOLD}")

ax.set_title(f"User {USER_ID} — Risk Over Time (Pricing from Overall Score)")
ax.set_xlabel("Time"); ax.set_ylabel("Risk Score")
plt.xticks(rotation=35)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc="upper left")

# In-figure summary "card"
box_text = (
    f"Overall Risk (pricing): {summary['Overall Risk (pricing)']:.4f}\n"
    f"Est. Monthly Premium:  ${summary['Est. Monthly Premium ($)']:.2f}\n"
    f"Aggregation: {summary['Aggregation Mode']}\n"
    f"Trips: {summary['Total Trips']}  |  "
    f"Avg/Min/Max: {summary['Avg Trip Risk']:.3f}/"
    f"{summary['Min Trip Risk']:.3f}/{summary['Max Trip Risk']:.3f}\n"
    f"Std Dev: {summary['Std Dev']:.3f}  |  Trend Δ: {summary['Recent Trend (Δ)']:.3f}  |  "
    f">{RISK_THRESHOLD}: {summary['Trips > Threshold']} ({summary['% > Threshold']}%)"
)
bbox = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95, edgecolor="#d1d5db")
ax.text(0.99, 0.99, box_text, transform=ax.transAxes, va="top", ha="right",
        fontsize=10, bbox=bbox)

# Hover pop-ups (if mplcursors present): show trip risk + exposure + rolling overall + rolling premium
if HAS_MPLCURSORS:
    cursor = mplcursors.cursor(line_risk, hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        i = sel.index
        ts_str = pd.to_datetime(ts.iloc[i]).strftime("%Y-%m-%d %H:%M")
        r = float(risk.iloc[i])
        exp_km = float(exposure.iloc[i])
        r_over = float(overall_series.iloc[i])
        prem = float(price_series.iloc[i])
        sel.annotation.set(
            text=(
                f"{ts_str}\n"
                f"Trip Risk: {r:.3f}\n"
                f"Exposure (km): {exp_km:.2f}\n"
                f"Overall (roll): {r_over:.3f}\n"
                f"Premium (roll): ${prem:.2f}"
            ),
            fontsize=9
        )
        sel.annotation.get_bbox_patch().set(fc="white", ec="#d1d5db", alpha=0.95)

plt.tight_layout()
plt.show()
