#!/usr/bin/env python3
# /bin/dashboard.py
# Streamlit dashboard: computes overall user risk (EWMA / Mean / Exposure+Credibility),
# prices off that overall risk with a logistic curve, and visualizes the result.

import os
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient

# -------------------- Defaults (adjustable in sidebar) --------------------
DEFAULT_DB_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DEFAULT_DB_NAME = os.getenv("MONGO_DB", "telematics")
DEFAULT_COLLECTION = os.getenv("MONGO_COLL", "risk_scores")

RISK_THRESHOLD = 0.75
EXPOSURE_FEATURE_DEFAULT = os.getenv("EXPOSURE_FEATURE", "trip_len_km")  # features[EXPOSURE_FEATURE] or *_x

DEFAULT_AGG_MODE = "Exposure+Credibility"   # UI label
DEFAULT_EWM_ALPHA = 0.30
DEFAULT_CRED_K = 20

DEFAULT_BASE_MONTHLY = 120.0
DEFAULT_MIN_FACTOR = 0.85
DEFAULT_MAX_FACTOR = 1.25
DEFAULT_LOG_A = 6.0
DEFAULT_LOG_B = 0.55
# -------------------------------------------------------------------------


# ----------------------------- Helper functions --------------------------
def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'timestamp' column exists; fallback to ObjectId.generation_time."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = df["_id"].apply(lambda oid: getattr(oid, "generation_time", pd.NaT))
    return df

def exposure_from_features_row(feat: dict | None, key: str) -> float:
    """Read exposure proxy (e.g., km) from features; supports *_x fallback."""
    if not isinstance(feat, dict):
        return 0.0
    val = feat.get(key, feat.get(f"{key}_x", 0.0))
    try:
        return float(val or 0.0)
    except Exception:
        return 0.0

def get_exposure_series(df: pd.DataFrame, key: str) -> pd.Series:
    return df["features"].apply(lambda f: exposure_from_features_row(f, key)).astype(float).clip(lower=0.0)

def compute_overall_series(
    risk: pd.Series,
    exposure: pd.Series,
    mode_label: str,         # "Exposure+Credibility" | "EWMA" | "Mean"
    ewm_alpha: float,
    cred_k: float,
    portfolio_mean: float,
) -> pd.Series:
    """Return a rolling overall-risk series aligned with risk index."""
    r = risk.astype(float).clip(0, 1)
    if mode_label == "EWMA":
        return r.ewm(alpha=float(ewm_alpha)).mean().clip(0, 1)
    if mode_label == "Mean":
        return r.expanding().mean().clip(0, 1)
    # Exposure + Credibility
    wsum = exposure.astype(float).clip(lower=0.0).cumsum() + 1e-9
    wrsum = (exposure * r).cumsum()
    wmean = wrsum / wsum
    n = pd.Series(np.arange(1, len(r) + 1, dtype=float), index=r.index, dtype=float)
    cred_w = n / (n + float(cred_k))
    overall = cred_w * wmean + (1.0 - cred_w) * float(portfolio_mean)
    return overall.clip(0, 1)

def behavior_factor_logistic(r: float, min_factor: float, max_factor: float, a: float, b: float) -> float:
    """Map risk in [0,1] → factor in [min_factor, max_factor] using a logistic curve."""
    r = clamp01(float(r))
    sig = 1.0 / (1.0 + np.exp(-a * (r - b)))
    return float(min_factor + (max_factor - min_factor) * sig)

def price_from_overall(overall_risk: float, base_monthly: float, min_factor: float, max_factor: float, a: float, b: float) -> float:
    return round(base_monthly * behavior_factor_logistic(overall_risk, min_factor, max_factor, a, b), 2)
# -------------------------------------------------------------------------


# ------------------------------- UI setup ---------------------------------
st.set_page_config(page_title="Telematics Risk Dashboard", layout="wide")
st.title("Telematics Risk & Pricing")
st.caption("User-level pricing from overall risk (MongoDB backend). Run with: streamlit run bin/dashboard.py")

# Sidebar → data/source
st.sidebar.header("Data")
db_uri = st.sidebar.text_input("MongoDB URI", DEFAULT_DB_URI)
db_name = st.sidebar.text_input("DB Name", DEFAULT_DB_NAME)
coll_name = st.sidebar.text_input("Collection", DEFAULT_COLLECTION)

# Connect and populate user list
try:
    mongo = MongoClient(db_uri, serverSelectionTimeoutMS=3000)
    col = mongo[db_name][coll_name]
    user_ids = sorted([u for u in col.distinct("user_id") if u is not None])
except Exception as e:
    col = None
    user_ids = []
    st.error(f"MongoDB connection failed: {e}")

user_id = st.sidebar.selectbox("User ID", user_ids or [0], index=0)

# Filters
st.sidebar.header("Filters")
use_date = st.sidebar.checkbox("Filter by date range", value=False)
if use_date:
    today = datetime.now()
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", today)
else:
    start_date = end_date = None

# Overall score config
st.sidebar.header("Overall Score")
agg_mode = st.sidebar.selectbox("Aggregation Mode", ["Exposure+Credibility", "EWMA", "Mean"],
                                index=["Exposure+Credibility","EWMA","Mean"].index(DEFAULT_AGG_MODE))
ewm_alpha = st.sidebar.slider("EWMA α", 0.05, 0.9, DEFAULT_EWM_ALPHA, 0.05)
cred_k = st.sidebar.slider("Credibility k", 0, 100, DEFAULT_CRED_K, 1)

# Exposure feature key
st.sidebar.header("Exposure Feature")
exposure_key = st.sidebar.text_input("Key in features{}", EXPOSURE_FEATURE_DEFAULT)

# Pricing curve config
st.sidebar.header("Pricing Curve")
base_monthly = st.sidebar.number_input("Base Monthly ($)", min_value=10.0, value=DEFAULT_BASE_MONTHLY, step=5.0)
min_factor = st.sidebar.number_input("Min Factor (best)", min_value=0.5, value=DEFAULT_MIN_FACTOR, step=0.01, format="%.2f")
max_factor = st.sidebar.number_input("Max Factor (worst)", min_value=0.6, value=DEFAULT_MAX_FACTOR, step=0.01, format="%.2f")
log_a = st.sidebar.slider("Logistic Slope (a)", 1.0, 12.0, DEFAULT_LOG_A, 0.5)
log_b = st.sidebar.slider("Logistic Midpoint (b)", 0.1, 0.9, DEFAULT_LOG_B, 0.01)

st.sidebar.markdown("---")
export_csv = st.sidebar.checkbox("Enable CSV download", value=True)

if col is None:
    st.stop()

# ------------------------------- Load data --------------------------------
records = list(col.find({"user_id": user_id}).sort([("_id", 1)]))
if not records:
    st.info(f"No trips found for user_id={user_id}. Insert some trips and refresh.")
    st.stop()

df = ensure_timestamp(pd.DataFrame(records)).sort_values("timestamp").reset_index(drop=True)

if use_date:
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    df = df.loc[mask].copy()
    if df.empty:
        st.warning("No trips in the selected date range.")
        st.stop()

# Series
risk = df["risk_score"].astype(float).clip(0, 1)
exposure = get_exposure_series(df, key=exposure_key)

# Portfolio mean for credibility blend
agg = list(col.aggregate([{"$group": {"_id": None, "m": {"$avg": "$risk_score"}}}]))
portfolio_mean = float(agg[0]["m"]) if agg else 0.5

# Overall + pricing
overall = compute_overall_series(
    risk=risk,
    exposure=exposure,
    mode_label=agg_mode,
    ewm_alpha=ewm_alpha,
    cred_k=cred_k,
    portfolio_mean=portfolio_mean,
)
price_series = overall.apply(lambda r: price_from_overall(r, base_monthly, min_factor, max_factor, log_a, log_b))
overall_final = float(overall.iloc[-1])
premium_final = float(price_series.iloc[-1])

# ------------------------------- KPIs -------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trips", len(df))
c2.metric("Overall Risk (pricing)", f"{overall_final:.3f}")
c3.metric("Est. Monthly Premium", f"${premium_final:,.2f}")
c4.metric("Avg Trip Risk", f"{risk.mean():.3f}")

# ------------------------------- Chart ------------------------------------
plot_df = pd.DataFrame({
    "timestamp": df["timestamp"],
    "Trip Risk": risk.values,
    "Overall Risk": overall.values,
    "Premium ($)": price_series.values,
    "Exposure (km)": exposure.values,
})

base = alt.Chart(plot_df).encode(x=alt.X("timestamp:T", title="Time"))
risk_line = base.mark_line(point=True, color="#2563eb").encode(
    y=alt.Y("Trip Risk:Q", title="Risk"),
    tooltip=[
        alt.Tooltip("timestamp:T", title="Time"),
        alt.Tooltip("Trip Risk:Q", format=".3f"),
        alt.Tooltip("Exposure (km):Q", format=".2f"),
        alt.Tooltip("Overall Risk:Q", format=".3f"),
        alt.Tooltip("Premium ($):Q", format="$.2f"),
    ],
).properties(title=f"User {user_id} — Risk Over Time")
overall_line = base.mark_line(strokeDash=[4,4], color="#0ea5e9").encode(y="Overall Risk:Q")
threshold_rule = alt.Chart(pd.DataFrame({"y": [RISK_THRESHOLD]})).mark_rule(strokeDash=[6,4], color="#ef4444").encode(y="y:Q")

st.altair_chart((risk_line + overall_line + threshold_rule).interactive(), use_container_width=True)

# ------------------------------- Table ------------------------------------
if "trip_id" not in df.columns:
    df["trip_id"] = df["_id"].astype(str)
if "segment_id" not in df.columns:
    df["segment_id"] = df.get("features", {}).apply(lambda f: (f or {}).get("segment_id", ""))

table = df.copy()
table["exposure_km"] = exposure.values
table["overall_risk_rolling"] = overall.values
table["premium_rolling"] = price_series.values
table = table[["timestamp", "trip_id", "segment_id", "risk_score", "exposure_km", "overall_risk_rolling", "premium_rolling"]]

st.subheader("Trips")
st.dataframe(table, use_container_width=True)

# ------------------------------ Explainer ---------------------------------
formula_block = (
    "```text\n"
    f"behavior_factor = {min_factor:.2f} + ({max_factor:.2f} - {min_factor:.2f}) * "
    f"1/(1 + exp(-{log_a:.2f} * (risk - {log_b:.2f})))\n"
    f"monthly_premium = {base_monthly:.2f} * behavior_factor\n"
    "```"
)
with st.expander("How pricing is calculated"):
    st.markdown(
        "- **Overall risk** (default = Exposure+Credibility): exposure-weighted rolling mean "
        f"of trip risks, credibility-blended to the portfolio mean using **N/(N+{cred_k})**."
    )
    st.markdown(
        f"- **Pricing curve (logistic)** maps overall risk [0..1] to a factor in "
        f"[{min_factor:.2f}, {max_factor:.2f}] and then multiplies by base monthly **${base_monthly:.2f}**."
    )
    st.markdown(formula_block)

# ------------------------------ Download ----------------------------------
if export_csv:
    st.download_button(
        "Download CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"user_{user_id}_trips.csv",
        mime="text/csv",
        use_container_width=True
    )

st.caption("© Telematics POC — Streamlit dashboard (bin)")
