#!/usr/bin/env python3
# /src/api.py

from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo import MongoClient

# ------------------- Config -------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "telematics")
COLL_NAME = os.getenv("MONGO_COLL", "risk_scores")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("XGB_MODEL", ROOT_DIR / "models" / "xgb_model.json"))

FEATURE_ORDER = [
    "trip_len_km_x", "trip_duration_min_x", "avg_speed_kph_x", "std_speed_x",
    "jerk_std_x", "hard_brake_rate_x", "hard_accel_rate_x",
    "pct_time_speeding_x", "max_speed_over_kph_x",
]
INPUT_TO_MODEL = {
    "trip_len_km": "trip_len_km_x",
    "trip_duration_min": "trip_duration_min_x",
    "avg_speed_kph": "avg_speed_kph_x",
    "std_speed": "std_speed_x",
    "jerk_std": "jerk_std_x",
    "hard_brake_rate": "hard_brake_rate_x",
    "hard_accel_rate": "hard_accel_rate_x",
    "pct_time_speeding": "pct_time_speeding_x",
    "max_speed_over_kph": "max_speed_over_kph_x",
}

DEFAULT_RISK_THRESHOLD = 0.75
DEFAULT_AGG_MODE: Literal["exposure_cred", "ewma", "mean"] = "exposure_cred"
DEFAULT_EWM_ALPHA = 0.30
DEFAULT_CRED_K = 20.0
DEFAULT_EXPOSURE_FEATURE = "trip_len_km"

DEFAULT_BASE_MONTHLY = 120.0
DEFAULT_MIN_FACTOR = 0.85
DEFAULT_MAX_FACTOR = 1.25
DEFAULT_LOG_A = 6.0
DEFAULT_LOG_B = 0.55

# ------------------- App/DB/Model -------------------
app = FastAPI(title="Telematics Risk API", version="1.0.0")

mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLL_NAME]

booster = xgb.Booster()
if not MODEL_PATH.exists():
    raise RuntimeError(f"XGBoost model not found at {MODEL_PATH}")
booster.load_model(str(MODEL_PATH))

# ------------------- Schemas -------------------
class TripIn(BaseModel):
    user_id: int = Field(..., ge=0)
    trip_id: Optional[str] = None
    segment_id: Optional[str] = None
    trip_len_km: float
    trip_duration_min: float
    avg_speed_kph: float
    std_speed: float
    jerk_std: float
    hard_brake_rate: float
    hard_accel_rate: float
    pct_time_speeding: float
    max_speed_over_kph: float
    meta: Optional[Dict[str, Any]] = None

class TripOut(BaseModel):
    _id: str
    user_id: int
    trip_id: Optional[str] = None
    segment_id: Optional[str] = None
    risk_score: float
    timestamp: str
    features: Dict[str, Any]

class SummaryOut(BaseModel):
    user_id: int
    total_trips: int
    aggregation_mode: str
    overall_risk_pricing: float
    monthly_premium: float
    avg_trip_risk: float
    min_trip_risk: float
    max_trip_risk: float
    std_trip_risk: float
    pct_over_threshold: float
    trips_over_threshold: int
    series: Optional[List[Dict[str, Any]]] = None

# ------------------- Helpers -------------------
def _make_feature_row(payload: TripIn) -> pd.DataFrame:
    md = {INPUT_TO_MODEL[k]: getattr(payload, k) for k in INPUT_TO_MODEL}
    X = pd.DataFrame([[md[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict_risk(Xrow: pd.DataFrame) -> float:
    dm = xgb.DMatrix(Xrow)
    pred = float(booster.predict(dm)[0])
    return float(np.clip(pred, 0.0, 1.0))

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = df["_id"].apply(lambda oid: getattr(oid, "generation_time", pd.NaT))
    return df

def _get_exposure_from_features(feat: dict, key: str = DEFAULT_EXPOSURE_FEATURE) -> float:
    if not isinstance(feat, dict): return 0.0
    val = feat.get(key, feat.get(f"{key}_x", 0.0))
    try: return float(val or 0.0)
    except Exception: return 0.0

def _compute_overall_series(risk: pd.Series, exposure: pd.Series,
                            mode: str = DEFAULT_AGG_MODE, ewm_alpha: float = DEFAULT_EWM_ALPHA,
                            cred_k: float = DEFAULT_CRED_K, portfolio_mean: float = 0.5) -> pd.Series:
    r = risk.astype(float).clip(0, 1)
    if mode == "ewma":   return r.ewm(alpha=float(ewm_alpha)).mean().clip(0, 1)
    if mode == "mean":   return r.expanding().mean().clip(0, 1)
    wsum = exposure.astype(float).clip(lower=0.0).cumsum() + 1e-9
    wrsum = (exposure * r).cumsum()
    wmean = wrsum / wsum
    n = pd.Series(np.arange(1, len(r) + 1), index=r.index, dtype=float)
    cred_w = n / (n + float(cred_k))
    overall = cred_w * wmean + (1.0 - cred_w) * float(portfolio_mean)
    return overall.clip(0, 1)

def _behavior_factor_logistic(r: float, min_f: float, max_f: float, a: float, b: float) -> float:
    r = float(np.clip(r, 0.0, 1.0))
    sig = 1.0 / (1.0 + np.exp(-a * (r - b)))
    return float(min_f + (max_f - min_f) * sig)

def _price_from_overall(overall_risk: float, base_monthly: float = DEFAULT_BASE_MONTHLY,
                        min_factor: float = DEFAULT_MIN_FACTOR, max_factor: float = DEFAULT_MAX_FACTOR,
                        a: float = DEFAULT_LOG_A, b: float = DEFAULT_LOG_B) -> float:
    bf = _behavior_factor_logistic(overall_risk, min_factor, max_factor, a, b)
    return round(base_monthly * bf, 2)

def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc); out["_id"] = str(out.get("_id")); return out

# ------------------- Routes -------------------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH.name, "db": f"{DB_NAME}.{COLL_NAME}"}

@app.post("/trips", response_model=TripOut)
def score_and_store_trip(trip: TripIn):
    try:
        X = _make_feature_row(trip)
        risk = _predict_risk(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"scoring failed: {e}")

    features_payload = {
        **{k: getattr(trip, k) for k in INPUT_TO_MODEL},
        **{INPUT_TO_MODEL[k]: getattr(trip, k) for k in INPUT_TO_MODEL},
        "segment_id": trip.segment_id,
        "meta": trip.meta or {},
    }
    doc = {
        "user_id": trip.user_id,
        "trip_id": trip.trip_id,
        "segment_id": trip.segment_id,
        "risk_score": float(risk),
        "timestamp": datetime.utcnow().isoformat(),
        "features": features_payload,
    }
    _id = collection.insert_one(doc).inserted_id
    saved = collection.find_one({"user_id": trip.user_id, "_id": _id})
    return _serialize(saved)

@app.get("/users/{user_id}/trips", response_model=List[TripOut])
def list_trips(user_id: int, limit: int = Query(50, ge=1, le=500), skip: int = Query(0, ge=0)):
    cur = collection.find({"user_id": user_id}).sort([("_id", -1)]).skip(skip).limit(limit)
    return [_serialize(d) for d in cur]

@app.get("/users/{user_id}/summary", response_model=SummaryOut)
def user_summary(user_id: int,
                 agg_mode: Literal["exposure_cred", "ewma", "mean"] = DEFAULT_AGG_MODE,
                 ewm_alpha: float = DEFAULT_EWM_ALPHA, cred_k: float = DEFAULT_CRED_K,
                 base_monthly: float = DEFAULT_BASE_MONTHLY, min_factor: float = DEFAULT_MIN_FACTOR,
                 max_factor: float = DEFAULT_MAX_FACTOR, log_a: float = DEFAULT_LOG_A,
                 log_b: float = DEFAULT_LOG_B, include_series: bool = True):
    recs = list(collection.find({"user_id": user_id}).sort([("_id", 1)]))
    if not recs:
        raise HTTPException(status_code=404, detail="no trips for user")

    df = _ensure_timestamp(pd.DataFrame(recs)).sort_values("timestamp").reset_index(drop=True)
    risk = df["risk_score"].astype(float).clip(0, 1)
    exposure = df["features"].apply(lambda f: _get_exposure_from_features(f, DEFAULT_EXPOSURE_FEATURE))

    agg = list(collection.aggregate([{"$group": {"_id": None, "m": {"$avg": "$risk_score"}}}]))
    portfolio_mean = float(agg[0]["m"]) if agg else 0.5

    overall = _compute_overall_series(risk, exposure, agg_mode, ewm_alpha, cred_k, portfolio_mean)
    overall_final = float(overall.iloc[-1])
    premium_final = _price_from_overall(overall_final, base_monthly, min_factor, max_factor, log_a, log_b)

    out = {
        "user_id": user_id,
        "total_trips": int(len(df)),
        "aggregation_mode": agg_mode,
        "overall_risk_pricing": round(overall_final, 4),
        "monthly_premium": premium_final,
        "avg_trip_risk": round(float(risk.mean()), 4),
        "min_trip_risk": round(float(risk.min()), 4),
        "max_trip_risk": round(float(risk.max()), 4),
        "std_trip_risk": round(float(risk.std()), 4),
        "pct_over_threshold": round(float((risk > DEFAULT_RISK_THRESHOLD).mean() * 100), 2),
        "trips_over_threshold": int((risk > DEFAULT_RISK_THRESHOLD).sum()),
        "series": None,
    }

    if include_series:
        out["series"] = [
            {
                "timestamp": pd.to_datetime(df["timestamp"].iloc[i]).isoformat(),
                "risk": float(risk.iloc[i]),
                "overall": float(overall.iloc[i]),
                "premium": _price_from_overall(float(overall.iloc[i]),
                                               base_monthly, min_factor, max_factor, log_a, log_b),
                "exposure_km": float(exposure.iloc[i]),
                "trip_id": str(df.get("trip_id", pd.Series([""] * len(df))).iloc[i]),
            }
            for i in range(len(df))
        ]
    return out

# Optional: run directly without uvicorn module path issues
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
