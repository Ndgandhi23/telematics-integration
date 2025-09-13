import pandas as pd
import xgboost as xgb
import random
from pymongo import MongoClient
from pathlib import Path

# === Load XGBoost model ===
model_path = Path(__file__).resolve().parent.parent / "models" / "xgb_model.json"
model = xgb.Booster()
model.load_model(model_path)

# === Connect to MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["telematics"]
collection = db["risk_scores"]

# === Load trip features ===
data_path = Path(__file__).resolve().parent.parent / "data" / "D_trip_features.csv"
df = pd.read_csv(data_path)

# === Sample a row and drop ID columns ===
sample = df.sample(1).to_dict(orient="records")[0]
X = pd.DataFrame([sample])
X = X.drop(columns=["trip_id", "segment_id"], errors="ignore")

# === Rename features to match model training ===
X.columns = [col + "_x" for col in X.columns]

# === Convert to DMatrix ===
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
dX = xgb.DMatrix(X)

# === Predict risk score ===
risk_score = float(model.predict(dX)[0])

# === Assign random user_id ===
user_id = random.randint(0, 5)

# === Insert into MongoDB ===
entry = {
    "user_id": user_id,
    "trip_id": sample.get("trip_id"),
    "segment_id": sample.get("segment_id"),
    "risk_score": round(risk_score, 4),
    "features": sample
}
collection.insert_one(entry)

print(f"[✓] Inserted trip with risk_score={risk_score:.4f} for user_id={user_id}")
