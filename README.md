Telematics Integration in Auto Insurance

Setup

Clone the repo
git clone https://github.com/Ndgandhi23/telematics-integration.git

cd telematics-integration

Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

Make sure MongoDB is running locally (default: mongodb://localhost:27017).

How to Run

Executables are in the bin/ folder:

Insert scores into DB
python bin/insert_score_mongo.py

Analytics (Matplotlib plot)
python bin/analytics_user_risk_pricing.py

Interactive dashboard (Streamlit)
streamlit run bin/dashboard.py

REST API (FastAPI)
uvicorn src.api:app --reload
→ Docs: http://127.0.0.1:8000/docs

Model & Data

Model: XGBoost, trained on tabular trip features (regression).

Data: Pre-generated realistic telematics trips based on NYC roads.

Road geometry (Google SnapToRoads + OSMnx)

Second-by-second trip simulation (speed, accel, jerk)

Trip features (braking, speeding, smoothness)

Rule-based risk labels in [0, 1]

Artifacts are included:

models/xgb_model.json (trained model)

models/metrics.json (validation results)

data/ (Stage A–E outputs)

Notes

No external APIs required at runtime.

Everything runs locally with Python + MongoDB.

Dashboards and APIs let you explore risk scores and pricing.
