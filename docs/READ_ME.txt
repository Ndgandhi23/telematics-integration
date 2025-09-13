**Stage A — Harvest Road Geometry**
This stage creates road-aligned coordinate sequences using the Google Roads API (SnapToRoads). Random seed points are sampled inside a fixed New York City bounding box. Each seed is paired with a nearby offset to form a short path. That path is sent to SnapToRoads with interpolation enabled, which “snaps” the input points onto the nearest drivable road and fills in extra points along the actual road geometry.
Segments that do not meet a minimum density are retried with new seeds until quality is achieved. Each accepted point is written to data/A_coords.csv with the following fields:
segment_id — unique ID for the road segment


seq — order of the point within the segment


lat, lng — snapped coordinate on the road (after Google interpolation)


place_id — Google Place ID if available


source — identifies SnapToRoads as the source (google_snap_interpolate)


seed_lat, seed_lng — original random seed before snapping


seed — RNG seed used to generate the segment


attempt — retry counter if earlier seeds failed


All geometry is harvested in the New York City area. This provides a dense, varied urban road network for proof-of-concept experiments, though the same method can be extended to any other region by changing the bounding box.

**Stage B — Enrich with OSM Road Attributes**
In this stage, each coordinate from Stage A is matched to the nearest OpenStreetMap (OSM) road edge using OSMnx. We then extract, normalize, and impute attributes so that every segment has complete and consistent road context. The goal is to turn simple geometry into enriched records that describe what kind of road it is, how fast vehicles typically travel, and what physical characteristics the road has.
1. Road classification
OSM highway tags are mapped into a clean hierarchy (motorway, trunk, primary, secondary, tertiary, residential, service, living_street, or other). Each class also receives a numeric rank for easy sorting or modeling.
2. Speed limits
Speed data is standardized into miles per hour (mph) with explicit tracking of its source:
Direct OSM tags are treated as authoritative.


If no tag is present inside NYC, we assign a speed by sampling from a weighted set of realistic defaults (15–50 mph).


Outside NYC, a fallback table by highway type is used (e.g., motorway ≈ 105 kph → converted to mph).
 Each record stores both the chosen speed and a confidence label (high, medium, low).


3. Physical attributes
Additional OSM fields are parsed, cleaned, and normalized, including:
Lane count, oneway direction, surface type


Bridge/tunnel flags


Road width (converted to meters where possible)


Lighting (lit) and cycleway information bucketed into standard categories


4. Data integrity
Every coordinate is enriched even if some OSM lookups fail. Rows with missing or unparseable attributes are still written with placeholders, ensuring consistent alignment with Stage A. A counter tracks how many values came from tags, defaults, or imputations.
Output
The result is written to data/B_attrs.csv. Each row links back to the Stage A segment ID and now contains both raw OSM attributes and adjusted values (e.g., speed limits always present, lanes always numeric where possible).
By the end of Stage B, road segments are no longer just polylines — they carry rich, standardized metadata about their type, speed environment, and physical characteristics, forming a reliable foundation for downstream analysis.


**Stage C — Simulate Telematics (1 Hz)**
This stage turns each road segment into a synthetic, second-by-second trip timeline. It uses the Stage A geometry and Stage B speed context to produce realistic motion signals.
Inputs
 data/A_coords.csv (ordered polyline points) and data/B_attrs.csv (per-point speed limits and road metadata).
What we do
Resample the road line at ~12 m spacing to get evenly spaced points (improves timing realism and distance math).


Derive a base speed limit per segment as the median of available point-level limits; remember the source (tag vs. default/imputed).


Generate a 1 Hz speed series using a style profile (“average”, “calm”, “aggressive”) with: baseline speed around the limit, stochastic variance, random full stops, and occasional brake/accel bursts. If the limit wasn’t from a direct tag, we slightly down-weight baseline speed.


Compute kinematics by finite differences: per-second acceleration and jerk.


Accumulate odometer using great-circle distances between consecutive points.


Timestamp the trip as a contiguous sequence of Unix seconds.


Quality control: segments that resample to fewer than 6 points are skipped (too short to be meaningful).


Output
 data/C_trips.csv, one row per second, with fields:
trip_id — deterministic per segment and RNG seed


segment_id — source road segment


ts — Unix timestamp (seconds)


seq — index within the trip


lat, lng — resampled coordinates (7-digit precision)


speed_mps — simulated speed (m/s)


accel_mps2 — per-second acceleration (m/s²)


jerk_mps3 — per-second jerk (m/s³)


odometer_m — cumulative distance along the trip (meters)


The result is a clean, 1 Hz telematics timeseries per trip that reflects road context (via speed limits) and driver style, ready for downstream feature engineering.

**Stage D — Trip-Level Feature Engineering**
This stage condenses each 1 Hz trip timeline into a single feature row suitable for modeling.
Inputs
 data/C_trips.csv (per-second telematics for each trip) and data/B_attrs.csv (per-point road attributes; used to recover a local speed limit).
What we compute
 • Basic exposure: trip length (meters → kilometers) and duration (seconds → minutes).
 • Central tendency/variability: average speed (kph) and standard deviation of speed.
 • Smoothness/aggression: standard deviation of jerk (m/s³).
 • Event rates: counts of “hard” accelerations and brakes per kilometer, where hard events are defined by acceleration thresholds > +3.0 m/s² and < −3.0 m/s².
 • Speeding behavior: fraction of seconds above 105% of the local limit (limit taken from Stage B; converted to m/s for this check).
 • Peak overspeed: maximum speed over the limit reported as “max_speed_over_kph”.
 • Segment linkage: segment_id is carried through for traceability.
Output
 data/D_trip_features.csv, one row per trip, with fields:
• trip_id — unique trip identifier
 • segment_id — source road segment
 • trip_len_km — total distance traveled (km)
 • trip_duration_min — trip duration (minutes)
 • avg_speed_kph — mean speed over the trip (kph)
 • std_speed — standard deviation of speed
 • jerk_std — standard deviation of jerk
 • hard_brake_rate — hard brake events per km
 • hard_accel_rate — hard acceleration events per km
 • pct_time_speeding — share of seconds above 105% of the limit
 • max_speed_over_kph — maximum amount over the speed limit, reported in kph
Notes
 • Trips with fewer than 5 samples are discarded to avoid unstable statistics.
 • Speed limit is recovered using the trip’s first (segment_id, seq) lookup in Stage B.
 • Units are handled explicitly when computing speeding fractions; ensure consistency when interpreting “max_speed_over_kph.”
Stage E — Rule-Based Risk Labeling
This stage assigns a numeric risk label in [0, 1] to each trip using transparent, domain-inspired rules applied to the trip features.
Inputs
 data/D_trip_features.csv (one row per trip with speed, smoothness, event rates, exposure).
What we compute
 • Speeding component: combines percent of time speeding (capped) and peak overspeed, with higher weight on sustained speeding.
 • Aggressiveness: square-root–scaled rates of hard accelerations and hard brakes.
 • Smoothness penalty: square-root function of jerk variability.
 • Exposure: mild increase with trip length and duration using a log transform.
 • Fatigue boost: small add-on when duration is long and aggressive events are frequent.
 • Safe override: small reduction for clearly gentle, short trips.
 • Realism: add zero-mean Gaussian noise to mimic label imperfection.
 • Finalize: sum components, then clamp to the [0, 1] range.
Output
 data/E_labeled.csv, same columns as Stage D plus:
 • risk_score — final numeric label in [0, 1].
Notes
 • Weights are intentionally modest to avoid extreme labels and to keep the student model learnable.
 • The design favors sustained unsafe patterns (time speeding) over single spikes (max overspeed), and lightly rewards short, smooth trips.

**Stage F — Student Model Training (XGBoost)**
This stage fits a supervised learner to approximate the rule-based labels from Stage E. The handcrafted rules act as a teacher; the model serves as a student that generalizes those patterns into a predictive function.
Inputs
data/D_trip_features.csv — engineered trip features


data/E_labeled.csv — rule-based risk scores


Method
Teacher → student setup: labels come from rules; XGBoost learns them.


Schema cleanup: drop IDs and duplicates; coerce features to numeric.


Train/validation split: 80/20 with fixed seed.


Objective: squared-error regression, evaluated with MAE.


Early stopping: halts boosting when validation MAE stops improving.


Artifacts: model saved as models/xgb_model.json; metrics written to models/metrics.json.


Why XGBoost
Captures nonlinear interactions between features.


Strong performance on tabular data with limited tuning.


Robust to noisy labels via regularization and early stopping.


Produces interpretable feature importances.


Lightweight, efficient, and production-ready.


Outputs
xgb_model.json — trained model


metrics.json — validation metrics (MAE, R²)
insert_score_mongo.py — Insert a Scored Trip into MongoDB
This utility takes a random trip’s engineered features, scores it with the trained model, and saves the result to MongoDB.
Process:
Model load — XGBoost booster is read from models/xgb_model.json.


Database connection — Connects to the local telematics.risk_scores collection.


Feature sampling — Picks one trip from data/D_trip_features.csv, drops IDs, and renames columns to match the model schema.


Scoring — Converts the row into a DMatrix and predicts a risk score in the range [0, 1].


User assignment — Attaches a random user_id for demonstration.


Insertion — Stores a document with user_id, trip_id, segment_id, predicted risk_score, and original features.


Resulting document contains: user_id, trip_id, segment_id, risk_score, and a nested features object with trip_len_km, duration, average speed, variability, and other metrics.
analytics_user_risk_pricing.py — User Risk Analytics & Pricing Visualization
Purpose
 Generates a pricing-ready view of a single user’s risk over time by loading scored trips from MongoDB, computing an overall (rolling) risk using a chosen aggregation method, mapping that overall risk to a monthly premium via a logistic curve, and rendering a clean Matplotlib chart with optional hover tooltips.
Data source
 Reads from MongoDB telematics.risk_scores for a specified user_id. If no explicit timestamp exists, the ObjectID generation time is used. Exposure is pulled from the stored features (defaults to trip_len_km, with a fallback to trip_len_km_x).
Key computations
Trip risk series: per-trip risk_score clipped to [0, 1], ordered by timestamp.


Overall (rolling) risk: selectable aggregator


exposure_cred: exposure-weighted mean blended toward portfolio average with a credibility factor k


ewma: exponentially weighted moving average with configurable alpha


mean: expanding simple average


Pricing: converts overall risk to a monthly premium using a logistic factor with tunable base price, min/max factors, slope, and midpoint.


Summary stats: total trips, overall risk used for pricing, estimated premium, mean/min/max, standard deviation, recent trend, threshold exceedances and percentages.


Configuration knobs
USER_ID, Mongo URI/DB/collection names


Aggregation mode and parameters (EWM_ALPHA, CREDIBILITY_K, EXPOSURE_FEATURE)


Pricing curve parameters (BASE_MONTHLY, MIN_FACTOR, MAX_FACTOR, LOGISTIC_A, LOGISTIC_B)


Risk alert threshold for quick scanning (RISK_THRESHOLD)


Visual style (Matplotlib style sheet name), optional hover via mplcursors if installed


Output
Console summary of pricing-oriented metrics for the selected user


Matplotlib line chart showing trip risk over time, rolling overall risk, threshold line, and an in-figure summary card (final overall risk, estimated monthly premium, aggregation mode, trip counts, distribution stats). Hover tooltips (if available) display trip-level risk, exposure, rolling overall risk, and rolling premium.


Role in the project
 Provides an at-a-glance, pricing-centric view that connects model outputs (trip risks) to a user’s evolving overall risk and an interpretable premium, suitable for screenshots and stakeholder review.

**dashboard.py — Streamlit Risk & Pricing Dashboard**
Purpose
 Interactive dashboard that reads scored trips from MongoDB, computes a user’s overall risk using a selectable aggregator, maps that risk to a monthly premium via a logistic curve, and visualizes trends with an Altair chart and KPI cards.
Data source
 MongoDB collection telematics.risk_scores. Uses document timestamps if present, otherwise ObjectID time. Exposure is read from each document’s features (defaults to trip_len_km, with a *_x fallback).
Key features
User selector: choose user_id from the database.


Date filter: optional start/end date range.


Overall risk modes: Exposure+Credibility (exposure-weighted mean blended toward portfolio average), EWMA (α slider), or Mean (expanding average).


Pricing controls: base monthly amount, min/max behavior factors, logistic slope (a) and midpoint (b).


KPIs: Trips, Overall Risk (pricing), Estimated Monthly Premium, Average Trip Risk.


Chart: Trip Risk line with points, rolling Overall Risk line, and a visual threshold. Tooltips show time, trip risk, exposure, rolling overall, and rolling premium.


Table: timestamped trips with risk, exposure, rolling overall risk, and rolling premium.


Export: optional CSV download of the table.


Explainer: short text describing the overall-risk blend and logistic pricing mapping.


Configuration
 Mongo URI, DB and collection names; aggregation mode and parameters (EWMA α, credibility k); pricing parameters (base, min/max factors, logistic a and b); exposure feature key; risk threshold; visual style handled internally by Altair and Streamlit.
Outputs
 A web UI showing KPI metrics, an interactive risk timeline with rolling overall risk and threshold, and a data table per user. Optional CSV export provides the displayed rows for external analysis.

**api.py — FastAPI Scoring & Pricing API**
Purpose
 Serve trip scoring, persistence, retrieval, and pricing-oriented summaries over a MongoDB backend using a pre-trained XGBoost model.
Model & Data
Loads the booster from models/xgb_model.json at startup and clamps predictions to the [0, 1] range.


Uses MongoDB collection telematics.risk_scores; timestamps are stored as ISO-8601. If a record lacks a timestamp, the ObjectID time is used for ordering.


Configuration
MONGO_URI (default: mongodb://localhost:27017/), MONGO_DB (default: telematics), MONGO_COLL (default: risk_scores)


XGB_MODEL (path override for the model)


Defaults for summaries: risk threshold 0.75; aggregation mode exposure_cred; EWMA α 0.30; credibility k 20; exposure feature trip_len_km; pricing curve base 120, min factor 0.85, max factor 1.25, slope 6.0, midpoint 0.55.


Feature schema (request → model)
Inputs accepted: trip_len_km, trip_duration_min, avg_speed_kph, std_speed, jerk_std, hard_brake_rate, hard_accel_rate, pct_time_speeding, max_speed_over_kph, plus user_id, optional trip_id/segment_id/meta.


Internally mapped to the model’s expected columns with “_x” suffix (e.g., avg_speed_kph → avg_speed_kph_x).


Fixed feature order ensures stable scoring; non-numeric values are coerced to numbers with safe defaults.


Endpoints
GET /health — Liveness check with model filename and database info.


POST /trips — Scores a single trip payload, persists a document, and returns the saved record. The features block echoes both the original field names and the mapped “_x” keys for traceability.


GET /users/{user_id}/trips — Paged listing (limit/skip) of a user’s most recent scored trips, newest first.


GET /users/{user_id}/summary — Computes a pricing-ready overview for a user: rolling overall risk (chosen aggregator), estimated monthly premium (logistic curve), distribution stats, and an optional per-trip series of timestamped values (risk, overall, premium, exposure, trip_id).


Aggregation modes for summaries
exposure_cred: exposure-weighted rolling mean blended toward the portfolio average using N/(N+k) credibility.


ewma: exponentially weighted moving average with α control.


mean: expanding simple average across trips.


Pricing mapping
Overall risk in [0, 1] is transformed to a behavior factor between min and max via a logistic function; monthly premium equals base price times that factor. All parameters are adjustable via query string.


Behavior & guarantees
Predictions and overall risk are clipped to [0, 1].


Every successful score persists a document containing user_id, optional trip_id/segment_id, risk_score, timestamp, and a features object with both original and “_x” fields.


Summaries include portfolio blending based on the current collection’s average risk, enabling stable early pricing for users with few trips.

