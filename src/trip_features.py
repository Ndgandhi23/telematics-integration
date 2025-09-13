#!/usr/bin/env python3
# /src/make_features.py

import csv, argparse, statistics
from pathlib import Path
from collections import defaultdict

def read_points(path):
    """Read C_trips.csv into {trip_id: [row,...]}"""
    trips = defaultdict(list)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            trips[row["trip_id"]].append(row)
    return trips

def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def compute_features(trip_rows, attrs_map):
    """Aggregate one trip (list of rows) into feature dict"""
    n = len(trip_rows)
    if n < 5:
        return None

    speeds = [safe_float(r["speed_mps"]) for r in trip_rows]
    acc = [safe_float(r["accel_mps2"]) for r in trip_rows]
    jerk = [safe_float(r["jerk_mps3"]) for r in trip_rows]
    odo = [safe_float(r["odometer_m"]) for r in trip_rows]

    trip_len_m = max(odo) - min(odo)
    trip_dur_s = n

    # base stats
    avg_speed = sum(speeds)/n
    std_speed = statistics.pstdev(speeds)
    jerk_std = statistics.pstdev(jerk)

    # hard events
    hard_brakes = sum(1 for a in acc if a < -3.0)
    hard_accels = sum(1 for a in acc if a > +3.0)

    # derive rates per km
    km = max(trip_len_m/1000.0, 0.001)
    hard_brake_rate = hard_brakes/km
    hard_accel_rate = hard_accels/km

    # speeding: need limit from attrs (use first point’s segment+seq)
    seg_id = trip_rows[0]["segment_id"]
    seq0 = int(trip_rows[0]["seq"])
    key = (seg_id, seq0)
    limit = None
    if key in attrs_map:
        try:
            limit = float(attrs_map[key].get("maxspeed_mph", 30.0))
        except: pass
    if not limit: limit = 30.0
    limit_mps = limit * 0.44704
    speeding_time = sum(1 for v in speeds if v > 1.05*limit_mps)
    pct_time_speeding = speeding_time / n

    return {
        "trip_id": trip_rows[0]["trip_id"],
        "segment_id": seg_id,
        "trip_len_km": round(trip_len_m/1000.0, 3),
        "trip_duration_min": round(trip_dur_s/60.0, 2),
        "avg_speed_kph": round(avg_speed*3.6, 2),
        "std_speed": round(std_speed, 3),
        "jerk_std": round(jerk_std, 3),
        "hard_brake_rate": round(hard_brake_rate, 3),
        "hard_accel_rate": round(hard_accel_rate, 3),
        "pct_time_speeding": round(pct_time_speeding, 3),
        "max_speed_over_kph": round(max(speeds)*3.6 - limit, 2)
    }

def read_attrs(path):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["segment_id"], int(row["seq"]))
            out[key] = row
    return out

def main():
    ap = argparse.ArgumentParser(description="Stage D: Feature engineering from trips")
    ap.add_argument("--trips", default="data/C_trips.csv")
    ap.add_argument("--attrs", default="data/B_attrs.csv")
    ap.add_argument("--out",   default="data/D_trip_features.csv")
    args = ap.parse_args()

    trips = read_points(args.trips)
    attrs_map = read_attrs(args.attrs)

    features = []
    for tid, rows in trips.items():
        feat = compute_features(rows, attrs_map)
        if feat: features.append(feat)

    # write out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(features[0].keys()))
        w.writeheader(); w.writerows(features)

    print(f"[done] wrote {len(features)} trip-level features → {args.out}")

if __name__ == "__main__":
    main()
