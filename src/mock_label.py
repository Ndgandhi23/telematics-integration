#!/usr/bin/env python3
# /src/mock_label.py

import csv
import argparse
from pathlib import Path
import numpy as np

def score_trip(row):
    try:
        avg_speed = float(row["avg_speed_kph"])
        pct_time_speeding = float(row["pct_time_speeding"])
        max_over = float(row["max_speed_over_kph"])
        brake = float(row["hard_brake_rate"])
        accel = float(row["hard_accel_rate"])
        jerk = float(row["jerk_std"])
        trip_len = float(row["trip_len_km"])
        trip_dur = float(row["trip_duration_min"])
    except Exception:
        return None

    # 🚦 Slightly reduced scoring weights
    speeding = 0.25 * min(1.2, pct_time_speeding) + 0.12 * min(1.5, max_over / 20)
    aggressive = 0.2 * np.sqrt(accel / 5) + 0.15 * np.sqrt(brake / 5)
    jerk_factor = 0.08 * np.sqrt(jerk / 3)
    exposure = 0.12 * np.log1p(trip_len / 10 + trip_dur / 30)

    risk = speeding + aggressive + jerk_factor + exposure

    # 🛑 Fatigue boost
    if trip_dur > 60 and (accel + brake > 6):
        risk += 0.04

    # ✅ Safe override (cap)
    if avg_speed < 30 and jerk < 0.5 and trip_len < 3:
        risk -= 0.08

    # 🔊 Realistic label noise
    risk += np.random.normal(0, 0.03)

    return max(0.0, min(1.0, risk))


def label_file(in_path, out_path):
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["risk_score"]
        rows = []
        for row in reader:
            score = score_trip(row)
            if score is None:
                continue
            row["risk_score"] = f"{score:.3f}"
            rows.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Stage E: Slightly lowered risk labeling")
    ap.add_argument("--features", default="data/D_trip_features.csv", help="Input features CSV")
    ap.add_argument("--out", default="data/E_labeled.csv", help="Output labeled CSV")
    args = ap.parse_args()
    label_file(args.features, args.out)
    print(f"[done] wrote risk labels to {args.out}")


if __name__ == "__main__":
    main()
