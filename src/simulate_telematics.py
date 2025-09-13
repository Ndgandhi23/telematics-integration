#!/usr/bin/env python3
# /src/simulate_telematics.py
import csv, math, random, argparse, time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

# ---------------- geo utils ----------------
EARTH_R = 6371000.0  # meters

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

def interpolate_along_polyline(pts: List[Tuple[float,float]], target_spacing_m: float) -> List[Tuple[float,float]]:
    """Return resampled points at ~target_spacing_m along the polyline."""
    if len(pts) < 2:
        return pts[:]
    segs = []
    cum = [0.0]
    for i in range(len(pts)-1):
        d = haversine_m(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        segs.append(d); cum.append(cum[-1]+d)
    total = cum[-1]
    if total <= 0:
        return [pts[0]]
    n = max(2, int(round(total/target_spacing_m)) + 1)
    out = []
    tvals = [i*(total/(n-1)) for i in range(n)]
    si = 0
    for tv in tvals:
        while si < len(segs) and tv > cum[si+1]:
            si += 1
        if si >= len(segs):
            out.append(pts[-1]); continue
        seg_len = max(segs[si], 1e-9)
        frac = (tv - cum[si]) / seg_len
        lat = pts[si][0] + frac*(pts[si+1][0]-pts[si][0])
        lon = pts[si][1] + frac*(pts[si+1][1]-pts[si][1])
        out.append((lat, lon))
    return out

# ---------------- units ----------------
def mph_to_mps(mph: float) -> float:
    return mph * 0.44704

# ---------------- IO ----------------
def read_coords(path: str) -> Dict[str, List[Tuple[int,float,float]]]:
    """Read Stage A coords file into {segment_id: [(seq,lat,lng), ...]}"""
    segs = defaultdict(list)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            segs[row["segment_id"]].append((int(row["seq"]), float(row["lat"]), float(row["lng"])))
    out = {}
    for sid, items in segs.items():
        items.sort(key=lambda x: x[0])
        out[sid] = items
    return out

def read_attrs(path: str) -> Dict[Tuple[str,int], Dict[str,str]]:
    """Read Stage B attrs file into {(segment_id, seq): row}"""
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["segment_id"], int(row["seq"]))
            out[key] = row
    return out

# ---------------- driving style knobs ----------------
def style_params(style: str) -> Dict[str, float]:
    s = style.lower()
    if s == "calm":
        return dict(base_mu=0.80, var=0.08, stop_rate_km=0.8, brake_prob=0.004, accel_prob=0.004)
    if s == "aggressive":
        return dict(base_mu=1.10, var=0.18, stop_rate_km=0.3, brake_prob=0.02, accel_prob=0.02)
    return dict(base_mu=0.95, var=0.12, stop_rate_km=0.5, brake_prob=0.01, accel_prob=0.01)

def clamp(v, lo, hi): return hi if v>hi else lo if v<lo else v

# ---------------- core simulation ----------------
def base_limit_for_segment(seg_items, attrs_map, segment_id) -> Tuple[float, str]:
    """Median per-point speed limit in mph + best source tag"""
    vals, sources = [], []
    for seq, _, _ in seg_items:
        a = attrs_map.get((segment_id, seq))
        if not a: continue
        try:
            v = a.get("maxspeed_mph")
            if v not in ("", None):
                vals.append(float(v))
                sources.append(a.get("speed_limit_source",""))
        except: pass
    if not vals:
        return 30.0, "fallback_30"
    vals.sort()
    limit = vals[len(vals)//2]
    if sources:
        pref = ["tag", "nyc_default_", "imputed_by_highway"]
        def score(s): return next((i for i,p in enumerate(pref) if s.startswith(p)), 99)
        best = sorted(sources, key=lambda s:(score(s), s))[0]
    else:
        best = "unknown"
    return limit, best

def synth_speed_series_mps(n, limit_mph, style, spacing_m, limit_source, rnd) -> List[float]:
    """Simulate speeds at 1 Hz with noise, stops, and rare spikes."""
    p = style_params(style)
    base_factor = rnd.uniform(0.6, 1.2) * p["base_mu"]
    if not str(limit_source).startswith("tag"):
        base_factor *= 0.9
    base = mph_to_mps(limit_mph) * base_factor
    var = p["var"]

    speeds = [max(0.0, rnd.gauss(base, var*base))]
    stop_prob = p["stop_rate_km"] * (spacing_m / 1000.0)
    brake_prob, accel_prob = p["brake_prob"], p["accel_prob"]
    vmax = mph_to_mps(max(limit_mph*1.3, 35.0))

    i = 1
    while i < n:
        v_prev = speeds[-1]
        if rnd.random() < stop_prob:  # random stop
            stop_dur = rnd.randint(5, 20)
            for _ in range(stop_dur):
                speeds.append(0.0); i += 1
                if i >= n: break
            continue
        v = rnd.gauss(base, var*base)
        roll = rnd.random()
        if roll < brake_prob:
            v = max(0.0, v_prev - rnd.uniform(4.0, 7.0))
        elif roll < brake_prob + accel_prob:
            v = v_prev + rnd.uniform(3.0, 6.0)
        speeds.append(clamp(v, 0.0, vmax))
        i += 1
    return speeds[:n]

def derive_kinematics(speeds: List[float]) -> Tuple[List[float], List[float]]:
    """Compute acceleration & jerk from speed series."""
    acc = [0.0] + [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
    jerk = [0.0] + [acc[i] - acc[i-1] for i in range(1, len(acc))]
    return acc, jerk

def simulate_trip_on_segment(segment_id, seg_items, attrs_map, trip_seed, start_ts, spacing_m, style) -> List[Dict]:
    """Generate one trip on a road segment"""
    rnd = random.Random(trip_seed)
    pts = [(lat, lng) for _, lat, lng in seg_items]
    resampled = interpolate_along_polyline(pts, spacing_m)
    n = len(resampled)
    if n < 6:   # changed from 10 → 6
        return []

    limit_mph, src = base_limit_for_segment(seg_items, attrs_map, segment_id)
    speeds = synth_speed_series_mps(n, limit_mph, style, spacing_m, src, rnd)
    acc, jerk = derive_kinematics(speeds)

    ts = [start_ts + i for i in range(n)]
    odo = [0.0]
    for i in range(1, n):
        d = haversine_m(resampled[i-1][0], resampled[i-1][1], resampled[i][0], resampled[i][1])
        odo.append(odo[-1] + d)

    trip_id = f"{segment_id}__{trip_seed}"
    return [
        dict(trip_id=trip_id, segment_id=segment_id, ts=ts[i], seq=i,
             lat=round(resampled[i][0],7), lng=round(resampled[i][1],7),
             speed_mps=round(speeds[i],3), accel_mps2=round(acc[i],3),
             jerk_mps3=round(jerk[i],3), odometer_m=round(odo[i],2))
        for i in range(n)
    ]

# ---------------- CLI & main ----------------
def write_rows_csv(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if not rows: f.write(""); return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Stage C: Simulate telematics trips.")
    ap.add_argument("--coords", default="data/A_coords.csv")
    ap.add_argument("--attrs",  default="data/B_attrs.csv")
    ap.add_argument("--trips",  type=int, default=1000)
    ap.add_argument("--out",    default="data/C_trips.csv")
    ap.add_argument("--spacing_m", type=float, default=12.0)
    ap.add_argument("--seed",   type=int, default=123)
    ap.add_argument("--style",  choices=["average","calm","aggressive"], default="average")
    args = ap.parse_args()

    random.seed(args.seed)
    segs, attrs_map = read_coords(args.coords), read_attrs(args.attrs)
    if not segs: raise SystemExit(f"No segments found in {args.coords}")

    segment_ids = list(segs.keys())
    rows_out, now, skipped = [], int(time.time()), 0

    for i in range(args.trips):
        segment_id = random.choice(segment_ids)
        trip_seed = random.randint(0, 2**31-1)
        trip_rows = simulate_trip_on_segment(segment_id, segs[segment_id], attrs_map,
                                             trip_seed, now, args.spacing_m, args.style)
        if trip_rows:
            rows_out.extend(trip_rows)
            now += len(trip_rows) + random.randint(5, 60)
        else:
            skipped += 1
        if (i+1) % 100 == 0:
            print(f"[progress] {i+1}/{args.trips} trips, skipped={skipped}")

    write_rows_csv(args.out, rows_out)
    print(f"[done] wrote {len(rows_out)} rows from {args.trips} trips → {args.out}")
    print(f"[info] skipped {skipped} trips (too short)")

if __name__ == "__main__":
    main()
