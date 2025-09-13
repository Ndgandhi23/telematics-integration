#!/usr/bin/env python3
import os, csv, time, math, random, argparse, json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import requests

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TIMEOUT_S = 30
QPS_SLEEP_S = 0.10  # be polite to Google APIs

def ensure_parent(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def random_point_in_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    min_lat, min_lng, max_lat, max_lng = bbox
    return (random.uniform(min_lat, max_lat), random.uniform(min_lng, max_lng))

def nudge_by_km(lat: float, lng: float, km: float) -> Tuple[float, float]:
    """Move roughly km diagonally; accounts for longitude shrink with latitude."""
    dlat = km / 111.0
    dlng = km / (111.0 * max(1e-6, abs(math.cos(math.radians(lat)))))
    # randomize sign for variety
    dlat *= random.choice([-1, 1])
    dlng *= random.choice([-1, 1])
    return (lat + dlat, lng + dlng)

def snap_segment(sess: requests.Session, lat: float, lng: float, km_span: float = 0.15) -> List[Dict[str, Any]]:
    """Snap a tiny 2-point segment near (lat,lng); return snappedPoints list."""
    lat2, lng2 = nudge_by_km(lat, lng, km_span)
    path = f"{lat},{lng}|{lat2},{lng2}"
    r = sess.get(
        "https://roads.googleapis.com/v1/snapToRoads",
        params={"path": path, "interpolate": "true", "key": API_KEY},
        timeout=TIMEOUT_S,
    )
    if r.status_code == 403:
        raise SystemExit(f"403 PERMISSION_DENIED from Roads API: {r.text[:200]}")
    r.raise_for_status()
    time.sleep(QPS_SLEEP_S)
    return r.json().get("snappedPoints", [])

def dedupe_round(points: List[Dict[str, Any]], precision: int = 6) -> List[Dict[str, Any]]:
    """Round lat/lng and drop duplicates while preserving order."""
    seen = set()
    out = []
    for p in points:
        lat = round(p["location"]["latitude"], precision)
        lng = round(p["location"]["longitude"], precision)
        key = (lat, lng)
        if key in seen:
            continue
        seen.add(key)
        q = dict(p)
        q["location"] = {"latitude": lat, "longitude": lng}
        out.append(q)
    return out

def harvest(
    out_csv: str,
    bbox: Tuple[float, float, float, float],
    segments: int,
    min_points: int,
    max_retries: int,
    km_span: float,
    seed: Optional[int] = None,
) -> None:
    if not API_KEY:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY first.")
    ensure_parent(out_csv)

    sess = requests.Session()
    fieldnames = [
        "segment_id","seq","lat","lng","place_id","source",
        "seed_lat","seed_lng","seed","attempt"
    ]
    append = Path(out_csv).exists()
    f = open(out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not append:
        w.writeheader()

    if seed is not None:
        random.seed(seed)

    seg_written = 0
    seg_id_base = int(time.time())

    while seg_written < segments:
        seg_seed = random.randint(0, 2**31 - 1)
        random.seed(seg_seed)
        seed_lat, seed_lng = random_point_in_bbox(bbox)

        last_err = None
        snapped_clean: List[Dict[str, Any]] = []
        used_attempt = 0

        for attempt in range(1, max_retries + 1):
            try:
                snapped = snap_segment(sess, seed_lat, seed_lng, km_span=km_span)
                snapped = dedupe_round(snapped, precision=6)
                if len(snapped) >= min_points:
                    snapped_clean = snapped
                    used_attempt = attempt
                    break
                last_err = f"too few points ({len(snapped)})"
            except requests.HTTPError as e:
                last_err = f"HTTP {e.response.status_code}: {e.response.text[:180]}"
            except Exception as e:
                last_err = str(e)
            # backoff
            time.sleep(min(2 ** (attempt - 1) * 0.25, 2.0))

        if not snapped_clean:
            print(f"[skip] seed ({seed_lat:.6f},{seed_lng:.6f}) -> {last_err}")
            continue

        segment_id = f"{seg_id_base}_{seg_written:06d}"
        for i, p in enumerate(snapped_clean):
            w.writerow({
                "segment_id": segment_id,
                "seq": i,
                "lat": p["location"]["latitude"],
                "lng": p["location"]["longitude"],
                "place_id": p.get("placeId"),
                "source": "google_snap_interpolate",
                "seed_lat": round(seed_lat, 6),
                "seed_lng": round(seed_lng, 6),
                "seed": seg_seed,
                "attempt": used_attempt,
            })
        f.flush()
        seg_written += 1
        if seg_written % 25 == 0:
            print(f"[progress] segments: {seg_written}/{segments}")

    f.close()
    print(f"[done] wrote {segments} segments to {out_csv}")

def run_preview(bbox, max_retries=5, km_span=0.15):
    """Your original preview flow: prints one JSON preview and exits."""
    if not API_KEY:
        raise SystemExit("Set GOOGLE_MAPS_API_KEY first.")
    sess = requests.Session()
    last_err: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        lat, lng = random_point_in_bbox(bbox)
        try:
            snapped = snap_segment(sess, lat, lng, km_span=km_span)
            if snapped:
                out = [{
                    "lat": p["location"]["latitude"],
                    "lng": p["location"]["longitude"],
                    "place_id": p.get("placeId")
                } for p in snapped]
                print(json.dumps({
                    "attempt": attempt,
                    "seed_point": {"lat": lat, "lng": lng},
                    "snapped_points_preview": out[:10],
                    "count": len(out)
                }, indent=2))
                return
            else:
                last_err = "empty snappedPoints"
        except requests.HTTPError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(min(2 ** (attempt - 1) * 0.25, 2.0))
    print(json.dumps({"error": f"Failed after {max_retries} attempts", "last_error": last_err}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harvest road-aligned coordinates (Stage A) or run a one-off preview.")
    ap.add_argument("--bbox", default="40.58,-74.05,40.90,-73.78",
                    help="min_lat,min_lng,max_lat,max_lng (default: NYC-tight)")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--preview", action="store_true", help="Just print one preview JSON and exit")
    ap.add_argument("--segments", type=int, default=200, help="How many segments to harvest (ignored in --preview)")
    ap.add_argument("--min_points", type=int, default=8, help="Minimum snapped points required to keep a segment")
    ap.add_argument("--km_span", type=float, default=0.15, help="Approx diagonal span of each tiny segment in km")
    ap.add_argument("--out", default="data/A_coords.csv", help="Output CSV (ignored in --preview)")
    ap.add_argument("--seed", type=int, default=None, help="Global RNG seed (optional)")
    args = ap.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    if args.preview:
        run_preview(bbox=bbox, max_retries=args.max_retries, km_span=args.km_span)
    else:
        harvest(out_csv=args.out, bbox=bbox, segments=args.segments,
                min_points=args.min_points, max_retries=args.max_retries,
                km_span=args.km_span, seed=args.seed)
