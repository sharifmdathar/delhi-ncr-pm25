#!/usr/bin/env python3
"""
Station coordinate verification script.

Checks:
1. Blank coordinates
2. Swapped lat/lon (lat outside [-90, 90] or lon outside [-180, 180])
2. Stations outside Delhi-NCR bounding box
3. Suspiciously close pairs (expected: Lodhi Road IITM/IMD, Pusa DPCC/IMD)
4. Isolated stations (>20 km from nearest neighbor)
5. Optional: compare against reference CSV (station_id,lat,lon)
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from geopy.distance import geodesic

# Delhi-NCR bounding box (approximate)
LAT_MIN, LAT_MAX = 28.2, 29.0
LON_MIN, LON_MAX = 76.5, 77.8

# Expected close pairs (distance < 1 km)
EXPECTED_CLOSE = [
    ("DL017", "DL018"),  # Lodhi Road IITM / IMD
    ("DL030", "DL031"),  # Pusa DPCC / IMD
]

# Thresholds
CLOSE_THRESHOLD_KM = 1.0
ISOLATED_THRESHOLD_KM = 20.0
COORD_TOLERANCE_KM = 1.5  # for reference comparison


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two points in km using geopy."""
    return geodesic((lat1, lon1), (lat2, lon2)).km


def check_stations(df, reference_path=None, tolerance_km=COORD_TOLERANCE_KM):
    """Run all checks on the stations DataFrame."""
    issues = []
    warnings = []

    # 1. Blank coordinates
    blank_lat = df[df['lat'].isna()]['station_id'].tolist()
    blank_lon = df[df['lon'].isna()]['station_id'].tolist()
    if blank_lat:
        issues.append(f"Blank latitude: {blank_lat}")
    if blank_lon:
        issues.append(f"Blank longitude: {blank_lon}")

    # 2. Swapped lat/lon or invalid ranges
    for _, row in df.iterrows():
        sid = row['station_id']
        lat, lon = row['lat'], row['lon']
        if not (LAT_MIN <= lat <= LAT_MAX):
            issues.append(f"{sid}: latitude {lat} outside Delhi-NCR bounds [{LAT_MIN}, {LAT_MAX}]")
        if not (LON_MIN <= lon <= LON_MAX):
            issues.append(f"{sid}: longitude {lon} outside Delhi-NCR bounds [{LON_MIN}, {LON_MAX}]")
        if not (-90 <= lat <= 90):
            issues.append(f"{sid}: latitude {lat} outside valid range [-90, 90] — possible lat/lon swap")
        if not (-180 <= lon <= 180):
            issues.append(f"{sid}: longitude {lon} outside valid range [-180, 180] — possible lat/lon swap")

    # 3. Stations outside Delhi-NCR bbox
    outside = df[
        (df['lat'] < LAT_MIN) | (df['lat'] > LAT_MAX) |
        (df['lon'] < LON_MIN) | (df['lon'] > LON_MAX)
    ]['station_id'].tolist()
    if outside:
        warnings.append(f"Stations outside Delhi-NCR bbox: {outside}")

    # 3. Pairwise distances
    coords = df.set_index('station_id')[['lat', 'lon']].to_dict('index')
    stations = list(coords.keys())
    close_pairs = []
    for i, s1 in enumerate(stations):
        for s2 in stations[i+1:]:
            d = haversine_km(coords[s1]['lat'], coords[s1]['lon'],
                           coords[s2]['lat'], coords[s2]['lon'])
            if d < CLOSE_THRESHOLD_KM:
                close_pairs.append((s1, s2, d))

    if close_pairs:
        expected_set = set(tuple(sorted(p)) for p in EXPECTED_CLOSE)
        found_set = set(tuple(sorted((p[0], p[1]))) for p in close_pairs)
        unexpected = found_set - expected_set
        if unexpected:
            warnings.append(f"Unexpected close pairs (<{CLOSE_THRESHOLD_KM} km): {sorted(unexpected)}")
        missing_expected = expected_set - found_set
        if missing_expected:
            warnings.append(f"Expected close pairs not found: {sorted(missing_expected)}")

    # 4. Isolated stations
    isolated = []
    for s1 in stations:
        min_dist = min(
            haversine_km(coords[s1]['lat'], coords[s1]['lon'],
                       coords[s2]['lat'], coords[s2]['lon'])
            for s2 in stations if s2 != s1
        )
        if min_dist > ISOLATED_THRESHOLD_KM:
            isolated.append((s1, round(min_dist, 1)))
    if isolated:
        warnings.append(f"Isolated stations (> {ISOLATED_THRESHOLD_KM} km from nearest): {isolated}")

    # 4. Reference comparison
    if reference_path and Path(reference_path).exists():
        ref = pd.read_csv(reference_path)
        ref = ref.set_index('station_id')[['lat', 'lon']].to_dict('index')
        for sid in stations:
            if sid in ref:
                d = haversine_km(coords[sid]['lat'], coords[sid]['lon'],
                               ref[sid]['lat'], ref[sid]['lon'])
                if d > tolerance_km:
                    warnings.append(f"{sid}: {d:.2f} km from reference (>{tolerance_km} km)")

    # Report
    print(f"Total stations: {len(df)}")
    print(f"Issues: {len(issues)}")
    print(f"Warnings: {len(warnings)}")
    print()
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")
        print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        print()
    if not issues and not warnings:
        print("All checks passed!")

    return len(issues) == 0


def main():
    ap = argparse.ArgumentParser(description="Validate station coordinates")
    ap.add_argument("stations", help="Path to stations.csv")
    ap.add_argument("--reference", help="Reference CSV (station_id,lat,lon)")
    ap.add_argument("--tolerance", type=float, default=1.5, help="Max km from reference")
    args = ap.parse_args()

    df = pd.read_csv(args.stations)
    ok = check_stations(df, args.reference, args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()