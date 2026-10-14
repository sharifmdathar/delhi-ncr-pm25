#!/usr/bin/env python3
"""Pull PM2.5 data from OpenAQ API (backup for data.gov.in).

Free, no key required. Rate-limited (~100 req/min).
Covers Delhi-NCR stations with historical data.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


API_BASE = "https://api.openaq.org/v2"
HEADERS = {"Accept": "application/json"}

# Delhi-NCR bounding box
LAT_MIN, LAT_MAX = 28.2, 28.9
LON_MIN, LON_MAX = 76.8, 77.8


def get_locations():
    """Fetch all OpenAQ locations in Delhi-NCR."""
    params = {
        "country": "IN",
        "coordinates": f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}",
        "limit": 1000,
        "page": 1,
    }
    all_locs = []
    while True:
        resp = requests.get(f"{API_BASE}/locations", params=params, headers=HEADERS)
        if resp.status_code == 429:
            time.sleep(60)
            continue
        resp.raise_for_status()
        data = resp.json()
        all_locs.extend(data["results"])
        if not data.get("meta", {}).get("found", 0) > len(all_locs):
            break
        params["page"] += 1
        time.sleep(0.5)
    return all_locs


def get_measurements(location_id, param="pm25", start="2019-01-01", end="2024-12-31"):
    """Fetch hourly PM2.5 for a location."""
    params = {
        "location_id": location_id,
        "parameter": param,
        "date_from": start,
        "date_to": end,
        "limit": 10000,
        "page": 1,
    }
    all_meas = []
    while True:
        resp = requests.get(f"{API_BASE}/measurements", params=params, headers=HEADERS)
        if resp.status_code == 429:
            time.sleep(60)
            continue
        resp.raise_for_status()
        data = resp.json()
        all_meas.extend(data["results"])
        if not data.get("meta", {}).get("found", 0) > len(all_meas):
            break
        params["page"] += 1
        time.sleep(0.5)
    return all_meas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/openaq_pm25.csv")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    print("Fetching Delhi-NCR locations from OpenAQ...")
    locations = get_locations()
    print(f"Found {len(locations)} locations")

    # Filter for PM2.5
    pm25_locs = [l for l in locations if any(p["parameter"] == "pm25" for p in l.get("parameters", []))]
    print(f"Locations with PM2.5: {len(pm25_locs)}")

    all_rows = []
    for loc in pm25_locs:
        loc_id = loc["id"]
        name = loc["name"]
        lat, lon = loc["coordinates"]["latitude"], loc["coordinates"]["longitude"]
        print(f"  Fetching {name} ({loc_id})...")
        try:
            meas = get_measurements(loc_id, start=args.start, end=args.end)
            for m in meas:
                all_rows.append({
                    "station_id": f"OPENAQ_{loc_id}",
                    "station_name": name,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": m["date"]["utc"],
                    "pm25": m["value"],
                    "unit": m["unit"],
                    "source": "OpenAQ",
                })
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.5)  # rate limit

    if all_rows:
        df = pd.DataFrame(all_rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} rows to {args.out}")
    else:
        print("No data retrieved")


if __name__ == "__main__":
    main()