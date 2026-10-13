#!/usr/bin/env python3
"""Pull Open-Meteo historical weather data for Delhi-NCR stations.

Uses Open-Meteo historical API (free, no registration).
- Get wind u/v, temperature, humidity, rainfall per station.

Usage:
    python scripts/pull_openmeteo.py

Output:
    data/raw/weather.csv -- hourly weather per station
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relativehumidity_2m": "rh",
    "windspeed_10m": "wind_speed_10m",
    "winddirection_10m": "wind_direction_10m",
    "precipitation": "rainfall",
}

# Delhi-NCR bounding box (approximate)
# Delhi, Noida, Gurgaon, Ghaziabad, Faridabad
DELPHI_LAT_MIN = 28.2
DELPHI_LAT_MAX = 28.8
DELHI_LON_MIN = 76.8
DELHI_LON_MAX = 77.8


def fetch_weather_api():
    """Fetch from Open-Meteo historical API.

    NOTE: Placeholder. In production, this would make real API calls
    using requests to the Open-Meteo API endpoint.
    """
    # Placeholder - would use requests to Open-Meteo API
    return None


def main():
    parser = argparse.ArgumentParser(description="Pull Open-Meteo weather data")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: data/raw/weather.csv)")
    args = parser.parse_args()

    df = fetch_weather_api()
    if df is not None:
        out_path = Path(args.out) if args.out else RAW_DIR / "weather.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved Open-Meteo data to {out_path}")
        return

    # Fallback: check for existing backfill data
    backfill = RAW_DIR / "weather_backfill.csv"
    if backfill.exists():
        df = pd.read_csv(backfill)
        print(f"Loaded weather backfill data from {backfill} "
              f"({len(df)} rows, {len(df.columns)} cols)")
        out_path = Path(args.out) if args.out else RAW_DIR / "weather.csv"
        df.to_csv(out_path, index=False)
        return

    print("ERROR: No weather data available. "
          "Please provide Open-Meteo API access or local backfill CSV.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()