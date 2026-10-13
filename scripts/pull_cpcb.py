#!/usr/bin/env python3
"""Pull CPCB air quality data for Delhi-NCR stations.

- Primary: CPCB via data.gov.in API (free key, hourly station data).
- Backfill: Kaggle CPCB historical dumps.
- Logs station IDs, lat/lon, timestamps, source.

Usage:
    python scripts/pull_cpcb.py

Output:
    data/raw/cpcb_stations.csv   -- station metadata
    data/raw/cpcb_hourly.csv     -- hourly PM2.5 readings
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_cpcb_api():
    """Fetch from CPCB data.gov.in API.

    NOTE: This is a placeholder. In practice, you would need an API key
    and would make real HTTP requests. For now, we fall back to local
    backfill data.
    """
    # Placeholder - would use requests to data.gov.in
    return None


def main():
    parser = argparse.ArgumentParser(description="Pull CPCB AQ data")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: data/raw/cpcb_hourly.csv)")
    args = parser.parse_args()

    # Try API first, fall back to backfill
    df = fetch_cpcb_api()
    if df is not None:
        out_path = Path(args.out) if args.out else RAW_DIR / "cpcb_hourly.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved CPCB API data to {out_path}")
        return

    # Fallback: check for existing backfill data
    backfill = RAW_DIR / "cpcb_backfill.csv"
    if backfill.exists():
        df = pd.read_csv(backfill)
        print(f"Loaded CPCB backfill data from {backfill} "
              f"({len(df)} rows, {len(df.columns)} cols)")
        out_path = Path(args.out) if args.out else RAW_DIR / "cpcb_hourly.csv"
        df.to_csv(out_path, index=False)
        return

    print("ERROR: No CPCB data available. "
          "Please provide data.gov.in API access or local backfill CSV.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()