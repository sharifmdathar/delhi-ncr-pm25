#!/usr/bin/env python3
"""
Download CPCB yearly data from Vonter/india-cpcb-aqi GitHub releases and extract
Delhi + Faridabad stations to per-station hourly CSVs.

The release assets are Parquet files with 15-minute CPCB readings
(PM2.5 + station-level weather) for all Indian stations, 2009-2025.

Usage:
    python scripts/fetch_cpcb_years.py --years 2019 2020 2021 2022 2023 2024 2025
    python scripts/fetch_cpcb_years.py --keep-parquet   # don't delete after processing
"""
import argparse
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = "Vonter/india-cpcb-aqi"
RAW = Path("data/raw")
YEARS_DIR = RAW / "cpcb_years"
OUT_DIR = RAW / "cpcb_station_csv"

# Cities/states we care about (Delhi NCR)
TARGET_CITIES = {"Delhi", "Faridabad", "Ghaziabad", "Noida", "Gurgaon", "Gurugram",
                 "Greater Noida", "Bahadurgarh", "Anand Vihar"}
TARGET_STATES = {"Delhi", "Haryana", "Uttar Pradesh"}

# Cities kept in the modelling set (Delhi-NCR proper)
NCR_CITIES = {"Delhi", "Faridabad", "Ghaziabad", "Noida", "Gurgaon", "Gurugram"}

# Columns we keep
COLS = {
    "Station ID": "station_id",
    "State": "state",
    "City": "city",
    "Station Name": "station_name",
    "Timestamp": "timestamp",
    "PM2.5 (µg/m³)": "pm25",
    "AT (°C)": "temp_c",
    "RH (%)": "rh",
    "WS (m/s)": "wind_speed",
    "WD (deg)": "wind_dir",
    "RF (mm)": "rainfall",
}


def download(year: int, out_dir: Path) -> Path:
    """Download the Parquet asset for a year (or the CSV zip as fallback)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"cpcb-{year}.parquet"
    if pq.exists() and pq.stat().st_size > 10_000_000:
        print(f"  [skip] {pq.name} already present")
        return pq

    url = f"https://github.com/{REPO}/releases/download/{year}/cpcb-air-quality-{year}.parquet"
    print(f"  [get ] {url}")
    rc = subprocess.call(["curl", "-sL", "--retry", "3", "-o", str(pq), url])
    if rc != 0 or not pq.exists() or pq.stat().st_size < 1_000_000:
        if pq.exists():
            pq.unlink()
        raise RuntimeError(f"download failed for {year}")
    print(f"  [ok  ] {pq.name} ({pq.stat().st_size/1e6:.0f} MB)")
    return pq


def process_year(pq_path: Path, out_dir: Path) -> pd.DataFrame:
    """Extract target-region stations from a yearly Parquet file.

    Timestamps in the source are UTC. CPCB/Delhi operate on IST, so we convert
    to IST then drop tz to keep naive local time consistent with the rest of
    the pipeline.
    """
    print(f"  [read] {pq_path.name}")
    df = pd.read_parquet(pq_path)

    # Normalize column names (source has 'PM2.5 (µg/m³)' with unicode)
    ren = {}
    for src, dst in COLS.items():
        if src in df.columns:
            ren[src] = dst
        else:
            # fall back to a looser match
            for c in df.columns:
                if c == src or (dst == "pm25" and "PM2.5" in str(c)) or \
                   (dst == "temp_c" and str(c).startswith("AT ")) or \
                   (dst == "rh" and str(c).startswith("RH")) or \
                   (dst == "wind_speed" and str(c).startswith("WS ")) or \
                   (dst == "wind_dir" and str(c).startswith("WD ")) or \
                   (dst == "rainfall" and str(c).startswith("RF ")):
                    ren[c] = dst
                    break
    df = df.rename(columns=ren)

    keep = [c for c in ["station_id", "state", "city", "station_name", "timestamp",
                         "pm25", "temp_c", "rh", "wind_speed", "wind_dir", "rainfall"]
             if c in df.columns]
    df = df[keep].copy()

    # Filter to target region
    region = df[df["state"].isin(TARGET_STATES) | df["city"].isin(TARGET_CITIES)]
    if len(region) == 0:
        print("  [warn] no target-region rows")
        return pd.DataFrame()
    # UTC -> IST (+5:30), then make naive
    region["timestamp"] = pd.to_datetime(region["timestamp"], errors="coerce", utc=True)
    region = region.dropna(subset=["timestamp"])
    region["timestamp"] = region["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Numeric coercion
    for c in ["pm25", "temp_c", "rh", "wind_speed", "wind_dir", "rainfall"]:
        if c in region.columns:
            region[c] = pd.to_numeric(region[c], errors="coerce")

    # CPCB sentinels -> NaN
    if "pm25" in region.columns:
        region.loc[region["pm25"] < 0, "pm25"] = np.nan
        region.loc[region["pm25"] > 2000, "pm25"] = np.nan

    # Station id -> our naming; use CPCB site id when present
    if "station_id" not in region.columns:
        region["station_id"] = region["station_name"]

    print(f"  [keep] {len(region)} rows, {region['station_id'].nunique()} stations, "
          f"{region['timestamp'].min()} -> {region['timestamp'].max()}")

    return region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--keep-parquet", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    YEARS_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in args.years:
        print(f"\n=== {year} ===")
        try:
            pq = download(year, YEARS_DIR)
            region = process_year(pq, out_dir)
            if len(region):
                frames.append(region)
            if not args.keep_parquet:
                pq.unlink()
                print(f"  [del ] {pq.name}")
        except Exception as e:
            print(f"  [FAIL] {year}: {e}")
            continue

    if not frames:
        print("no data collected")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["station_id", "timestamp"])

    # Keep only Delhi-NCR cities for the modelling set
    ncr = combined[combined["city"].isin(NCR_CITIES)].copy()
    print(f"\n=== Combined: {len(combined)} rows, {combined['station_id'].nunique()} stations ===")
    print(f"=== Delhi-NCR subset: {len(ncr)} rows, {ncr['station_id'].nunique()} stations ===")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, g in ncr.groupby("station_id"):
        g = g.sort_values("timestamp")
        safe = str(sid).replace("/", "-")
        g.to_csv(out_dir / f"{safe}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    cat = combined[["station_id", "state", "city", "station_name"]].drop_duplicates(subset=["station_id"])
    cat.to_csv(RAW / "cpcb_station_catalog.csv", index=False)
    print(f"\n=== Station catalog: {len(cat)} stations -> {RAW/'cpcb_station_catalog.csv'} ===")
    print(cat[cat["city"].isin(NCR_CITIES)].to_string(index=False))


if __name__ == "__main__":
    import numpy as np
    main()
