#!/usr/bin/env python3
"""Fetch ERA5 reanalysis wind for every station, from the Open-Meteo archive.

Why this exists
---------------
Nine of the 42 stations report no wind at all in the CPCB files, and the
per-station wind vectors that do exist agree only weakly (mean cross-station
correlation of the u-component is 0.11).  The wind-aware graph is therefore
built on a noisy, partly missing local wind field.  ERA5 is a reanalysis, so
it has the opposite properties: complete coverage, but a coarse grid (about
25 km), which near-flattened the field across Delhi.

Fetching ERA5 for the same stations therefore gives a clean second treatment
of "what if the edge directions came from a smooth, complete wind field
instead of a noisy, partly missing one?".  It is an ablation on the *source*
of the wind, not on the graph.

The API is free and needs no registration.  Times are returned in IST
(Asia/Kolkata) so they line up with the CPCB local-time records.

Usage
-----
    python scripts/fetch_era5_wind.py --years 2019 2020 2021 2022 2023 2024 2025

Output
------
    data/raw/era5_wind/era5_wind.csv
        station_id,timestamp,era5_wind_u,era5_wind_v,era5_wind_speed,era5_wind_dir
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "era5_wind"
API = "https://archive-api.open-meteo.com/v1/archive"

STATION_CSV = PROJECT_ROOT / "data" / "raw" / "station_coords.csv"


def load_coords():
    """station_id -> (lat, lon) for exactly the stations in the frozen dataset.

    The frozen dataset is authoritative: it is the station set the models were
    trained and evaluated on.  ``data/raw/station_coords.csv`` lists more
    stations than the dataset keeps and has unfilled coordinates for some of
    them, so it is only a fallback.
    """
    import numpy as np
    npz = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=True) as z:
            sids = [str(s) for s in z["station_ids"]]
            coords = z["coords"]            # (N, 2) = (lat, lon)
        return pd.DataFrame({
            "station_id": sids,
            "lat": coords[:, 0].astype(float),
            "lon": coords[:, 1].astype(float),
        })

    df = pd.read_csv(STATION_CSV)
    cols = {c.lower(): c for c in df.columns}
    sid = cols.get("station_id", df.columns[0])
    lat = cols.get("lat", cols.get("latitude"))
    lon = cols.get("lon", cols.get("longitude"))
    out = df[[sid, lat, lon]].rename(columns={sid: "station_id", lat: "lat", lon: "lon"})
    return out.dropna(subset=["lat", "lon"])


def fetch_chunk(lats, lons, year, models="era5", retries=4):
    """One API call for a list of coordinates over one calendar year."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Kolkata",
        "models": models,
        "wind_speed_unit": "ms",
    }
    url = API + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    import urllib.request
    import json
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:                       # noqa: BLE001
            last = e
            try:
                detail = e.read().decode()[:200]
            except Exception:                       # noqa: BLE001
                detail = ""
            print(f"    attempt {attempt+1}/{retries} failed: {e} {detail}")
            if "429" in str(e) or "Too Many Requests" in detail:
                time.sleep(65)                      # the API asks for one minute
            else:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo failed for {year}: {last}\n  URL: {url[:400]}")


def parse(payload, stations, year):
    """Turn one API response into a long-format dataframe."""
    res = payload if isinstance(payload, list) else [payload]
    frames = []
    for (sid, _lat, _lon), p in zip(stations, res):
        h = p.get("hourly")
        if not h:
            continue
        t = pd.to_datetime(h["time"])
        spd = np.asarray(h["wind_speed_10m"], dtype=float)
        dirn = np.asarray(h["wind_direction_10m"], dtype=float)
        wd = np.deg2rad(dirn)
        # Meteorological convention: WD is the direction the wind comes FROM.
        u = -spd * np.sin(wd)
        v = -spd * np.cos(wd)
        frames.append(pd.DataFrame({
            "station_id": sid,
            "timestamp": t,
            "era5_wind_u": u,
            "era5_wind_v": v,
            "era5_wind_speed": spd,
            "era5_wind_dir": dirn,
        }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--chunk", type=int, default=10, help="locations per API call")
    ap.add_argument("--out", default=str(OUT_DIR / "era5_wind.csv"))
    ap.add_argument("--force", action="store_true", help="refetch cached years")
    args = ap.parse_args()

    coords = load_coords()
    stations = list(coords.itertuples(index=False, name=None))
    print(f"Fetching ERA5 wind for {len(stations)} stations, {args.years}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for year in args.years:
        # per-year cache: the API rate-limits hard, so never refetch a year
        cache = OUT_DIR / f"era5_wind_{year}.csv"
        if cache.exists() and not args.force:
            print(f"  {year}: cached")
            all_frames.append(pd.read_csv(cache, parse_dates=["timestamp"]))
            continue
        parts = []
        for i in range(0, len(stations), args.chunk):
            chunk = stations[i:i + args.chunk]
            payload = fetch_chunk([c[1] for c in chunk], [c[2] for c in chunk], year)
            df = parse(payload, chunk, year)
            if len(df):
                parts.append(df)
            print(f"  {year}: stations {i}..{i+len(chunk)-1} -> {len(df):,} rows")
            time.sleep(2)
        if parts:
            yr = pd.concat(parts, ignore_index=True)
            yr.to_csv(cache, index=False)
            all_frames.append(yr)

    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(["station_id", "timestamp"])
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(out):,} rows, "
          f"{out['station_id'].nunique()} stations, "
          f"{out['timestamp'].min()} -> {out['timestamp'].max()}")


if __name__ == "__main__":
    main()
