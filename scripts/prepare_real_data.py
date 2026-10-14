#!/usr/bin/env python
"""
Convert real data into the two CSVs your loader expects:

    data/raw/cpcb_hourly.csv   station_id,timestamp,pm25
    data/raw/weather.csv       station_id,timestamp,temp_c,rh,wind_u,wind_v,rainfall

All timestamps are written as NAIVE IST (UTC+5:30), the same clock CPCB uses,
so only ERA5 (which is UTC) needs converting.

You provide data/raw/stations.csv, one row per station:

    station_id,name,lat,lon,files
    DL001,Shadipur,<lat>,<lon>,fileA.csv;fileB.csv

`files` = one or more per-station CSVs (semicolon separated, relative to --src),
e.g. the 2017-2023 hourly file plus the 2024-25 15-minute file from OpenCity.

Usage:
    python prepare_real_data.py pm25 --src data/raw/opencity [--dayfirst]
    python fetch_era5.py                      # once, needs a free CDS account
    python prepare_real_data.py weather --era5 data/raw/era5
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
START, END = "2019-01-01", "2024-12-31 23:59"
TS_NAMES = ["timestamp", "from date", "datetime", "date", "time"]
PM_NAMES = ["pm2.5", "pm25", "pm 2.5"]


def _find(cols, names):
    low = {str(c).lower().strip(): c for c in cols}
    for n in names:                      # priority order
        for k, c in low.items():
            if n in k:
                return c
    return None


def read_station_file(path, ts_col, pm_col, dayfirst):
    df = pd.read_csv(path)
    ts_col = ts_col or _find(df.columns, TS_NAMES)
    pm_col = pm_col or _find(df.columns, PM_NAMES)
    if ts_col is None or pm_col is None:
        sys.exit(f"Can't find timestamp / PM2.5 column in {path}.\n"
                 f"Columns are: {list(df.columns)}\n"
                 f"Re-run with --ts-col and --pm-col.")
    ts = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=dayfirst)
    if getattr(ts.dt, "tz", None) is not None:          # tz-aware -> naive IST
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    pm = pd.to_numeric(df[pm_col], errors="coerce")
    s = pd.Series(pm.values, index=ts)
    s = s[s.index.notna()]
    print(f"  {Path(path).name}: {ts.notna().mean():.0%} timestamps parsed, "
          f"{s.index.min()} -> {s.index.max()} (cols: {ts_col!r}, {pm_col!r})")
    return s


def build_pm25(args):
    st = pd.read_csv(args.stations)
    frames = []
    for _, r in st.iterrows():
        print(f"{r['station_id']} ({r['name']})")
        parts = [read_station_file(Path(args.src) / f.strip(), args.ts_col,
                                   args.pm_col, args.dayfirst)
                 for f in str(r["files"]).split(";") if f.strip()]
        s = pd.concat(parts).sort_index()
        s = s.where((s > 0) & (s <= 1500))          # stuck zeros / absurd spikes -> NaN
        h = s.resample("1h").mean().loc[START:END].dropna()   # also handles 15-min files
        frames.append(pd.DataFrame({"station_id": r["station_id"],
                                    "timestamp": h.index,
                                    "pm25": h.values.round(2)}))
    out = pd.concat(frames).sort_values(["station_id", "timestamp"])
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW / "cpcb_hourly.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\nWrote {RAW/'cpcb_hourly.csv'}: {len(out):,} rows")

    # coverage report: % of hours present per station per year
    out["year"] = pd.to_datetime(out["timestamp"]).dt.year
    hours = out.groupby(["station_id", "year"]).size().unstack(fill_value=0)
    days = {y: (366 if y % 4 == 0 else 365) * 24 for y in hours.columns}
    print("\nCoverage (% of hours with data):")
    print((hours / pd.Series(days) * 100).round(0).astype(int).to_string())


def build_weather(args):
    import xarray as xr
    st = pd.read_csv(args.stations)
    files = sorted(glob.glob(str(Path(args.era5) / "*.nc")))
    if not files:
        sys.exit(f"No .nc files in {args.era5}. Run fetch_era5.py first.")
    ds = xr.merge([xr.open_dataset(f) for f in files], compat="no_conflicts")
    if "valid_time" in ds.coords:                       # newer CDS naming
        ds = ds.rename({"valid_time": "time"})
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    frames = []
    for _, r in st.iterrows():
        p = ds.sel(latitude=r["lat"], longitude=r["lon"], method="nearest").to_dataframe()
        t = p["t2m"] - 273.15
        td = p["d2m"] - 273.15
        rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
        df = pd.DataFrame({"temp_c": t, "rh": rh.clip(0, 100),
                           "wind_u": p["u10"], "wind_v": p["v10"],
                           "rainfall": p["tp"].clip(lower=0) * 1000})   # m -> mm
        # UTC -> IST is +5:30, which lands on :30. Interpolate onto the IST hour grid.
        df.index = df.index + pd.Timedelta(hours=5, minutes=30)
        grid = pd.date_range(df.index.min().ceil("h"), df.index.max().floor("h"), freq="1h")
        df = df.reindex(df.index.union(grid)).interpolate("time").reindex(grid)
        df = df.loc[START:END]
        df.insert(0, "station_id", r["station_id"])
        df.index.name = "timestamp"
        frames.append(df.reset_index())
    out = pd.concat(frames)[["station_id", "timestamp", "temp_c", "rh",
                             "wind_u", "wind_v", "rainfall"]]
    out.to_csv(RAW / "weather.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
               float_format="%.3f")
    print(f"Wrote {RAW/'weather.csv'}: {len(out):,} rows")
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pm25")
    a.add_argument("--src", default="data/raw/opencity")
    a.add_argument("--stations", default="data/raw/stations.csv")
    a.add_argument("--ts-col")
    a.add_argument("--pm-col")
    a.add_argument("--dayfirst", action="store_true",
                   help="timestamps are DD-MM-YYYY (check the printed min/max dates!)")
    a.set_defaults(fn=build_pm25)
    b = sub.add_parser("weather")
    b.add_argument("--era5", default="data/raw/era5")
    b.add_argument("--stations", default="data/raw/stations.csv")
    b.set_defaults(fn=build_weather)
    args = ap.parse_args()
    args.fn(args)
