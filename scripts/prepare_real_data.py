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
import calendar
import glob
import re
import sys
from collections import Counter
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


MONTHS = {
    m.lower(): i for i, m in enumerate(calendar.month_name) if m
}
# Add abbreviated month names
MONTHS.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
})
HDR_RE = re.compile(r"^([A-Za-z]+)-(\d{2,4})$")


def read_wide(path):
    """OpenCity 2017-2023 layout: 'Year,YYYY' lines, then per month a header
    'Month-YYYY,00:00:00,...,23:00:00' followed by multiple rows per day:
    'pollutant_id,v0,...,v23' where pollutant_id=1 is PM2.5, 2=PM10, etc.
    We only keep pollutant_id=1 (PM2.5)."""
    year = month = None
    hours = list(range(24))
    stamps, values = [], []
    unknown = Counter()
    current_day = None
    seen_day = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            cells = [c.strip() for c in line.rstrip("\r\n").split(",")]
            c0 = cells[0].strip('"').strip()
            if not c0:
                continue
            if c0.lower() == "year":
                continue                                  # month headers carry the year too
            m = HDR_RE.match(c0)
            if m and m.group(1).lower() in MONTHS:
                month, year_str = MONTHS[m.group(1).lower()], m.group(2)
                year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
                hh = [int(c[:2]) for c in cells[1:] if re.match(r"^\d{2}:\d{2}", c)]
                hours = hh if len(hh) == 24 else list(range(24))
                continue
            if c0.isdigit() and year is not None:
                day_num = int(c0)
                try:
                    day0 = pd.Timestamp(year, month, day_num)
                except ValueError:                        # e.g. 31 in a 30-day month
                    continue
                # Only process the first pollutant row (PM2.5) for each day
                if day_num not in seen_day:
                    seen_day.add(day_num)
                    vals = pd.to_numeric(pd.Series(cells[1:25]), errors="coerce").tolist()
                    vals += [np.nan] * (24 - len(vals))
                    for h, v in zip(hours, vals):
                        stamps.append(day0 + pd.Timedelta(hours=h))
                        values.append(v)
                continue
            # Handle float-like row labels (e.g., "1.0", "2.0")
            try:
                day_num_f = float(c0)
                day_num = int(day_num_f)
                if day_num_f == day_num and year is not None:
                    try:
                        day0 = pd.Timestamp(year, month, day_num)
                    except ValueError:
                        continue
                    if day_num not in seen_day:
                        seen_day.add(day_num)
                        vals = pd.to_numeric(pd.Series(cells[1:25]), errors="coerce").tolist()
                        vals += [np.nan] * (24 - len(vals))
                        for h, v in zip(hours, vals):
                            stamps.append(day0 + pd.Timedelta(hours=h))
                            values.append(v)
                    continue
            except ValueError:
                pass
            unknown[c0] += 1
    if unknown:
        sys.exit(f"{path}: unexpected row labels {dict(unknown.most_common(5))}. "
                 f"The file may hold several pollutants -- paste me a sample around them.")
    s = pd.Series(values, index=pd.DatetimeIndex(stamps), dtype=float).sort_index()
    if s.index.has_duplicates:
        print(f"  WARNING: duplicate timestamps in {Path(path).name}; averaging them")
        s = s.groupby(level=0).mean()
    return s


def read_station_file(path, ts_col, pm_col, dayfirst, tz_is_utc=False):
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    if first.startswith("Year,"):
        s = read_wide(path)
        s = s[s.notna()]
        print(f"  {Path(path).name}: wide hourly layout, {len(s):,} values, "
              f"{s.index.min()} -> {s.index.max()}")
        return s
    df = pd.read_csv(path)
    ts_col = ts_col or _find(df.columns, TS_NAMES)
    pm_col = pm_col or _find(df.columns, PM_NAMES)
    if ts_col is None or pm_col is None:
        sys.exit(f"Can't find timestamp / PM2.5 column in {path}.\n"
                 f"Columns are: {list(df.columns)}\n"
                 f"Re-run with --ts-col and --pm-col.")
    ts = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=dayfirst)
    if getattr(ts.dt, "tz", None) is not None:
        if tz_is_utc:                                     # genuine UTC -> IST wall clock
            ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:                                             # label says +0000 but clock is IST
            ts = ts.dt.tz_localize(None)
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
                                   args.pm_col, args.dayfirst, args.tz_is_utc)
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
    print("\nNetwork-wide mean PM2.5 by year (sanity check: Delhi is typically ~90-110 ug/m3 per year;")
    print("values near 170-200 would mean the 2017-2023 files hold AQI index values, not concentrations):")
    print(out.groupby("year")["pm25"].mean().round(0).to_string())
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
    a.add_argument("--tz-is-utc", action="store_true",
                   help="timestamps with +0000 really are UTC (default: treat the clock as IST)")
    a.set_defaults(fn=build_pm25)
    b = sub.add_parser("weather")
    b.add_argument("--era5", default="data/raw/era5")
    b.add_argument("--stations", default="data/raw/stations.csv")
    b.set_defaults(fn=build_weather)
    args = ap.parse_args()
    args.fn(args)
