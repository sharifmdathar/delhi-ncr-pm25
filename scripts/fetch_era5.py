#!/usr/bin/env python
"""
Download ERA5 hourly surface weather for Delhi-NCR (2019-2024) into data/raw/era5/.

CDS rejects requests whose "cost" is too large ("cost limits exceeded"), and the cost
is not simply proportional to what you'd expect. So this script is adaptive: it asks
for a whole year first and, if CDS says the request is too large, splits it into
halves (6 months, 3 months, ... down to 1 month) automatically. Finished files are
skipped on re-run, so it is safe to stop and restart.

Instant variables and accumulated precipitation are requested separately so each
request returns a single plain .nc file.

One-time setup: ~/.cdsapirc with your CDS key, ERA5 licence accepted,
pip install cdsapi xarray netCDF4
"""
from pathlib import Path

import requests

DATASET = "reanalysis-era5-single-levels"
OUT = Path("data/raw/era5")
# [North, West, South, East] -- must contain every station in stations.csv
AREA = [29.5, 76.5, 28.0, 78.0]
INSTANT = ["2m_temperature", "2m_dewpoint_temperature",
           "10m_u_component_of_wind", "10m_v_component_of_wind"]
ACCUM = ["total_precipitation"]
c = None          # cdsapi client, created in main()


def too_large(err):
    msg = str(err).lower()
    return "cost limits" in msg or "too large" in msg


def retrieve(variables, year, months, days, target):
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = Path(str(target) + ".part")
    c.retrieve(DATASET, request, str(tmp))
    tmp.replace(target)


def fetch_chunk(tag, variables, year, months, days=None, name=None):
    days = days or [f"{d:02d}" for d in range(1, 32)]
    target = OUT / (name or f"era5_{tag}_{year}_{months[0]:02d}-{months[-1]:02d}.nc")
    if target.exists() and target.stat().st_size > 0:
        print("skip", target.name)
        return
    print("requesting", target.name)
    try:
        retrieve(variables, year, months, days, target)
    except requests.HTTPError as e:
        if too_large(e) and len(months) > 1:
            mid = len(months) // 2
            print(f"  too large -> splitting months {months[0]}-{months[-1]} in two")
            fetch_chunk(tag, variables, year, months[:mid], days)
            fetch_chunk(tag, variables, year, months[mid:], days)
        else:
            raise


def main():
    global c
    import cdsapi
    OUT.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    for tag, variables in (("inst", INSTANT), ("tp", ACCUM)):
        # 31 Dec 2018 only: UTC->IST is +5:30, so 2019-01-01 00:00-05:00 IST needs it
        fetch_chunk(tag, variables, 2018, [12], days=["31"], name=f"era5_{tag}_2018.nc")
        for year in range(2019, 2025):
            fetch_chunk(tag, variables, year, list(range(1, 13)))
    print("done:", len(list(OUT.glob("*.nc"))), "files in", OUT)


if __name__ == "__main__":
    main()
