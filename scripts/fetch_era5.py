#!/usr/bin/env python
"""
Download ERA5 hourly surface weather for Delhi-NCR, 2019-2024, into data/raw/era5/.

One-time setup:
  1. Free account at https://cds.climate.copernicus.eu
  2. Open the ERA5 single-levels dataset page, accept its licence
  3. Copy the url/key snippet from your CDS profile into ~/.cdsapirc
  4. pip install cdsapi xarray netCDF4

Instant variables and accumulated precipitation are requested separately so each
request returns a single plain .nc file.
"""
from pathlib import Path

import cdsapi

OUT = Path("data/raw/era5")
OUT.mkdir(parents=True, exist_ok=True)

# [North, West, South, East] -- must contain every station in stations.csv
AREA = [29.3, 76.5, 28.0, 77.9]
# 2018 is included because UTC->IST shifts +5:30, so 2019-01-01 00:00-05:00 IST
# needs ERA5 values from 2018-12-31 UTC.
YEARS = range(2018, 2025)
INSTANT = ["2m_temperature", "2m_dewpoint_temperature",
           "10m_u_component_of_wind", "10m_v_component_of_wind"]
ACCUM = ["total_precipitation"]

c = cdsapi.Client()
for year in YEARS:
    for tag, variables in (("inst", INSTANT), ("tp", ACCUM)):
        target = OUT / f"era5_{tag}_{year}.nc"
        if target.exists():
            print("skip", target)
            continue
        print("requesting", target)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": variables,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(target),
        )
