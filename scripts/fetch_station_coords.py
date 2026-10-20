#!/usr/bin/env python3
"""Regenerate the station coordinate table.

The four small metadata tables under ``data/raw/`` are the only files in this
repository that are derived from the CPCB records, and they are tracked so
that the graphs can be built without a 1.3 GB download.  This script
regenerates them.

    python scripts/fetch_station_coords.py                    # from the tracked tables
    python scripts/fetch_station_coords.py --xkdr_key aqi_...  # from XKDR

The XKDR route is the authoritative one: XKDR maintains a registry of Indian
monitoring stations with coordinates, and it is the source the coordinates in
this repository came from.  It needs a free key (https://airquality.xkdr.org/signup).
Without a key the script falls back to the tracked tables and says so.

Licence
-------
The station metadata is derived from the CPCB continuous monitoring network,
which is published under the Open Database Licence (ODbL 1.0).  These files
are offered under the same licence -- see LICENCE.md.
"""
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
XKDR_BASE = "https://airquality.xkdr.org/v1"


def from_xkdr(key):
    """Fetch the station registry from XKDR."""
    import json
    import urllib.request
    url = f"{XKDR_BASE}/stations"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode())
    rows = []
    for s in payload.get("stations", payload if isinstance(payload, list) else []):
        rows.append({
            "station_id": s.get("id") or s.get("station_id"),
            "state": s.get("state", ""),
            "city": s.get("city", ""),
            "station_name": s.get("name") or s.get("station_name", ""),
            "lat": s.get("latitude", s.get("lat")),
            "lon": s.get("longitude", s.get("lon")),
        })
    return pd.DataFrame(rows)


def from_tracked():
    """Rebuild the four tables from the tracked catalogue and coordinates."""
    coords = pd.read_csv(RAW / "station_coords.csv")
    cat = pd.read_csv(RAW / "cpcb_station_catalog.csv")
    coords.to_csv(RAW / "stations.csv", index=False)
    cat.to_csv(RAW / "cpcb_stations.csv", index=False)
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xkdr_key", default=None,
                    help="XKDR API key (https://airquality.xkdr.org/signup)")
    args = ap.parse_args()

    if args.xkdr_key:
        try:
            df = from_xkdr(args.xkdr_key)
            if len(df) == 0:
                raise RuntimeError("XKDR returned no stations")
            df.to_csv(RAW / "station_coords.csv", index=False)
            print(f"Wrote {RAW/'station_coords.csv'}: {len(df)} stations from XKDR")
            from_tracked()
            return
        except Exception as e:                       # noqa: BLE001
            print(f"XKDR fetch failed ({e}); falling back to the tracked tables")

    df = from_tracked()
    print(f"Rebuilt {RAW/'stations.csv'} and {RAW/'cpcb_stations.csv'} "
          f"({len(df)} stations with coordinates)")


if __name__ == "__main__":
    main()
