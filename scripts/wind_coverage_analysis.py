#!/usr/bin/env python3
"""Does the wind-aware graph help only where wind is actually measured?

21% of the stations (9 of 42) report no wind at all in the CPCB files.  For
those stations the wind-aware graph is built from the network-mean wind
direction, which carries no local information.  If the wind-aware graph works
for the reason we claim, its advantage over the distance graph should be
concentrated at the stations that do have a measured wind.

We therefore split the test metrics by whether the target station has wind
observations, and compare the configurations within each subset.
"""
import sys
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset

OUT = PROJECT_ROOT / "outputs"
CONFIGS = ["stgcn_identity", "stgcn_wind_aware", "stgcn_distance",
           "graph_wavenet_identity", "graph_wavenet_wind_aware", "graph_wavenet_distance"]


def main():
    ds = load_dataset(str(PROJECT_ROOT / "data" / "processed" /
                           "delhi_ncr_pm25_2019_2025.npz"))
    sids = list(ds["station_ids"])
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    X = ds["X_full"]
    wu = np.nanmean(X[trs][:, :, 3], axis=0)
    has_wind = ~np.isnan(wu)
    n_with = int(has_wind.sum())
    print(f"stations with a measured prevailing wind (train split): {n_with}/{len(sids)}")
    print(f"  without: {[s for s, h in zip(sids, has_wind) if not h]}")
    print("  (those stations get the network-mean wind direction in the "
          "wind-aware graph)")

    # ---- per-station MAE from the spatial analysis (averaged over seeds) ----
    spatial = json.loads((OUT / "analysis_spatial.json").read_text())
    per_cfg = {}
    for cfg in CONFIGS:
        keys = [k for k in spatial if k.startswith(cfg + "_s")]
        if not keys:
            continue
        acc = defaultdict(list)
        for k in keys:
            for row in spatial[k]:
                acc[row["station"]].append(row["mae"])
        per_cfg[cfg] = {s: float(np.mean(v)) for s, v in acc.items()}

    # ---- compare the configurations within each subset ---------------------
    print("\n=== per-station MAE by whether the station has a measured wind ===")
    header = f"{'config':<30}{'has wind (n=%d)' % n_with:>20}{'no wind (n=%d)' % (len(sids)-n_with):>20}"
    print(header)
    print("-" * len(header))
    rows = []
    for cfg in CONFIGS:
        if cfg not in per_cfg:
            continue
        w = np.array([per_cfg[cfg][s] for s in sids])
        a = w[has_wind].mean()
        b = w[~has_wind].mean()
        print(f"{cfg:<30}{a:>20.2f}{b:>20.2f}")
        rows.append({"config": cfg, "mae_has_wind": float(a), "mae_no_wind": float(b)})

    print("\n=== penalties relative to the no-graph model, within each subset ===")
    for arch, base, wind, dist in [
        ("STGCN", "stgcn_identity", "stgcn_wind_aware", "stgcn_distance"),
        ("GraphWaveNet", "graph_wavenet_identity", "graph_wavenet_wind_aware",
         "graph_wavenet_distance")]:
        if base not in per_cfg:
            continue
        b = np.array([per_cfg[base][s] for s in sids])
        wv = np.array([per_cfg[wind][s] for s in sids])
        dv = np.array([per_cfg[dist][s] for s in sids])
        print(f"\n{arch}")
        for label, m in [("has wind", has_wind), ("no wind", ~has_wind)]:
            print(f"  {label:<9} wind-aware penalty {(wv[m]-b[m]).mean():+.3f}   "
                  f"distance penalty {(dv[m]-b[m]).mean():+.3f}   "
                  f"(wind-vs-distance {(wv[m]-dv[m]).mean():+.3f})")

    out = {"n_stations_with_wind": n_with,
           "n_stations": len(sids),
           "stations_without_wind": [s for s, h in zip(sids, has_wind) if not h],
           "per_config": rows}
    (OUT / "wind_coverage_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'wind_coverage_analysis.json'}")


if __name__ == "__main__":
    main()
