#!/usr/bin/env python3
"""Spatial error analysis.

Two questions:
  1. Do the graph variants help the peripheral stations more than the central
     ones?  (A mechanism claim about what graph structure buys.)
  2. Where does the error concentrate, and does the graph change the ranking of
     stations?
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
    coords = ds["coords"]
    sids = list(ds["station_ids"])
    cat = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "cpcb_station_catalog.csv")
    cat = cat.set_index("station_id")

    # distance from the centroid of the network (a proxy for "peripheral")
    ctr = coords.mean(axis=0)
    rad = np.sqrt(((coords - ctr) ** 2).sum(axis=1))       # km-ish (deg)
    rad_km = rad * 111.0

    spatial = json.loads((OUT / "analysis_spatial.json").read_text())

    # per-station MAE per config, averaged over seeds
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

    # ---- 1. does the graph help the periphery more? ---------------------
    print("=== graph benefit vs distance from the network centroid ===")
    rows = []
    for arch, base, wind, dist in [
        ("STGCN", "stgcn_identity", "stgcn_wind_aware", "stgcn_distance"),
        ("GraphWaveNet", "graph_wavenet_identity", "graph_wavenet_wind_aware",
         "graph_wavenet_distance")]:
        if base not in per_cfg or wind not in per_cfg:
            continue
        d_wind = np.array([per_cfg[wind][s] - per_cfg[base][s] for s in sids])
        d_dist = np.array([per_cfg[dist][s] - per_cfg[base][s] for s in sids])
        r = np.corrcoef(rad_km, d_wind)[0, 1]
        rd = np.corrcoef(rad_km, d_dist)[0, 1]
        # split at the median radius
        med = np.median(rad_km)
        near = rad_km <= med
        rows.append({
            "arch": arch,
            "wind_graph_delta_near": float(d_wind[near].mean()),
            "wind_graph_delta_far": float(d_wind[~near].mean()),
            "dist_graph_delta_near": float(d_dist[near].mean()),
            "dist_graph_delta_far": float(d_dist[~near].mean()),
            "corr_radius_vs_wind_penalty": float(r),
            "corr_radius_vs_dist_penalty": float(rd),
        })
        print(f"\n{arch}  (positive delta = the graph is worse than no graph)")
        print(f"  wind-aware graph penalty: near {rows[-1]['wind_graph_delta_near']:+.2f}   "
              f"far {rows[-1]['wind_graph_delta_far']:+.2f}")
        print(f"  distance graph penalty : near {rows[-1]['dist_graph_delta_near']:+.2f}   "
              f"far {rows[-1]['dist_graph_delta_far']:+.2f}")
        print(f"  corr(radius, wind-aware penalty) = {r:+.3f}")
        print(f"  corr(radius, distance  penalty) = {rd:+.3f}")

    # ---- 2. where the error lives ---------------------------------------
    print("\n=== per-station MAE, best config vs baselines ===")
    base = "stgcn_identity"
    maes = np.array([per_cfg[base][s] for s in sids])
    order = np.argsort(-maes)
    cat_s = cat.reindex(sids)
    print(f"  {'station':<12}{'city':<10}{'radius km':>11}{'no graph':>11}{'wind':>9}{'dist':>9}")
    for i in order[:6]:
        print(f"  {sids[i]:<12}{str(cat_s['city'].get(sids[i], '?'))[:9]:<10}"
              f"{rad_km[i]:>11.1f}{maes[i]:>11.2f}"
              f"{per_cfg['stgcn_wind_aware'][sids[i]]:>9.2f}"
              f"{per_cfg['stgcn_distance'][sids[i]]:>9.2f}")
    print("  ... best:")
    for i in order[-4:]:
        print(f"  {sids[i]:<12}{str(cat_s['city'].get(sids[i], '?'))[:9]:<10}"
              f"{rad_km[i]:>11.1f}{maes[i]:>11.2f}"
              f"{per_cfg['stgcn_wind_aware'][sids[i]]:>9.2f}"
              f"{per_cfg['stgcn_distance'][sids[i]]:>9.2f}")

    out = {"per_station_mae": per_cfg,
           "radius_km": {s: float(r) for s, r in zip(sids, rad_km)},
           "graph_penalty_by_radius": rows}
    (OUT / "spatial_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'spatial_analysis.json'}")


if __name__ == "__main__":
    main()
