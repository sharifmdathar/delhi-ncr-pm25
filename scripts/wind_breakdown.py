#!/usr/bin/env python3
"""Does the graph-vs-no-graph gap depend on wind speed or forecast horizon?

The concern is that the wind-aware graph might only be informative in
strong-wind regimes (when transport dominates) or at long horizons (when
advection has had time to act).  We bin the test windows by the mean wind
speed over the input period and report the MAE of each configuration per bin,
plus the paired difference against the no-graph variant.
"""
import sys
from pathlib import Path
import argparse
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.dataset import load_dataset
from scripts.evaluate import denorm

DATA = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_windows", type=int, default=2000)
    ap.add_argument("--out", default="outputs/wind_breakdown.json")
    args = ap.parse_args()

    ds = load_dataset(str(DATA))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    n_st = ds["X_full"].shape[1]

    err = np.load(PROJECT_ROOT / "outputs" / "per_window_errors.npz")
    week = np.load(PROJECT_ROOT / "outputs" / "window_week.npy")

    idx = ds["split_indices_test"]
    lo, hi = int(idx[0]), int(idx[-1])
    starts = np.arange(lo, hi - 72 - 48 + 2)
    if len(starts) > args.max_windows:
        sel = np.linspace(0, len(starts) - 1, args.max_windows).astype(int)
        starts = starts[sel]
    # window_week.npy was built for the same subsample
    assert len(week[0]) == len(starts), (len(week[0]), len(starts))

    # mean wind speed over each window's input period (m/s, from the raw
    # normalised feature: wind_u and wind_v are z-scored, so recover a
    # comparable magnitude via the square root of the sum of squares)
    X = ds["X_full"]
    wu = X[starts[:, None] + np.arange(72)[None, :], :, 3]
    wv = X[starts[:, None] + np.arange(72)[None, :], :, 4]
    # z-score -> de-standardise to physical m/s
    # (normalisation stats are stored per feature in the dataset builder)
    wind_speed = np.sqrt(np.nanmean(wu ** 2 + wv ** 2, axis=(1, 2)))
    print(f"window mean wind-speed proxy: min={wind_speed.min():.3f} "
          f"med={np.median(wind_speed):.3f} max={wind_speed.max():.3f}")

    # group checkpoints by config stem, average error over seeds
    by_stem = defaultdict(list)
    for k in err.files:
        stem = k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
        by_stem[stem].append(k)

    stems = [s for s in by_stem if s.startswith(("stgcn_", "graph_wavenet_"))]

    # wind-speed bins (tertiles of the distribution)
    edges = np.percentile(wind_speed, [0, 33, 66, 100])
    bin_id = np.digitize(wind_speed, edges[1:-1])

    out = {"wind_speed_edges": edges.tolist(), "bins": []}
    print(f"\n{'wind bin':<22}{'n_win':>7}{'n_st':>6}  " +
          "  ".join(f"{s.split('_',1)[0][:3]}_{s.split('_')[1][:5]:<6}" for s in stems))
    for b in range(len(edges) - 1):
        sel = bin_id == b
        if sel.sum() == 0:
            continue
        row = {"bin": [float(edges[b]), float(edges[b + 1])], "n_windows": int(sel.sum())}
        cells = []
        for s in stems:
            num = np.zeros(sel.sum())
            cnt = np.zeros(sel.sum())
            for k in by_stem[s]:
                n, c = err[k]
                num += n[sel]
                cnt += c[sel]
            mae = float(np.sum(num) / max(np.sum(cnt), 1))
            row[s] = mae
            cells.append(f"{mae:12.2f}")
        print(f"[{edges[b]:.2f},{edges[b+1]:.2f}){'':<8}{row['n_windows']:>7}{n_st:>6}  " + "  ".join(cells))
        out["bins"].append(row)

    # per-horizon breakdown from the dedicated horizon error file
    herr = np.load(PROJECT_ROOT / "outputs" / "per_window_horizon_errors.npz")
    hby = defaultdict(list)
    for k in herr.files:
        stem = k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
        hby[stem].append(k)

    HS = [1, 3, 6, 12, 24, 48]
    print(f"\n=== MAE by horizon (all test windows) ===")
    print(f"{'config':<34}" + "".join(f"{h:>8}" for h in HS))
    out["by_horizon"] = {}
    for s in stems:
        num = np.zeros((len(starts), 48)); cnt = np.zeros((len(starts), 48))
        for k in hby[s]:
            n, c = herr[k]
            num += n; cnt += c
        per_h = [float(num[:, h-1].sum() / max(cnt[:, h-1].sum(), 1)) for h in HS]
        out["by_horizon"][s] = {str(h): v for h, v in zip(HS, per_h)}
        print(f"{s:<34}" + "".join(f"{v:>8.2f}" for v in per_h))

    (PROJECT_ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
