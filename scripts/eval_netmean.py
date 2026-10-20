#!/usr/bin/env python3
"""Evaluate the network-mean diagnostic.

The question: is the only useful spatial information in this problem the
regional signal?  A no-graph model is given the network-wide mean PM2.5 as an
extra input channel.  If it matches the graph models, the graph contributes
nothing beyond the regional signal, which would explain both the main null
result and the leave-station-out result (where the graph lost to a
distance-weighted average of the neighbours).
"""
import sys
from pathlib import Path
import json
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset
from src.train.metrics import compute_metrics, compute_horizon_metrics
from scripts.evaluate import (denorm, model_predictions, persistence_predictions,
                              seasonal_naive_predictions, hour_of_week_predictions,
                              ridge_predictions)

DATA = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
HORIZONS = [1, 3, 6, 12, 24, 48]


def main():
    ds = load_dataset(str(DATA))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])

    idx = ds["split_indices_test"]
    lo, hi = int(idx[0]), int(idx[-1])
    starts = np.arange(lo, hi - 72 - 48 + 2)
    if len(starts) > 2000:
        sel = np.linspace(0, len(starts) - 1, 2000).astype(int)
        starts = starts[sel]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    per_win = {}

    def add(name, p, t, m):
        p = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
        t = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
        md = m > 0
        overall = compute_metrics(p, t, md)
        per_h = compute_horizon_metrics(p, t, md)
        row = {"name": name, "mae": overall["mae"], "rmse": overall["rmse"],
               "r2": overall["r2"]}
        for h in HORIZONS:
            row[f"h{h}"] = per_h[f"horizon_{h}h"]["mae"]
        err = np.abs(p - t) * md
        per_win[name] = (err.sum(axis=(1, 2)), md.sum(axis=(1, 2)))
        rows.append(row)

    # baselines and the reference models
    p, t, m = persistence_predictions(ds, starts)
    add("Persistence", p, t, m)
    p, t, m = seasonal_naive_predictions(ds, starts, period=24)
    add("Seasonal naive (24h)", p, t, m)
    p, t, m = hour_of_week_predictions(ds, starts)
    add("Hour-of-week average", p, t, m)
    p, t, m = ridge_predictions(ds, starts)
    add("Ridge(24 lags)", p, t, m)

    for ck in sorted(glob.glob(str(PROJECT_ROOT / "checkpoints" / "*netmean*.pt"))):
        p, t, m = model_predictions(ds, ck, starts, device)
        add(Path(ck).stem, p, t, m)

    # the reference points: the same architectures without the network mean
    for stem, label in [("stgcn_identity", "STGCN, no graph"),
                        ("stgcn_wind_aware", "STGCN, wind-aware"),
                        ("graph_wavenet_identity", "GraphWaveNet, no graph"),
                        ("graph_wavenet_wind_aware", "GraphWaveNet, wind-aware")]:
        cks = [c for c in sorted(glob.glob(
                    str(PROJECT_ROOT / "checkpoints" / f"{stem}_s*.pt")))
               if "_lso" not in c and "_netmean" not in c and "_fold" not in c]
        if not cks:
            continue
        accs = []
        for c in cks:
            p, t, m = model_predictions(ds, c, starts, device)
            accs.append(compute_metrics(denorm(np.nan_to_num(p, nan=0.0), nm, ns),
                                         denorm(np.nan_to_num(t, nan=0.0), nm, ns),
                                         m > 0)["mae"])
        rows.append({"name": label, "mae": float(np.mean(accs)),
                     "rmse": None, "r2": None})

    rows.sort(key=lambda r: r["mae"])
    print(f"{'model':<44}{'MAE':>8}{'R2':>8}")
    print("-" * 60)
    for r in rows:
        r2 = f"{r['r2']:.3f}" if r.get("r2") is not None else "-"
        print(f"{r['name']:<44}{r['mae']:>8.2f}{r2:>8}")

    (PROJECT_ROOT / "outputs" / "eval_netmean.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(PROJECT_ROOT / "outputs" / "netmean_per_window.npz",
                        **{k: np.array(v, dtype=np.float64) for k, v in per_win.items()})
    wk = pd.to_datetime(ds["timestamps"][starts.astype(int) + 72], unit="s").isocalendar()
    np.save(PROJECT_ROOT / "outputs" / "netmean_window_week.npy",
            np.stack([wk.year.values, wk.week.values]).astype(np.int32))
    print(f"\nWrote outputs/eval_netmean.json")


if __name__ == "__main__":
    main()
