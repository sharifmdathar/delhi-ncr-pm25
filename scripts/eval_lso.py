#!/usr/bin/env python3
"""Leave-station-out evaluation.

The main experiment asks whether a graph helps a station that the model has
already trained on.  This one asks the harder question: can a graph help a
station the model has *never seen*?

Protocol
--------
K of the 42 stations are held out.  Their targets never enter the training
loss, and (in the "both" mode) their own input history is hidden as well, so
the only information available for them is what their neighbours carry.

Two modes, because they ask different questions:
  target  the station's targets are unseen, its own 72-hour history is visible
          -> "can the model generalise to an unseen monitor?"
  both    the station's targets and its own history are unseen
          -> "can the graph substitute for a station's own history?"

Baselines are refitted honestly: ridge is fitted on the *training* stations
only and applied to the held-out ones, so it has never seen their data either.
"""
import sys
from pathlib import Path
from collections import defaultdict
import argparse
import json
import glob
import re

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset
from src.train.metrics import compute_metrics, compute_horizon_metrics
from scripts.evaluate import (denorm, persistence_predictions, seasonal_naive_predictions,
                              hour_of_week_predictions, model_predictions,
                              _clean_lags)

DATA = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs"


def ridge_on_train_stations(ds, starts, holdout_idx, lag=24, alpha=1.0):
    """Ridge fitted on the training stations only, applied to every station.

    The held-out stations' own records are never used to fit anything, so this
    is the fair counterpart of the neural models.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    y = ds["y_full"]
    mask = ds["mask_y"]
    tr = ds["split_indices_train"]
    tr = slice(int(tr[0]), int(tr[-1]) + 1)
    y_train, m_train = y[tr], mask[tr]
    n_st = y.shape[1]
    keep = [j for j in range(n_st) if j not in set(holdout_idx)]

    # pool the training stations to fit one shared lag->horizon map
    feats, targs = [], []
    for j in keep:
        series, obs = y_train[:, j], m_train[:, j] > 0
        for i in range(lag, len(series)):
            if obs[i - lag:i].all() and obs[i]:
                feats.append(series[i - lag:i])
                targs.append(series[i])
    sc = StandardScaler().fit(np.array(feats))
    mdl = Ridge(alpha=alpha).fit(sc.transform(np.array(feats)), np.array(targs))

    preds, targets, masks = [], [], []
    for s in starts:
        s = int(s)
        m, e = s + 72, s + 72 + 48
        pr = np.zeros((48, n_st))
        for j in range(n_st):
            w = _clean_lags(y, mask, m, j, lag).reshape(1, -1)
            pr[:, j] = mdl.predict(sc.transform(w))[0]
        preds.append(pr)
        targets.append(y[m:e])
        masks.append(mask[m:e])
    return (np.stack(preds), np.stack(targets), np.stack(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_windows", type=int, default=1500)
    ap.add_argument("--out", default="outputs/lso_results.json")
    args = ap.parse_args()

    ds = load_dataset(str(DATA))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    sids = [str(s) for s in ds["station_ids"]]

    idx = ds["split_indices_test"]
    lo, hi = int(idx[0]), int(idx[-1])
    starts = np.arange(lo, hi - 72 - 48 + 2)
    if len(starts) > args.max_windows:
        sel = np.linspace(0, len(starts) - 1, args.max_windows).astype(int)
        starts = starts[sel]

    # ---- gather the LSO checkpoints -------------------------------------
    ckpts = sorted(glob.glob(str(CKPT / "*_lso*.pt")))
    if not ckpts:
        print("No leave-station-out checkpoints found.")
        return

    by_run = defaultdict(list)
    holdout_map = {}
    for c in ckpts:
        blob = torch.load(c, map_location="cpu", weights_only=False)
        ho = list(blob.get("holdout_stations") or [])
        mode = blob.get("holdout_mode", "target")
        key = (tuple(sorted(ho)), mode)
        by_run[key].append(c)
        holdout_map[key] = ho

    results = []
    for (ho_ids, mode), files in sorted(by_run.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ho_idx = [sids.index(s) for s in ho_ids]
        ho_mask = np.zeros(len(sids), dtype=bool)
        ho_mask[ho_idx] = True
        print(f"\n=== holdout mode={mode}, {len(ho_ids)} stations: {ho_ids} ===")

        rows = []
        per_win = {}

        def add(name, p, t, m):
            p = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
            t = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
            sel = (m > 0) & ho_mask[None, None, :]
            if sel.sum() == 0:
                return
            # per-window errors over the held-out stations, for the bootstrap
            err = np.abs(p - t) * sel
            per_win[name] = (err.sum(axis=(1, 2)), sel.sum(axis=(1, 2)))
            overall = compute_metrics(p, t, sel)
            per_h = compute_horizon_metrics(p, t, sel)
            row = {"name": name, "mae": overall["mae"], "rmse": overall["rmse"],
                   "r2": overall["r2"], "n_obs": int(sel.sum())}
            for h in [1, 3, 6, 12, 24, 48]:
                row[f"h{h}"] = per_h[f"horizon_{h}h"]["mae"]
            rows.append(row)
            print(f"  {name:<44} MAE={row['mae']:7.2f}  R2={row['r2']:6.3f}  "
                  f"1h={row['h1']:6.2f}  48h={row['h48']:6.2f}")

        # baselines (never fitted on the held-out stations)
        p, t, m = persistence_predictions(ds, starts)
        add("Persistence", p, t, m)
        p, t, m = seasonal_naive_predictions(ds, starts, period=24)
        add("Seasonal naive (24h)", p, t, m)
        p, t, m = hour_of_week_predictions(ds, starts)
        add("Hour-of-week average", p, t, m)
        p, t, m = ridge_on_train_stations(ds, starts, ho_idx)
        add("Ridge (fit on training stations only)", p, t, m)

        # the neural models
        for f in files:
            try:
                p, t, m = model_predictions(ds, f, starts, "cuda" if torch.cuda.is_available() else "cpu")
                add(Path(f).stem, p, t, m)
            except Exception as e:                       # noqa: BLE001
                print(f"  {Path(f).name}: FAILED {e}")

        results.append({"mode": mode, "holdout_stations": list(ho_ids),
                        "n_holdout": len(ho_ids), "rows": rows})
        np.savez_compressed(
            OUT / f"lso_per_window_{mode}.npz",
            **{k: np.array(v, dtype=np.float64) for k, v in per_win.items()})

    # week index for the evaluation windows (for the paired bootstrap)
    import pandas as pd
    _wk = pd.to_datetime(ds["timestamps"][starts.astype(int) + 72], unit="s").isocalendar()
    np.save(OUT / "lso_window_week.npy",
             np.stack([_wk.year.values, _wk.week.values]).astype(np.int32))

    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
