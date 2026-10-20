#!/usr/bin/env python3
"""Evaluate baselines and trained models on the frozen 2019-2025 dataset.

Reports masked metrics in original µg/m³, overall and per horizon, for:
  - Persistence (last observed value)
  - Ridge regression on the last K lags (the linear floor)
  - Every trained checkpoint in checkpoints/

Usage:
    python scripts/evaluate.py                       # all checkpoints
    python scripts/evaluate.py --models stgcn        # only these model names
    python scripts/evaluate.py --split test
"""

import sys
from pathlib import Path
import argparse
import json
import glob
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_dataset
from src.train.metrics import compute_metrics, compute_horizon_metrics
from src.data.splits import get_split
from scripts.train_all import build_model, _to_hn

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def denorm(a, mean, std):
    return np.expm1(np.clip(a * std + mean, -700, 700))


def persistence_predictions(ds, starts):
    """Last observed value repeated over the horizon."""
    y = ds["y_full"]
    mask = ds["mask_y"]
    preds, targets, masks = [], [], []
    for s in starts:
        s = int(s)
        m, e = s + 72, s + 72 + 48
        last = y[m - 1]
        # Walk back to the most recent observed value for each station
        for back in range(0, 72):
            cand = y[m - 1 - back]
            obs = mask[m - 1 - back]
            last = np.where(obs > 0, cand, last)
        preds.append(np.repeat(last[None, :], 48, axis=0))
        targets.append(y[m:e])
        masks.append(mask[m:e])
    return (np.stack(preds), np.stack(targets), np.stack(masks))


def _clean_lags(y, mask, m, j, lag):
    """Last `lag` values for station j, backfilled with the last observed value."""
    w = y[m - lag:m, j].astype(float).copy()
    obs = mask[m - lag:m, j] > 0
    # Find the most recent observed value (search backwards beyond the window too)
    last = np.nan
    for back in range(0, 200):
        idx = m - 1 - back
        if idx < 0:
            break
        if mask[idx, j] > 0:
            last = float(y[idx, j])
            break
    if not np.isfinite(last):
        last = 0.0
    for k in range(len(w) - 1, -1, -1):
        if obs[k] and np.isfinite(w[k]):
            last = w[k]
        else:
            w[k] = last
    return w


def ridge_predictions(ds, starts, lag=24, alpha=1.0):
    """Per-station ridge on the last `lag` hourly values, predicting each horizon.

    Fitted on the TRAIN split only.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    y = ds["y_full"]
    mask = ds["mask_y"]
    tr = ds["split_indices_train"]
    tr = slice(int(tr[0]), int(tr[-1]) + 1)
    y_train = y[tr]                      # (T, N)
    m_train = mask[tr]

    n_st = y.shape[1]
    models = []
    for j in range(n_st):
        series = y_train[:, j]
        obs = m_train[:, j] > 0
        feats, targs = [], []
        for i in range(lag, len(series)):
            w = series[i - lag:i]
            if obs[i - lag:i].all() and obs[i]:
                feats.append(w)
                targs.append(series[i])
        if len(feats) < 50:
            models.append(None)
            continue
        Xf = np.array(feats)
        yf = np.array(targs)
        sc = StandardScaler().fit(Xf)
        mdl = Ridge(alpha=alpha).fit(sc.transform(Xf), yf)
        models.append((sc, mdl))

    preds, targets, masks = [], [], []
    for s in starts:
        s = int(s)
        m, e = s + 72, s + 72 + 48
        pr = np.zeros((48, n_st))
        for j in range(n_st):
            if models[j] is None:
                pr[:, j] = y[m - 1, j]
                continue
            sc, mdl = models[j]
            w = _clean_lags(y, mask, m, j, lag).reshape(1, -1)
            pr[:, j] = mdl.predict(sc.transform(w))[0]
        preds.append(pr)
        targets.append(y[m:e])
        masks.append(mask[m:e])
    return (np.stack(preds), np.stack(targets), np.stack(masks))


def model_predictions(ds, ckpt_path, starts, device):
    """Run a trained checkpoint over the given windows."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    n_stations = ck["n_stations"]
    mname = ck.get("model")
    if mname is None:
        # filename is "<model>_<graph>_s<seed>.pt"
        mname = Path(ckpt_path).stem.rsplit("_s", 1)[0]
        for g in ("distance", "correlation", "wind_aware", "learned_adaptive", "identity"):
            if mname.endswith("_" + g):
                mname = mname[: -(len(g) + 1)]
                break
    model = build_model(mname, cfg, n_stations).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    X = torch.from_numpy(ds["X_full"])
    y = torch.from_numpy(ds["y_full"])
    mask = torch.from_numpy(ds["mask_y"])

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for s in starts:
            s = int(s)
            m, e = s + 72, s + 72 + 48
            xb = torch.nan_to_num(X[s:m], nan=0.0)
            mb = mask[s:m]
            xb = torch.cat([xb, mb.unsqueeze(-1)], dim=-1).unsqueeze(0).to(device)
            out = _to_hn(model(xb), 48, n_stations)
            preds.append(out.cpu().numpy())
            targets.append(y[m:e].numpy())
            masks.append(mask[m:e].numpy())
    return (np.stack(preds), np.stack(targets), np.stack(masks))


def score(name, preds, targets, masks, norm_mean, norm_std):
    p = denorm(np.nan_to_num(preds, nan=0.0), norm_mean, norm_std)
    t = denorm(np.nan_to_num(targets, nan=0.0), norm_mean, norm_std)
    m = masks > 0
    overall = compute_metrics(p, t, m)
    per_h = compute_horizon_metrics(p, t, m)
    row = {
        "name": name,
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "r2": overall["r2"],
    }
    for h in [1, 3, 6, 12, 24, 48]:
        row[f"h{h}"] = per_h[f"horizon_{h}h"]["mae"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split_mode", default="fixed", choices=["fixed", "expanding"])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--ckpt_suffix", default="", help="suffix to match checkpoint names")
    ap.add_argument("--max_windows", type=int, default=1500)
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(str(DATA_PATH))
    norm_mean = float(ds["norm_mean"])
    norm_std = float(ds["norm_std"])
    n_st = ds["X_full"].shape[1]

    split = get_split(ds, args.split_mode, args.fold)
    idx = split[args.split]
    starts = np.arange(int(idx[0]), int(idx[-1]) - 72 - 48 + 2)
    if len(starts) > args.max_windows:
        # Evenly spaced subsample so all months are represented
        sel = np.linspace(0, len(starts) - 1, args.max_windows).astype(int)
        starts = starts[sel]
    print(f"Split={args.split}: {len(starts)} windows, {n_st} stations")

    rows = []
    preds = {}

    print("\n--- Baselines ---")
    p, t, m = persistence_predictions(ds, starts)
    preds["Persistence"] = (p, t, m)
    rows.append(score("Persistence", p, t, m, norm_mean, norm_std))

    p, t, m = ridge_predictions(ds, starts)
    preds["Ridge(24 lags)"] = (p, t, m)
    rows.append(score("Ridge(24 lags)", p, t, m, norm_mean, norm_std))

    print("\n--- Models ---")
    for ck in sorted(glob.glob(str(CKPT_DIR / "*.pt"))):
        name = Path(ck).stem
        if "2019_2024" in name or "2024_2025" in name:
            continue          # stale checkpoints from the previous dataset
        mname = name.split("_s")[0]
        if args.fold and not name.endswith(args.fold):
            continue
        if args.ckpt_suffix and args.ckpt_suffix not in name:
            continue
        if args.models and not any(mm in name for mm in args.models):
            continue
        try:
            p, t, m = model_predictions(ds, ck, starts, device)
            preds[name] = (p, t, m)
            rows.append(score(name, p, t, m, norm_mean, norm_std))
        except Exception as e:
            print(f"  {name}: FAILED {e}")

    # Save per-window absolute errors (for paired bootstrap over test weeks)
    import numpy as _np
    per_win = {}
    for name, (p, t, m) in preds.items():
        pd_ = denorm(_np.nan_to_num(p, nan=0.0), norm_mean, norm_std)
        td = denorm(_np.nan_to_num(t, nan=0.0), norm_mean, norm_std)
        md = m > 0
        err = _np.abs(pd_ - td)                    # (n_win, 48, N)
        num = (err * md).sum(axis=(1, 2))          # per-window absolute error sum
        cnt = md.sum(axis=(1, 2))                  # per-window observed count
        # per-window, per-horizon error sums (for horizon breakdowns)
        num_h = (err * md).sum(axis=2)              # (n_win, 48)
        cnt_h = md.sum(axis=2)                      # (n_win, 48)
        per_win[name] = {"num": num.tolist(), "cnt": cnt.tolist(),
                         "num_h": num_h.tolist(), "cnt_h": cnt_h.tolist(),
                         "start_idx": int(starts[0])}
    _np.savez_compressed("outputs/per_window_errors.npz", **{
        k: _np.array([v["num"], v["cnt"]], dtype=_np.float64) for k, v in per_win.items()})
    _np.savez_compressed("outputs/per_window_horizon_errors.npz", **{
        k: _np.array([v["num_h"], v["cnt_h"]], dtype=_np.float64) for k, v in per_win.items()})
    # week index for each evaluation window (ISO week, for paired bootstrap)
    _wk = pd.to_datetime(ds["timestamps"][starts.astype(int) + 72], unit="s").isocalendar()
    _np.save("outputs/window_week.npy", _np.stack([_wk.year.values, _wk.week.values]).astype(_np.int32))
    print(f"Saved per-window errors for {len(per_win)} models")

    rows.sort(key=lambda r: r["mae"])
    hdr = f"{'model':<44}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'1h':>8}{'3h':>8}{'6h':>8}{'12h':>8}{'24h':>8}{'48h':>8}"
    print("\n" + "=" * len(hdr))
    print(f"{'MODEL':<44}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'1h':>8}{'3h':>8}{'6h':>8}{'12h':>8}{'24h':>8}{'48h':>8}")
    print("=" * len(hdr))
    for r in rows:
        print(f"{r['name']:<44}{r['mae']:>8.2f}{r['rmse']:>8.2f}{r['r2']:>8.3f}"
              f"{r['h1']:>8.2f}{r['h3']:>8.2f}{r['h6']:>8.2f}{r['h12']:>8.2f}"
              f"{r['h24']:>8.2f}{r['h48']:>8.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
