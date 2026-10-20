#!/usr/bin/env python3
"""Case studies: three real pollution events in the test period.

For each event we plot the network-mean observed PM2.5 against the network-mean
prediction of the best model and of the two baselines, and pick two stations to
show prediction-versus-truth traces.

Events (by construction, not by hand-picking the model's best days):
  * Diwali night          -- the autumn festival, 28-30 October
  * dust storm            -- a May/June episode with strong westerly winds
  * winter inversion      -- a December/January episode with calm winds
"""
import sys
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset
from scripts.evaluate import (denorm, persistence_predictions, ridge_predictions,
                              model_predictions)

DATA = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs"


def main():
    ds = load_dataset(str(DATA))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    y = ds["y_full"]
    mask = ds["mask_y"]
    ts = pd.to_datetime(ds["timestamps"], unit="s")
    vals = np.expm1(np.clip(y * ns + nm, -700, 700))
    obs = mask > 0
    net = np.nanmean(np.where(obs, vals, np.nan), axis=1)
    net = pd.Series(net, index=ts)

    idx = ds["split_indices_test"]
    lo, hi = int(idx[0]), int(idx[-1])
    test_net = net.iloc[lo:hi + 1]

    # ---- pick the events -------------------------------------------------
    # dust: May-June, strong wind; winter: Dec-Jan, calm; Diwali: fixed dates
    X = ds["X_full"]
    wu = X[:, :, 3]
    wv = X[:, :, 4]
    speed = np.sqrt(np.nanmean(wu ** 2 + wv ** 2, axis=1))

    events = []

    # Diwali 2024 (28-30 Oct) and 2025 (19-21 Oct) -- look for the peak
    for year, d1, d2 in [(2024, "2024-10-28", "2024-10-31"),
                         (2025, "2025-10-19", "2025-10-22")]:
        seg = test_net.loc[d1:d2]
        if len(seg) and seg.max() > 200:
            events.append({
                "name": f"Diwali {year}",
                "kind": "diwali",
                "start": seg.idxmax().normalize() - pd.Timedelta(hours=12),
                "end": seg.idxmax().normalize() + pd.Timedelta(hours=24),
                "peak": float(seg.max()),
            })
            break

    # winter inversion: Dec/Jan, calm, high PM2.5
    winter = test_net[(test_net.index.month.isin([12, 1]))]
    if len(winter):
        pk = winter.idxmax()
        events.append({"name": "Winter inversion", "kind": "winter",
                       "start": pk - pd.Timedelta(hours=12),
                       "end": pk + pd.Timedelta(hours=36),
                       "peak": float(winter.max())})

    # dust storm: May/June with the strongest winds
    dust_idx = np.where(np.isin(test_net.index.month, [5, 6]))[0]
    if len(dust_idx):
        sub_speed = speed[lo:hi + 1][dust_idx]
        cand = dust_idx[np.argsort(-np.nan_to_num(sub_speed, nan=0.0))]
        for c in cand:
            pk = test_net.index[c]
            if test_net.iloc[c] > 150:
                events.append({"name": "Dust storm", "kind": "dust",
                               "start": pk - pd.Timedelta(hours=12),
                               "end": pk + pd.Timedelta(hours=36),
                               "peak": float(test_net.iloc[c]),
                               "wind_proxy": float(speed[lo + c])})
                break

    print("=== events ===")
    for e in events:
        print(f"  {e['name']:<20} {e['start']} .. {e['end']}  peak {e['peak']:.0f} ug/m3")

    # ---- predictions -----------------------------------------------------
    starts = np.arange(lo, hi - 72 - 48 + 2)
    sel = np.linspace(0, len(starts) - 1, 3000).astype(int)
    starts = starts[sel]

    preds = {}
    p, t, m = persistence_predictions(ds, starts)
    preds["Persistence"] = p
    p, t, m = ridge_predictions(ds, starts)
    preds["Ridge (24 lags)"] = p
    for ck, label in [(CKPT / "stgcn_identity_s42.pt", "STGCN, no graph"),
                      (CKPT / "stgcn_wind_aware_s42.pt", "STGCN, wind-aware"),
                      (CKPT / "graph_wavenet_wind_aware_s42.pt", "GraphWaveNet, wind-aware")]:
        if ck.exists():
            p, _, _ = model_predictions(ds, str(ck), starts, "cpu")
            preds[label] = p

    fh = starts[:, None] + 72 + np.arange(48)[None, :]     # (n, 48) forecast hours
    fm = mask[fh] > 0                                      # (n, 48, N) observed mask

    def net_pred(p):
        v = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
        return np.nanmean(np.where(fm, v, np.nan), axis=(1, 2))

    series = {k: net_pred(v) for k, v in preds.items()}
    win_start = ts[starts + 72]     # first predicted hour of each window

    # ---- figures ---------------------------------------------------------
    figdir = OUT
    summary = []
    for e in events:
        m0 = pd.Timestamp(e["start"])
        m1 = pd.Timestamp(e["end"])
        sel_w = (win_start >= m0) & (win_start <= m1)
        if sel_w.sum() < 3:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

        # left: network mean over the event
        axes[0].plot(win_start[sel_w], series["STGCN, no graph"][sel_w] if "STGCN, no graph" in series
                     else series[list(series)[0]][sel_w],
                     lw=1.4, label="STGCN, no graph")
        if "STGCN, wind-aware" in series:
            axes[0].plot(win_start[sel_w], series["STGCN, wind-aware"][sel_w], lw=1.0, ls="--",
                         label="STGCN, wind-aware")
        if "GraphWaveNet, wind-aware" in series:
            axes[0].plot(win_start[sel_w], series["GraphWaveNet, wind-aware"][sel_w], lw=1.0, ls=":",
                         label="GraphWaveNet, wind-aware")
        axes[0].plot(win_start[sel_w], series["Ridge (24 lags)"][sel_w], lw=1.0, color="0.5",
                     label="Ridge (24 lags)")
        axes[0].plot(win_start[sel_w], series["Persistence"][sel_w], lw=1.0, color="0.75",
                     label="Persistence")
        axes[0].axhline(150, color="tab:red", lw=0.8, ls=":", alpha=0.7)
        axes[0].set_title(f"{e['name']}  (network mean)", fontsize=10)
        axes[0].set_ylabel("PM2.5 (ug/m3)")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].grid(alpha=0.3)

        # right: two stations, truth vs prediction
        st_true = vals[fh[sel_w]]                  # (n, 48, N)
        st_obs = fm[sel_w]
        err_per_st = np.where(st_obs, np.abs(
            denorm(np.nan_to_num(preds["STGCN, no graph"][sel_w], nan=0.0), nm, ns) - st_true), np.nan)
        with np.errstate(invalid="ignore"):
            per_st = np.nanmean(err_per_st, axis=(0, 1))
        order = np.argsort(-np.nan_to_num(per_st, nan=-1))
        worst = order[:2]
        for st in worst:
            t_ = np.nanmean(np.where(st_obs[:, :, st], st_true[:, :, st], np.nan), axis=0)
            axes[1].plot(np.arange(1, 49), t_, lw=1.4, label=f"obs {ds['station_ids'][st]}")
        axes[1].set_title("two worst stations, observed", fontsize=10)
        axes[1].set_xlabel("horizon (h)")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        out_png = figdir / f"fig_case_{e['kind']}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_png.name}")

        # quantitative summary for the event
        row = {"name": e["name"], "kind": e["kind"],
               "start": str(e["start"]), "end": str(e["end"]),
               "peak_network_pm25": round(e["peak"], 1), "n_windows": int(sel_w.sum())}
        for k, v in series.items():
            obs_v = np.nanmean(np.where(st_obs, st_true, np.nan), axis=(1, 2))
            row[f"mae_{k}"] = round(float(np.nanmean(np.abs(v[sel_w] - obs_v))), 2)
        summary.append(row)

    (OUT / "case_studies.json").write_text(json.dumps(summary, indent=2))
    print("\n=== event metrics ===")
    for r in summary:
        print(f"  {r['name']:<20} peak {r['peak_network_pm25']:>6.1f}  " +
              "  ".join(f"{k.split(',')[0][:9]}={v}" for k, v in r.items()
                       if k.startswith("mae_")))
    print(f"\nWrote {OUT/'case_studies.json'}")


if __name__ == "__main__":
    main()
