#!/usr/bin/env python3
"""Post-hoc analysis: episode performance, per-station errors, and figures.

Writes:
  outputs/analysis_episodes.json   metrics during severe episodes vs the rest
  outputs/analysis_spatial.json    per-station MAE / bias
  outputs/fig_error_by_horizon.png MAE vs horizon for every model
  outputs/fig_spatial_error.png    per-station error map (Delhi)
  outputs/fig_station_traces.png   predicted vs observed for a few stations
  outputs/fig_adjacency.png        the three fixed graph structures
"""

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import load_dataset
from src.data.graphs import build_graph
from src.eval.episodes import find_episodes
from src.train.metrics import compute_metrics
from scripts.evaluate import (denorm, persistence_predictions, ridge_predictions,
                              model_predictions, _clean_lags)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs"

EPISODE_THRESHOLD = 150.0
EPISODE_DURATION = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_windows", type=int, default=1200)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(str(DATA_PATH))
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    n_st = ds["X_full"].shape[1]
    station_ids = ds["station_ids"]

    idx = ds["split_indices_test"]
    starts = np.arange(int(idx[0]), int(idx[-1]) - 72 - 48 + 2)
    if len(starts) > args.max_windows:
        sel = np.linspace(0, len(starts) - 1, args.max_windows).astype(int)
        starts = starts[sel]
    print(f"test windows: {len(starts)}")

    # ---- episode mask over the test period -------------------------------
    y_test = ds["y_full"][int(idx[0]):int(idx[-1]) + 1]
    m_test = ds["mask_y"][int(idx[0]):int(idx[-1]) + 1]
    y_test_denorm = denorm(np.nan_to_num(y_test, nan=0.0), nm, ns)
    episode_flags = np.zeros(y_test.shape, dtype=bool)
    n_eps = 0
    for j in range(n_st):
        obs = m_test[:, j] > 0
        series = y_test_denorm[:, j]
        eps = find_episodes(series, EPISODE_THRESHOLD, EPISODE_DURATION, obs.astype(float))
        n_eps += len(eps)
        for s0, e0 in eps:
            episode_flags[s0:e0, j] = True
    print(f"episodes found (station-wise): {n_eps}; "
          f"episode hours = {episode_flags.sum()} / {episode_flags.size} "
          f"({episode_flags.sum()/episode_flags.size:.2%})")

    # ---- evaluate every model ------------------------------------------
    preds = {}
    p, t, m = persistence_predictions(ds, starts)
    preds["Persistence"] = (p, t, m)
    p, t, m = ridge_predictions(ds, starts)
    preds["Ridge(24 lags)"] = (p, t, m)
    for ck in sorted(CKPT_DIR.glob("*.pt")):
        name = ck.stem
        if "2019_2024" in name or "2024_2025" in name:
            continue
        if args.models and not any(mm in name for mm in args.models):
            continue
        try:
            preds[name] = model_predictions(ds, str(ck), starts, "cpu")
        except Exception as e:
            print(f"  skip {name}: {e}")

    # ---- episode vs non-episode metrics ---------------------------------
    # Build an episode mask aligned with the prediction arrays:
    # (n_windows, 48, n_stations) -> True where the observed target is
    # inside a sustained severe episode.
    t0 = int(idx[0])
    ep_mask = np.zeros((len(starts), 48, n_st), dtype=bool)
    for i, s in enumerate(starts):
        m0 = int(s) + 72 - t0
        ep_mask[i] = episode_flags[m0:m0 + 48]

    ep_rows = []
    for name, (p, t, m) in preds.items():
        pd_ = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
        td = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
        md = m > 0
        overall = compute_metrics(pd_, td, md)
        ep_sel = ep_mask & md
        non_sel = (~ep_mask) & md
        ep = compute_metrics(pd_[ep_sel], td[ep_sel], None) if ep_sel.sum() else {}
        ne = compute_metrics(pd_[non_sel], td[non_sel], None) if non_sel.sum() else {}
        ep_rows.append({
            "name": name, "mae": overall["mae"], "rmse": overall["rmse"],
            "r2": overall["r2"],
            "episode_mae": ep.get("mae"), "episode_rmse": ep.get("rmse"),
            "non_episode_mae": ne.get("mae"), "non_episode_rmse": ne.get("rmse"),
        })
    ep_rows.sort(key=lambda r: r["mae"])
    (OUT / "analysis_episodes.json").write_text(json.dumps({
        "threshold_ugm3": EPISODE_THRESHOLD, "duration_hours": EPISODE_DURATION,
        "n_episodes": n_eps,
        "episode_hour_fraction": float(episode_flags.sum() / episode_flags.size),
        "rows": ep_rows}, indent=2))

    print("\n=== EPISODE vs NON-EPISODE ===")
    print(f"{'model':<40}{'MAE':>8}{'ep MAE':>9}{'non-ep MAE':>11}")
    for r in ep_rows:
        e = f"{r['episode_mae']:.2f}" if r["episode_mae"] is not None else "-"
        n = f"{r['non_episode_mae']:.2f}" if r["non_episode_mae"] is not None else "-"
        print(f"{r['name']:<40}{r['mae']:>8.2f}{e:>9}{n:>11}")

    # ---- per-station errors ---------------------------------------------
    spatial = {}
    for name, (p, t, m) in preds.items():
        pd_ = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
        td = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
        md = m > 0
        per = []
        for j in range(n_st):
            sel = md[:, :, j]
            if sel.sum() == 0:
                per.append({"station": str(station_ids[j]), "mae": None, "bias": None})
                continue
            e = pd_[:, :, j][sel] - td[:, :, j][sel]
            per.append({"station": str(station_ids[j]),
                        "mae": float(np.abs(e).mean()), "bias": float(e.mean())})
        spatial[name] = per
    (OUT / "analysis_spatial.json").write_text(json.dumps(spatial, indent=2))

    # ---- figure: MAE by horizon -----------------------------------------
    from src.train.metrics import compute_horizon_metrics
    # Plot the seed-averaged configuration for each model family, plus baselines
    from collections import defaultdict
    groups = defaultdict(list)
    for name, (p, t, m) in preds.items():
        stem = name.rsplit("_s", 1)[0] if (name.rsplit("_s", 1)[-1].isdigit()
                                           and "_s" in name) else name
        groups[stem].append((p, t, m))

    fig, ax = plt.subplots(figsize=(9, 5))
    hs = [1, 3, 6, 12, 24, 48]
    styles = {
        "Persistence": ("k", "o", 2.0), "Ridge(24 lags)": ("tab:orange", "s", 1.8),
        "stgcn_identity": ("tab:green", "o", 1.6),
        "stgcn_wind_aware": ("tab:cyan", "^", 1.4),
        "stgcn_distance": ("tab:blue", "v", 1.4),
        "stgcn_correlation": ("tab:purple", "<", 1.4),
        "graph_wavenet_identity": ("tab:green", "o", 1.0),
        "graph_wavenet_wind_aware": ("tab:cyan", "^", 0.9),
        "graph_wavenet_distance": ("tab:blue", "v", 0.9),
        "graph_wavenet_correlation": ("tab:purple", "<", 0.9),
        "lstm_univariate_distance": ("tab:red", "D", 1.5),
        "lstm_multivariate_distance": ("tab:brown", "D", 1.2),
        "transformer_distance": ("tab:olive", "D", 1.2),
    }
    for stem, items in groups.items():
        col, mk, lw = styles.get(stem, (None, "o", 1.0))
        if col is None:
            continue
        curves = []
        for p, t, m in items:
            pd_ = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
            td = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
            ph = compute_horizon_metrics(pd_, td, m)
            curves.append([ph[f"horizon_{h}h"]["mae"] for h in hs])
        arr = np.array(curves)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        label = stem.replace("_distance", "").replace("_identity", " (no graph)")
        ax.plot(hs, mean, marker=mk, color=col, lw=lw, label=label)
        if len(arr) > 1:
            ax.fill_between(hs, mean - sd, mean + sd, color=col, alpha=0.15)
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Test MAE by forecast horizon (mean ± std over 3 seeds)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_by_horizon.png", dpi=140)
    plt.close(fig)

    # ---- figure: spatial error map ---------------------------------------
    best = min(preds.items(), key=lambda kv: compute_metrics(
        denorm(np.nan_to_num(kv[1][0], nan=0.0), nm, ns),
        denorm(np.nan_to_num(kv[1][1], nan=0.0), nm, ns), kv[1][2] > 0)["mae"])
    name, (p, t, m) = best
    pd_ = denorm(np.nan_to_num(p, nan=0.0), nm, ns)
    td = denorm(np.nan_to_num(t, nan=0.0), nm, ns)
    md = m > 0
    coords = ds["coords"]
    maes = []
    for j in range(n_st):
        sel = md[:, :, j]
        maes.append(np.abs(pd_[:, :, j][sel] - td[:, :, j][sel]).mean()
                    if sel.sum() else np.nan)
    maes = np.array(maes)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:, 1], coords[:, 0], c=maes, s=90, cmap="viridis",
                    edgecolor="k", linewidth=0.4)
    for j in range(n_st):
        ax.annotate(str(station_ids[j]), (coords[j, 1], coords[j, 0]),
                    fontsize=5.5, ha="center", va="bottom", color="w")
    fig.colorbar(sc, ax=ax, label="MAE (µg/m³)")
    ax.set_title(f"Per-station test MAE — {name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spatial_error.png", dpi=140)
    plt.close(fig)

    # ---- figure: station traces ------------------------------------------
    show = [0, 6, 12, 18]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 9), sharex=True)
    for ax, j in zip(axes, show):
        truth = td[:, :, j].mean(axis=1)
        pred = pd_[:, :, j].mean(axis=1)
        h = np.arange(len(truth))
        ax.plot(h, truth, lw=1.0, label="observed")
        ax.plot(h, pred, lw=1.0, alpha=0.85, label="predicted")
        ax.set_ylabel(f"{station_ids[j]}")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Test window (network-mean hourly PM2.5)")
    fig.suptitle(f"Predicted vs observed — {name}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_station_traces.png", dpi=140)
    plt.close(fig)

    # ---- figure: adjacency structures ------------------------------------
    tr = ds["split_indices_train"]
    trs = slice(int(tr[0]), int(tr[-1]) + 1)
    y_train = denorm(np.nan_to_num(ds["y_full"][trs], nan=0.0), nm, ns)
    with np.errstate(invalid="ignore"):
        wu = np.nanmean(ds["X_full"][trs][:, :, 3], axis=0)
        wv = np.nanmean(ds["X_full"][trs][:, :, 4], axis=0)
    wu = np.where(np.isnan(wu), np.nanmean(wu), wu)
    wv = np.where(np.isnan(wv), np.nanmean(wv), wv)

    graphs = {
        "distance": build_graph("distance", ds["coords"]),
        "correlation": build_graph("correlation", ds["coords"], pm25_train=y_train),
        "wind_aware": build_graph("wind_aware", ds["coords"], wind_u=wu, wind_v=wv),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (gname, A) in zip(axes, graphs.items()):
        im = ax.imshow(A, cmap="magma")
        ax.set_title(gname)
        ax.set_xlabel("destination station")
        ax.set_ylabel("source station")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fixed graph structures (train split only)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_adjacency.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote analysis to {OUT}")


if __name__ == "__main__":
    main()
