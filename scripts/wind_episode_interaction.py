#!/usr/bin/env python3
"""Why does the low-wind bin have twice the error?

Hypothesis: severe PM2.5 episodes in Delhi occur under calm, stable
conditions (no dispersion), so the low-wind bin of the test set carries most
of the episode hours, and the episode hours are where all the error is.

This script measures the episode-hour fraction per wind bin, and the graph
comparison restricted to episode hours vs non-episode hours.
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

DATA = PROJECT_ROOT / "data" / "processed" / "delhi_ncr_pm25_2019_2025.npz"


def main():
    ds = load_dataset(str(DATA))
    idx = ds["split_indices_test"]
    lo, hi = int(idx[0]), int(idx[-1])
    starts = np.arange(lo, hi - 72 - 48 + 2)
    sel = np.linspace(0, len(starts) - 1, 2000).astype(int)
    starts = starts[sel]

    X = ds["X_full"]
    wu = X[starts[:, None] + np.arange(72)[None, :], :, 3]
    wv = X[starts[:, None] + np.arange(72)[None, :], :, 4]
    wind_speed = np.sqrt(np.nanmean(wu ** 2 + wv ** 2, axis=(1, 2)))
    edges = np.percentile(wind_speed, [0, 33, 66, 100])
    bin_id = np.digitize(wind_speed, edges[1:-1])

    # --- episode flags on the raw hourly series of the test block ---
    # Uses the canonical definition from src/eval/episodes.py, applied
    # station-wise exactly as scripts/analyze.py does.
    y = ds["y_full"]
    mask = ds["mask_y"]
    nm, ns = float(ds["norm_mean"]), float(ds["norm_std"])
    vals = np.expm1(np.clip(y * ns + nm, -700, 700))          # ug/m3
    obs = mask > 0
    THRESH, MIN_H = 150.0, 6

    from src.eval.episodes import find_episodes
    ep = np.zeros(len(vals), dtype=bool)
    for j in range(y.shape[1]):
        o = obs[:, j]
        for s0, e0 in find_episodes(vals[:, j], THRESH, MIN_H, o.astype(float)):
            ep[s0:e0] = True

    ep_hours = ep[lo:hi]
    ep_window = np.array([ep_hours[s - lo:s - lo + 48].any() for s in starts])

    print("=== episode-hour fraction per wind bin (test block) ===")
    print(f"{'bin':<20}{'n_win':>7}{'frac of windows with an episode hour':>42}")
    for b in range(len(edges) - 1):
        m = bin_id == b
        if m.sum() == 0:
            continue
        print(f"[{edges[b]:.2f},{edges[b+1]:.2f}){'':<6}{m.sum():>7}"
              f"{ep_window[m].mean():>42.1%}")

    print("\n=== overall ===")
    print(f"  test hours with a severe episode: {ep_hours.mean():.1%}")
    print(f"  windows touching an episode     : {ep_window.mean():.1%}")

    # --- per-bin graph comparison, split by episode vs not ---
    err = np.load(PROJECT_ROOT / "outputs" / "per_window_errors.npz")
    by = defaultdict(list)
    for k in err.files:
        stem = k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
        by[stem].append(k)
    stems = [s for s in by if s.startswith(("stgcn_", "graph_wavenet_"))]

    out = {"wind_speed_edges": edges.tolist(),
           "episode_hour_fraction": float(ep_hours.mean()),
           "bins": []}
    print(f"\n{'subset':<26}" + "".join(f"{s.replace('graph_wavenet','gwn').replace('stgcn','stg'):>22}"
                                       for s in stems))
    for label, wsel in [("all windows", np.ones(len(starts), bool)),
                        ("episode windows", ep_window),
                        ("non-episode windows", ~ep_window)]:
        cells = []
        row = {"subset": label}
        for s in stems:
            num = np.zeros(len(starts)); cnt = np.zeros(len(starts))
            for k in by[s]:
                n, c = err[k]
                num += n; cnt += c
            m = wsel & (cnt > 0)
            mae = float(num[m].sum() / max(cnt[m].sum(), 1))
            row[s] = mae
            cells.append(f"{mae:>22.2f}")
        print(f"{label:<26}" + "".join(cells))
        out.setdefault("by_subset", {})[label] = row

    _wb_path = PROJECT_ROOT / "outputs" / "wind_breakdown.json"
    _existing = json.loads(_wb_path.read_text()) if _wb_path.exists() else {}
    # `out` must win: it carries the corrected by_subset
    _wb_path.write_text(json.dumps({**_existing, **out}, indent=2))
    print("\nupdated outputs/wind_breakdown.json")


if __name__ == "__main__":
    main()
