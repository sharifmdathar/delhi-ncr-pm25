#!/usr/bin/env python3
"""Paired bootstrap over test weeks for the leave-station-out comparison.

Same construction as scripts/bootstrap.py, but restricted to the held-out
stations and using the per-window errors written by scripts/eval_lso.py.
"""
import sys
from pathlib import Path
import json
import argparse
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

PAIRS = {
    "target": [
        ("stgcn_wind_aware_s42_lso10_target", "stgcn_identity_s42_lso10_target"),
        ("stgcn_distance_s42_lso10_target", "stgcn_identity_s42_lso10_target"),
        ("graph_wavenet_wind_aware_s42_lso10_target", "graph_wavenet_identity_s42_lso10_target"),
        ("graph_wavenet_distance_s42_lso10_target", "graph_wavenet_identity_s42_lso10_target"),
        ("stgcn_identity_s42_lso10_target", "Ridge (fit on training stations only)"),
        ("stgcn_identity_s42_lso10_target", "Seasonal naive (24h)"),
    ],
    "both": [
        ("stgcn_distance_s42_lso10_both", "stgcn_identity_s42_lso10_both"),
        ("stgcn_wind_aware_s42_lso10_both", "stgcn_identity_s42_lso10_both"),
        ("graph_wavenet_distance_s42_lso10_both", "graph_wavenet_identity_s42_lso10_both"),
        ("graph_wavenet_wind_aware_s42_lso10_both", "graph_wavenet_identity_s42_lso10_both"),
        # the fair comparisons in this mode: every one of these predicts the
        # held-out station from the OTHER stations only
        ("stgcn_distance_s42_lso10_both", "IDW neighbour (seasonal)"),
        ("stgcn_distance_s42_lso10_both", "IDW neighbour (persistence)"),
        ("stgcn_distance_s42_lso10_both", "Network mean (persistence)"),
        ("graph_wavenet_distance_s42_lso10_both", "IDW neighbour (seasonal)"),
        ("IDW neighbour (seasonal)", "Network mean (persistence)"),
    ],
    "target": [
        ("stgcn_distance_s42_lso10_target", "IDW neighbour (seasonal)"),
        ("stgcn_identity_s42_lso10_target", "IDW neighbour (seasonal)"),
        ("stgcn_identity_s42_lso10_target", "Network mean (persistence)"),
    ],
}


def per_week_mae(err, week, names):
    """Mean absolute error per ISO week for each named predictor."""
    out = {}
    for name in names:
        if name not in err.files:
            continue
        num, cnt = err[name]
        acc = defaultdict(lambda: [0.0, 0.0])
        for w, n, c in zip(week, num, cnt):
            acc[w][0] += n
            acc[w][1] += c
        out[name] = {w: (v[0] / v[1] if v[1] > 0 else np.nan) for w, v in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--out", default="outputs/lso_bootstrap.json")
    args = ap.parse_args()

    week = np.load(OUT / "lso_window_week.npy")
    week_keys = list(zip(week[0].tolist(), week[1].tolist()))
    rng = np.random.default_rng(0)

    results = []
    for mode, pairs in PAIRS.items():
        path = OUT / f"lso_per_window_{mode}.npz"
        if not path.exists():
            continue
        err = np.load(path)
        names = set()
        for a, b in pairs:
            names.add(a)
            names.add(b)
        pw = per_week_mae(err, week_keys, names)
        all_weeks = sorted({w for v in pw.values() for w in v})
        print(f"\n=== leave-station-out, mode={mode} ===")
        print(f"{'comparison':<70}{'dMAE':>8}{'95% CI':>22}{'p(A<B)':>9}")
        print("-" * 109)
        for a, b in pairs:
            if a not in pw or b not in pw:
                print(f"  (missing: {a if a not in pw else b})")
                continue
            weeks = [w for w in all_weeks if w in pw[a] and w in pw[b]]
            va = np.array([pw[a][w] for w in weeks])
            vb = np.array([pw[b][w] for w in weeks])
            d = va - vb
            idx = rng.integers(0, len(weeks), size=(args.n_boot, len(weeks)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_better = float((boot < 0).mean())
            print(f"{a + ' vs ' + b:<70}{d.mean():>8.2f}"
                  f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}"
                  f"{p_better:>9.3f}")
            results.append({
                "mode": mode, "a": a, "b": b, "n_weeks": len(weeks),
                "diff_mae": float(d.mean()), "ci95_low": float(lo),
                "ci95_high": float(hi), "p_a_better": p_better, "n_boot": args.n_boot,
            })

    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
