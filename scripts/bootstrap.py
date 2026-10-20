#!/usr/bin/env python3
"""Paired bootstrap over test weeks for the graph comparison.

For each pair of configurations we compute the per-week MAE (averaged over
seeds), then resample the ISO weeks with replacement, taking the SAME weeks for
both configurations (hence "paired"), and report the distribution of the MAE
difference.  This is the right test for a difference of the size we are
chasing: it respects the temporal correlation in the errors and the pairing
between models, which a per-seed standard deviation does not.
"""
import sys
from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

PAIRS = [
    ("stgcn_distance", "stgcn_wind_aware"),
    ("stgcn_distance", "stgcn_identity"),
    ("stgcn_wind_aware", "stgcn_identity"),
    ("graph_wavenet_distance", "graph_wavenet_wind_aware"),
    ("graph_wavenet_distance", "graph_wavenet_identity"),
    ("graph_wavenet_wind_aware", "graph_wavenet_identity"),
    ("graph_wavenet_learned_adaptive", "graph_wavenet_identity"),
    ("graph_wavenet_learned_adaptive", "graph_wavenet_wind_aware"),
    ("stgcn_identity", "lstm_univariate_distance"),
    ("stgcn_identity", "Ridge(24 lags)"),
    ("stgcn_identity", "Persistence"),
]


def per_week_mae(err_npz, week, stems):
    """Mean absolute error per ISO week, averaged over seeds, for each stem."""
    # group checkpoints by stem
    by_stem = defaultdict(list)
    for k in err_npz.files:
        stem = k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
        by_stem[stem].append(k)

    out = {}
    for stem in stems:
        if stem not in by_stem:
            continue
        wk = defaultdict(lambda: [0.0, 0.0])
        for k in by_stem[stem]:
            num, cnt = err_npz[k]
            for w, n, c in zip(week_key(week), num, cnt):
                wk[w][0] += n
                wk[w][1] += c
        out[stem] = {w: (v[0] / v[1] if v[1] > 0 else np.nan) for w, v in wk.items()}
    return out


def week_key(week):
    """(year, week) tuple for each evaluation window."""
    return list(zip(week[0].tolist(), week[1].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--out", default="outputs/bootstrap.json")
    args = ap.parse_args()

    err_npz = np.load(OUT / "per_window_errors.npz")
    week = np.load(OUT / "window_week.npy")

    stems = sorted({k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
                    for k in err_npz.files})
    pw = per_week_mae(err_npz, week, stems)

    rng = np.random.default_rng(0)
    all_weeks = sorted({w for s in pw.values() for w in s})
    results = []

    print(f"{'comparison':<62}{'dMAE':>8}{'95% CI':>22}{'p(A<B)':>9}")
    print("-" * 101)
    for a, b in PAIRS:
        if a not in pw or b not in pw:
            print(f"  (missing: {a if a not in pw else b})")
            continue
        weeks = [w for w in all_weeks if w in pw[a] and w in pw[b]]
        va = np.array([pw[a][w] for w in weeks])
        vb = np.array([pw[b][w] for w in weeks])
        d = va - vb                      # negative => A better
        idx = rng.integers(0, len(weeks), size=(args.n_boot, len(weeks)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_better = float((boot < 0).mean())
        print(f"{a + ' vs ' + b:<62}{d.mean():>8.2f}"
              f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}{p_better:>9.3f}")
        results.append({
            "a": a, "b": b, "n_weeks": len(weeks), "diff_mae": float(d.mean()),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "p_a_better": p_better, "n_boot": args.n_boot,
        })

    (OUT / "bootstrap.json").write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {OUT/'bootstrap.json'}")


if __name__ == "__main__":
    main()
