#!/usr/bin/env python3
"""Paired bootstrap over test weeks for the graph comparison.

The statistic
-------------
For a pair of predictors A and B we report the difference of their **pooled**
MAE over the test set:

    MAE_A = sum(|e_A|) / sum(n_observed)      (over all test windows)
    MAE_B = sum(|e_B|) / sum(n_observed)
    dMAE  = MAE_A - MAE_B

which is exactly the difference of the two MAEs quoted in the results tables.
The confidence interval comes from resampling the ISO weeks of the test period
with replacement -- the same weeks for both predictors, hence "paired" -- and
recomputing the pooled statistic on each resample.  Resampling weeks (rather
than windows) respects the temporal correlation in the errors.

An earlier version of this script averaged the *per-week* MAEs instead, which
is an unweighted average over weeks and therefore not the same quantity as the
pooled MAE in the tables.  On the main test set the two agree closely, but on
the ten-station leave-station-out subset they differ materially, which is how
the inconsistency was found.

Usage
-----
    python scripts/bootstrap.py --n_boot 50000
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
    ("stgcn_wind_aware_era5", "stgcn_wind_aware"),
    ("stgcn_wind_aware_era5", "stgcn_identity"),
    ("graph_wavenet_wind_aware_era5", "graph_wavenet_wind_aware"),
    ("graph_wavenet_wind_aware_era5", "graph_wavenet_identity"),
    ("stgcn_identity", "lstm_univariate_distance"),
    ("stgcn_identity", "Ridge(24 lags)"),
    ("stgcn_identity", "Persistence"),
    ("stgcn_identity", "Seasonal naive (24h)"),
    ("stgcn_identity", "Hour-of-week average"),
    ("stgcn_wind_aware", "Seasonal naive (24h)"),
    ("graph_wavenet_wind_aware", "Seasonal naive (24h)"),
    ("lstm_multivariate_distance", "Seasonal naive (24h)"),
    ("stgcn_identity_netmean", "stgcn_identity"),
    ("stgcn_identity_netmean", "stgcn_wind_aware"),
    ("stgcn_identity_netmean", "Seasonal naive (24h)"),
    ("stgcn_identity_netmean", "graph_wavenet_identity"),
]


def per_week_sums(err_npz, week, stems):
    """Per week, the summed absolute error and the observation count.

    Returns {stem: {week: (num, cnt)}}.  The sums (rather than a per-week
    MAE) are kept so that the pooled statistic can be recomputed on any
    resample of weeks.
    """
    out = {}
    for stem in stems:
        keys = [k for k in err_npz.files
                if (k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k) == stem]
        if not keys:
            continue
        wk = defaultdict(lambda: [0.0, 0.0])
        for k in keys:
            num, cnt = err_npz[k]
            for w, n, c in zip(week, num, cnt):
                wk[w][0] += n
                wk[w][1] += c
        out[stem] = dict(wk)
    return out


def pooled(sums, weeks, idx=None):
    """Pooled MAE over the given weeks (optionally a resampled index array)."""
    num = 0.0
    cnt = 0.0
    if idx is None:
        for w in weeks:
            n, c = sums[w]
            num += n
            cnt += c
    else:
        for j in idx:
            n, c = sums[weeks[j]]
            num += n
            cnt += c
    return num / cnt if cnt > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--out", default="outputs/bootstrap.json")
    args = ap.parse_args()

    err_npz = np.load(OUT / "per_window_errors.npz")
    week_arr = np.load(OUT / "window_week.npy")
    week = list(zip(week_arr[0].tolist(), week_arr[1].tolist()))

    stems = sorted({k.rsplit("_s", 1)[0] if ("_s" in k and k.rsplit("_s", 1)[-1].isdigit()) else k
                    for k in err_npz.files})
    sums = per_week_sums(err_npz, week, stems)

    rng = np.random.default_rng(0)
    all_weeks = sorted({w for s in sums.values() for w in s})
    results = []

    print(f"{'comparison':<62}{'dMAE':>8}{'95% CI':>22}{'p(A<B)':>9}")
    print("-" * 101)
    for a, b in PAIRS:
        if a not in sums or b not in sums:
            print(f"  (missing: {a if a not in sums else b})")
            continue
        weeks = [w for w in all_weeks if w in sums[a] and w in sums[b]]
        nw = len(weeks)
        # the point estimate is the pooled difference, i.e. exactly the
        # difference of the two MAEs in the results tables
        point = pooled(sums[a], weeks) - pooled(sums[b], weeks)
        idx = rng.integers(0, nw, size=(args.n_boot, nw))
        boot = np.empty(args.n_boot)
        for i in range(args.n_boot):
            boot[i] = pooled(sums[a], weeks, idx[i]) - pooled(sums[b], weeks, idx[i])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_better = float((boot < 0).mean())
        print(f"{a + ' vs ' + b:<62}{point:>8.2f}"
              f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}{p_better:>9.3f}")
        results.append({
            "a": a, "b": b, "n_weeks": nw, "diff_mae": float(point),
            "mae_a": float(pooled(sums[a], weeks)),
            "mae_b": float(pooled(sums[b], weeks)),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "p_a_better": p_better, "n_boot": args.n_boot,
        })

    (OUT / "bootstrap.json").write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {OUT/'bootstrap.json'}")


if __name__ == "__main__":
    main()
