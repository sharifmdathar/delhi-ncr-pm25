#!/usr/bin/env python3
"""Paired bootstrap over test weeks for the leave-station-out comparison.

Same statistic as scripts/bootstrap.py: the difference of the **pooled** MAEs,
with the interval from resampling the ISO weeks.  Two things were wrong in an
earlier version and both mattered:

  * it averaged the per-week MAEs, which is not the same quantity as the
    pooled MAE in the tables;
  * it compared one *seed* against the baseline while the tables report a
    *configuration* (the seed average).  The two differ enough to be
    confusing: the best seed of stgcn_distance scores 53.45 and the seed the
    bootstrap used scores 55.20, against IDW-seasonal's 41.24 -- gaps of 12.2
    and 14.0 for the same comparison.

Model configurations are therefore grouped by seed and averaged before the
comparison, so the number here is the difference of the two numbers in the
tables.
"""
import sys
from pathlib import Path
import json
import argparse
import re
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

# config key -> (human label, list of (name, checkpoint-name-template))
PAIRS = {
    "target": [
        ("stgcn_wind_aware", "stgcn_identity"),
        ("stgcn_distance", "stgcn_identity"),
        ("graph_wavenet_wind_aware", "graph_wavenet_identity"),
        ("graph_wavenet_distance", "graph_wavenet_identity"),
        ("stgcn_identity", "ridge"),
        ("stgcn_identity", "seasonal_naive"),
        ("stgcn_distance", "idw_seasonal"),
        ("stgcn_identity", "idw_seasonal"),
        ("stgcn_identity", "network_mean"),
    ],
    "both": [
        ("stgcn_distance", "stgcn_identity"),
        ("stgcn_wind_aware", "stgcn_identity"),
        ("graph_wavenet_distance", "graph_wavenet_identity"),
        ("graph_wavenet_wind_aware", "graph_wavenet_identity"),
        # the fair comparisons in this mode: every one of these predicts the
        # held-out station from the OTHER stations only
        ("stgcn_distance", "idw_seasonal"),
        ("stgcn_distance", "idw_persistence"),
        ("stgcn_distance", "network_mean"),
        ("graph_wavenet_distance", "idw_seasonal"),
        ("idw_seasonal", "network_mean"),
    ],
}

# the node-dropout variant of the "both" mode
PAIRS_ND = {
    "both": [
        ("stgcn_distance", "stgcn_identity"),
        ("stgcn_wind_aware", "stgcn_identity"),
        ("stgcn_distance", "idw_seasonal"),
        ("stgcn_distance", "network_mean"),
        ("graph_wavenet_distance", "idw_seasonal"),
    ],
}

# baseline names as they appear in the per-window error file
BASELINES = {
    "ridge": "Ridge (fit on training stations only)",
    "seasonal_naive": "Seasonal naive (24h)",
    "idw_seasonal": "IDW neighbour (seasonal)",
    "idw_persistence": "IDW neighbour (persistence)",
    "network_mean": "Network mean (persistence)",
}


def per_week_sums(err, week, stems):
    """Per week, the summed absolute error and observation count, averaged
    over the seeds of each configuration.

    ``stems`` maps a configuration key to the list of checkpoint names that
    belong to it.
    """
    out = {}
    for stem, keys in stems.items():
        keys = [k for k in keys if k in err.files]
        if not keys:
            continue
        wk = defaultdict(lambda: [0.0, 0.0])
        for k in keys:
            num, cnt = err[k]
            for w, n, c in zip(week, num, cnt):
                wk[w][0] += n / len(keys)
                wk[w][1] += c / len(keys)
        out[stem] = dict(wk)
    return out


def pooled(sums, weeks, idx=None):
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
    ap.add_argument("--out", default="outputs/lso_bootstrap.json")
    args = ap.parse_args()

    week_arr = np.load(OUT / "lso_window_week.npy")
    week = list(zip(week_arr[0].tolist(), week_arr[1].tolist()))
    rng = np.random.default_rng(0)

    results = []
    for mode, pairs in PAIRS.items():
        path = OUT / f"lso_per_window_{mode}.npz"
        if not path.exists():
            continue
        err = np.load(path)
        names = list(err.files)
        # group the model checkpoints by configuration (drop the seed)
        groups = defaultdict(list)
        for n in names:
            if "_lso" in n:
                groups[re.sub(r"_s\d+_lso", "_lso", n)].append(n)
            else:
                groups[n].append(n)
        sums = per_week_sums(err, week, groups)
        all_weeks = sorted({w for v in sums.values() for w in v})

        def key_for(cfg, mode):
            if cfg in BASELINES:
                return BASELINES[cfg]
            return f"{cfg}_lso10_{mode}"

        print(f"\n=== leave-station-out, mode={mode} ===")
        print(f"{'comparison':<66}{'dMAE':>8}{'95% CI':>22}{'p(A<B)':>9}")
        print("-" * 105)
        for a_cfg, b_cfg in pairs:
            a, b = key_for(a_cfg, mode), key_for(b_cfg, mode)
            if a not in sums or b not in sums:
                print(f"  (missing: {a if a not in sums else b})")
                continue
            weeks = [w for w in all_weeks if w in sums[a] and w in sums[b]]
            nw = len(weeks)
            point = pooled(sums[a], weeks) - pooled(sums[b], weeks)
            idx = rng.integers(0, nw, size=(args.n_boot, nw))
            boot = np.empty(args.n_boot)
            for i in range(args.n_boot):
                boot[i] = pooled(sums[a], weeks, idx[i]) - pooled(sums[b], weeks, idx[i])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_better = float((boot < 0).mean())
            print(f"{a_cfg + ' vs ' + b_cfg:<66}{point:>8.2f}"
                  f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}"
                  f"{p_better:>9.3f}")
            results.append({
                "mode": mode, "a": a_cfg, "b": b_cfg, "n_weeks": nw,
                "diff_mae": float(point),
                "mae_a": float(pooled(sums[a], weeks)),
                "mae_b": float(pooled(sums[b], weeks)),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "p_a_better": p_better, "n_boot": args.n_boot,
            })

    # ---- the node-dropout variant ---------------------------------------
    for mode, pairs in PAIRS_ND.items():
        path = OUT / f"lso_per_window_{mode}_nd.npz"
        if not path.exists():
            continue
        err = np.load(path)
        groups = defaultdict(list)
        for n in err.files:
            if "_lso" in n:
                groups[re.sub(r"_s\d+_lso", "_lso", n)].append(n)
            else:
                groups[n].append(n)
        sums = per_week_sums(err, week, groups)
        all_weeks = sorted({w for v in sums.values() for w in v})

        def key_for(cfg, mode):
            if cfg in BASELINES:
                return BASELINES[cfg]
            return f"{cfg}_lso10_{mode}_nd30"

        print(f"\n=== leave-station-out, mode={mode}, NODE DROPOUT ===")
        print(f"{'comparison':<66}{'dMAE':>8}{'95% CI':>22}{'p(A<B)':>9}")
        print("-" * 105)
        for a_cfg, b_cfg in pairs:
            a, b = key_for(a_cfg, mode), key_for(b_cfg, mode)
            if a not in sums or b not in sums:
                print(f"  (missing: {a if a not in sums else b})")
                continue
            weeks = [w for w in all_weeks if w in sums[a] and w in sums[b]]
            nw = len(weeks)
            point = pooled(sums[a], weeks) - pooled(sums[b], weeks)
            idx = rng.integers(0, nw, size=(args.n_boot, nw))
            boot = np.empty(args.n_boot)
            for i in range(args.n_boot):
                boot[i] = pooled(sums[a], weeks, idx[i]) - pooled(sums[b], weeks, idx[i])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_better = float((boot < 0).mean())
            print(f"{a_cfg + ' vs ' + b_cfg:<66}{point:>8.2f}"
                  f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}"
                  f"{p_better:>9.3f}")
            results.append({
                "mode": mode, "variant": "node-dropout",
                "a": a_cfg, "b": b_cfg, "n_weeks": nw,
                "diff_mae": float(point),
                "mae_a": float(pooled(sums[a], weeks)),
                "mae_b": float(pooled(sums[b], weeks)),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "p_a_better": p_better, "n_boot": args.n_boot,
            })

    (PROJECT_ROOT / args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
