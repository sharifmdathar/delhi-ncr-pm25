#!/usr/bin/env python3
"""Summarise the rolling-origin (expanding-window) evaluation.

Reads outputs/eval_fold*.json (written by scripts/evaluate.py with
--split_mode expanding) and produces a summary table plus the consistency
verdict: does the ranking of graph structures hold as the train/test boundary
moves?
"""
import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

FOLDS = ["fold0", "fold1", "fold2", "fold3", "fold4"]
TEST_YEAR = {"fold0": 2021, "fold1": 2022, "fold2": 2023, "fold3": 2024, "fold4": 2025}
CONFIGS = ["identity", "wind_aware", "distance"]
ARCHS = ["stgcn", "graph_wavenet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/expanding_window.json")
    args = ap.parse_args()

    rows = []
    for f in FOLDS:
        p = OUT / f"eval_{f}.json"
        if not p.exists():
            continue
        data = {r["name"]: r for r in json.loads(p.read_text())}
        for arch in ARCHS:
            row = {"fold": f, "test_year": TEST_YEAR[f], "arch": arch}
            for c in CONFIGS:
                hits = [v for k, v in data.items()
                        if k.endswith(f"_{f}") and f"_{c}_s" in k and k.startswith(arch)]
                row[c] = hits[0]["mae"] if hits else None
            if any(row[c] is not None for c in CONFIGS):
                rows.append(row)

    if not rows:
        print("no fold results found")
        return

    print(f"{'arch':<16}{'fold':<8}{'test yr':>9}{'no graph':>11}{'wind-aware':>12}{'distance':>11}{'  best':<12}{'margin':>9}")
    print("-" * 88)
    wins = {}
    for r in rows:
        present = {c: r[c] for c in CONFIGS if r.get(c) is not None}
        if not present:
            continue
        best = min(present, key=present.get)
        margin = sorted(present.values())[1] - present[best] if len(present) > 1 else 0.0
        key = (r["arch"], best)
        wins[key] = wins.get(key, 0) + 1
        print(f"{r['arch']:<16}{r['fold']:<8}{r['test_year']:>9}" +
              "".join(f"{(f'{r[c]:.2f}' if r[c] is not None else '-'):>11}" for c in CONFIGS) +
              f"  {best:<12}{margin:>9.2f}")

    print("\nfolds won:", {f"{a}/{b}": n for (a, b), n in wins.items()})
    out = {"rows": rows, "wins": {f"{a}/{b}": n for (a, b), n in wins.items()}}
    (PROJECT_ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
