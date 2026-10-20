#!/usr/bin/env python3
"""Regenerate the results tables in README.md and RESULTS_SUMMARY.md from the
JSON artefacts, so the prose and the numbers cannot drift apart."""
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

DISPLAY = {
    "stgcn_identity": "**STGCN, no graph**",
    "lstm_univariate_distance": "LSTM Univariate",
    "stgcn_wind_aware": "STGCN, wind-aware",
    "stgcn_distance": "STGCN, distance",
    "graph_wavenet_identity": "GraphWaveNet, no graph",
    "stgcn_correlation": "STGCN, correlation",
    "graph_wavenet_wind_aware": "GraphWaveNet, wind-aware",
    "graph_wavenet_distance": "GraphWaveNet, distance",
    "graph_wavenet_correlation": "GraphWaveNet, correlation",
    "lstm_multivariate_distance": "LSTM Multivariate",
    "transformer_distance": "Transformer",
    "Ridge(24 lags)": "Ridge (24 lags)",
    "Persistence": "Persistence",
}
ORDER = list(DISPLAY)


def load():
    rows = json.loads((OUT / "eval_results.json").read_text())
    by = defaultdict(list)
    for r in rows:
        m = re.match(r"^(.*)_s(\d+)$", r["name"])
        key = m.group(1) if m else r["name"]
        by[key].append(r)
    return by


def main():
    by = load()
    boot = {f"{b['a']}|{b['b']}": b for b in json.loads((OUT / "bootstrap.json").read_text())}

    # ---------------- main table ----------------
    lines = []
    for key in ORDER:
        if key not in by:
            continue
        rs = by[key]
        maes = [r["mae"] for r in rs]
        n = len(maes)
        mean = sum(maes) / n
        sd = (sum((m - mean) ** 2 for m in maes) / (n - 1)) ** 0.5 if n > 1 else 0.0
        best = min(rs, key=lambda r: r["mae"])
        label = DISPLAY[key]
        mae_s = f"{mean:.2f} ± {sd:.2f}" if n > 1 else f"{mean:.2f}"
        if key == "stgcn_identity":
            mae_s = f"**{mae_s}**"
        h = best
        h1 = f"{h.get('h1', float('nan')):.2f}"
        h48 = f"{h.get('h48', float('nan')):.2f}"
        lines.append(f"| {label} | {mae_s} | {best['rmse']:.2f} | {best['r2']:.3f} | {h1} | {h48} |")
    main_table = "\n".join(lines)

    # ---------------- graph ablation ----------------
    ab = []
    for label, sk, wk in [("**None (identity)**", "stgcn_identity", "graph_wavenet_identity"),
                          ("Wind-aware (directed)", "stgcn_wind_aware", "graph_wavenet_wind_aware"),
                          ("Distance (static)", "stgcn_distance", "graph_wavenet_distance"),
                          ("Correlation (static)", "stgcn_correlation", "graph_wavenet_correlation")]:
        cells = []
        for k in (sk, wk):
            if k not in by:
                cells.append("-")
                continue
            maes = [r["mae"] for r in by[k]]
            n = len(maes)
            mean = sum(maes) / n
            sd = (sum((m - mean) ** 2 for m in maes) / (n - 1)) ** 0.5 if n > 1 else 0.0
            cells.append(f"{mean:.2f} ± {sd:.2f}" if n > 1 else f"{mean:.2f}")
        ab.append(f"| {label} | {cells[0]} | {cells[1]} |")
    ablation = "\n".join(ab)

    # ---------------- bootstrap ----------------
    bl = []
    for key in ["stgcn_distance|stgcn_wind_aware", "graph_wavenet_distance|graph_wavenet_wind_aware",
                "stgcn_distance|stgcn_identity", "stgcn_wind_aware|stgcn_identity",
                "graph_wavenet_distance|graph_wavenet_identity", "graph_wavenet_wind_aware|graph_wavenet_identity"]:
        b = boot.get(key)
        if not b:
            continue
        a = DISPLAY.get(b["a"], b["a"].replace("_", " "))
        c = DISPLAY.get(b["b"], b["b"].replace("_", " "))
        verdict = ("not distinguishable" if b["ci95_low"] <= 0 <= b["ci95_high"]
                   else ("A better" if b["diff_mae"] < 0 else "B better"))
        bl.append(f"| {a} vs {c} | {b['diff_mae']:+.2f} | "
                  f"[{b['ci95_low']:.2f}, {b['ci95_high']:.2f}] | {b['p_a_better']:.3f} | {verdict} |")
    bootstrap = "\n".join(bl)

    # ---------------- rolling origin ----------------
    exp = json.loads((OUT / "expanding_window.json").read_text())
    el = []
    for r in exp["rows"]:
        present = {c: r[c] for c in ("identity", "wind_aware", "distance") if r.get(c) is not None}
        if not present:
            continue
        best = min(present, key=present.get)
        arch = "STGCN" if r["arch"] == "stgcn" else "GraphWaveNet"
        el.append(f"| {arch} | {r['fold']} | {r['test_year']} | "
                  f"{present.get('identity', float('nan')):.2f} | "
                  f"{present.get('wind_aware', float('nan')):.2f} | "
                  f"{present.get('distance', float('nan')):.2f} | {best.replace('_', '-')} |")
    rolling = "\n".join(el)

    print("###MAIN###")
    print(main_table)
    print("###ABLATION###")
    print(ablation)
    print("###BOOTSTRAP###")
    print(bootstrap)
    print("###ROLLING###")
    print(rolling)


if __name__ == "__main__":
    main()
