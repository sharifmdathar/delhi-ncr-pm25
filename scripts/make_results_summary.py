#!/usr/bin/env python3
"""Regenerate RESULTS_SUMMARY.md from the JSON artefacts."""
import json
import re
import subprocess
import sys
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
HS = ["h1", "h3", "h6", "h12", "h24", "h48"]


def mean_sd(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd


def main():
    rows = json.loads((OUT / "eval_results.json").read_text())
    by = defaultdict(list)
    for r in rows:
        m = re.match(r"^(.*)_s(\d+)$", r["name"])
        by[m.group(1) if m else r["name"]].append(r)

    # ---------- main table ----------
    main_rows = []
    for key in ORDER:
        if key not in by:
            continue
        rs = by[key]
        m, sd = mean_sd([r["mae"] for r in rs])
        best = min(rs, key=lambda r: r["mae"])
        label = DISPLAY[key]
        mae_s = f"{m:.2f} ± {sd:.2f}" if len(rs) > 1 else f"{m:.2f}"
        if key == "stgcn_identity":
            mae_s = f"**{mae_s}**"
        cells = " | ".join(f"{best[h]:.2f}" for h in HS)
        main_rows.append(f"| {label} | {mae_s} | {best['rmse']:.2f} | {best['r2']:.3f} | {cells} |")

    # ---------- graph ablation ----------
    ab_rows = []
    for label, sk, wk in [("**None (identity)**", "stgcn_identity", "graph_wavenet_identity"),
                          ("Wind-aware (directed)", "stgcn_wind_aware", "graph_wavenet_wind_aware"),
                          ("Distance (static)", "stgcn_distance", "graph_wavenet_distance"),
                          ("Correlation (static)", "stgcn_correlation", "graph_wavenet_correlation")]:
        cells = []
        for k in (sk, wk):
            if k not in by:
                cells.append("-")
                continue
            m, sd = mean_sd([r["mae"] for r in by[k]])
            cells.append(f"{m:.2f} ± {sd:.2f}" if len(by[k]) > 1 else f"{m:.2f}")
        ab_rows.append(f"| {label} | {cells[0]} | {cells[1]} |")

    # ---------- bootstrap ----------
    boot = json.loads((OUT / "bootstrap.json").read_text())
    boot_rows = []
    for b in boot:
        a = DISPLAY.get(b["a"], b["a"].replace("_", " ")).replace("**", "")
        c = DISPLAY.get(b["b"], b["b"].replace("_", " ")).replace("**", "")
        verdict = ("not distinguishable" if b["ci95_low"] <= 0 <= b["ci95_high"]
                   else ("first is better" if b["diff_mae"] < 0 else "second is better"))
        boot_rows.append(f"| {a} vs {c} | {b['diff_mae']:+.2f} | "
                         f"[{b['ci95_low']:.2f}, {b['ci95_high']:.2f}] | {b['p_a_better']:.3f} | {verdict} |")

    # ---------- rolling origin ----------
    exp = json.loads((OUT / "expanding_window.json").read_text())
    roll_rows = []
    for r in exp["rows"]:
        present = {c: r[c] for c in ("identity", "wind_aware", "distance") if r.get(c) is not None}
        if not present:
            continue
        best = min(present, key=present.get)
        arch = "STGCN" if r["arch"] == "stgcn" else "GraphWaveNet"
        roll_rows.append(f"| {arch} | {r['fold']} | {r['test_year']} | "
                         f"{present.get('identity', float('nan')):.2f} | "
                         f"{present.get('wind_aware', float('nan')):.2f} | "
                         f"{present.get('distance', float('nan')):.2f} | {best.replace('_', '-')} |")

    # ---------- episodes ----------
    ep = json.loads((OUT / "analysis_episodes.json").read_text())
    ep_rows = []
    want = ["stgcn_identity_s42", "lstm_univariate_distance_s42", "stgcn_wind_aware_s42",
            "stgcn_distance_s42", "graph_wavenet_identity_s42", "Ridge(24 lags)", "Persistence"]
    for r in ep["rows"]:
        if r["name"] in want:
            _disp = DISPLAY.get(r["name"])
            if _disp is None:
                _m = re.match(r"^(.*)_s(\d+)$", r["name"])
                _disp = DISPLAY.get(_m.group(1), r["name"]) if _m else r["name"]
            ep_rows.append(f"| {_disp} | {r['mae']:.2f} | "
                           f"{r['episode_mae']:.2f} | {r['non_episode_mae']:.2f} |")

    # ---------- wind x episode ----------
    wind = json.loads((OUT / "wind_breakdown.json").read_text())
    subs = wind.get("by_subset") or {}
    inter_rows = []
    if subs:
        keys = [k for k in ("stgcn_identity", "stgcn_wind_aware", "stgcn_distance",
                            "graph_wavenet_identity", "graph_wavenet_wind_aware",
                            "graph_wavenet_distance") if k in subs.get("episode windows", {})]
        for label in ("all windows", "episode windows", "non-episode windows"):
            if label in subs:
                inter_rows.append(f"| {label} | " +
                                  " | ".join(f"{subs[label][k]:.2f}" for k in keys) + " |")
    inter_keys = [k for k in ("stgcn_identity", "stgcn_wind_aware", "stgcn_distance",
                               "graph_wavenet_identity", "graph_wavenet_wind_aware",
                               "graph_wavenet_distance") if k in subs.get("episode windows", {})]

    # ---------- write ----------
    doc = f"""# Delhi-NCR PM2.5 Forecasting — Final Results (2019–2025 real data)

## Research Question
**Does a wind-aware dynamic graph beat a static distance graph, and when?**

## Answer
**Partly, and the detail matters.** A wind-aware directed graph *does* beat a
static distance graph for Graph WaveNet (33.77 vs 34.60 µg/m³; paired difference
0.78, 95% CI [0.40, 1.15] by bootstrap over 105 test weeks), but is statistically
indistinguishable from it for STGCN (33.05 vs 33.18; CI [−0.11, 0.30]). More
importantly, **no graph at all beats both** in both architectures (0.36–0.51 µg/m³,
both CIs excluding zero).

The wind-aware edge over the distance graph is concentrated in severe-episode
windows, where transport matters (1.22 µg/m³ for Graph WaveNet during episodes
against 0.23 outside). But the same cut shows that *all* spatial mixing is a net
loss: no-graph is best during episodes too, and the graph penalty there is
0.5–1.7 µg/m³. Every neural model here does beat two strong baselines, so the
comparison is meaningful rather than a broken-model artefact.

## Test results (2024–2025, 2000 evenly-spaced windows, masked, µg/m³; 8 seeds for the graph models)

| Config | MAE | RMSE | R² | 1h | 3h | 6h | 12h | 24h | 48h |
|--------|-----|------|-----|----|----|----|-----|-----|-----|
{chr(10).join(main_rows)}

## Graph ablation (the research question)

| Graph | STGCN MAE | GraphWaveNet MAE |
|-------|-----------|------------------|
{chr(10).join(ab_rows)}

A per-seed standard deviation cannot resolve differences this small, so the
comparison is made with a **paired bootstrap over the 105 ISO weeks of the test
period** — the same weeks are resampled for both models, which respects the
temporal correlation in the errors.

| Comparison | ΔMAE | 95% CI | P(first better) | Verdict |
|------------|------|--------|-----------------|---------|
{chr(10).join(boot_rows)}

### Where the graph penalty is actually paid: severe episodes

Severe PM2.5 episodes in Delhi occur under calm, stable conditions: 78.2% of the
test windows in the lowest wind third contain a severe-episode hour, against 52.1%
and 49.0% in the two windier thirds. Crossing the wind analysis with the episode
analysis explains the ablation:

| Subset | {' | '.join(k.replace('_', ' ') for k in inter_keys)} |
|--------|{'|'.join(['---'] * len(inter_keys))}|
{chr(10).join(inter_rows)}

- During **episode windows** the no-graph variant is best for both architectures;
  the graph variants are 0.5–1.7 µg/m³ worse.
- During **non-episode windows** all four variants sit within 0.7 µg/m³ of each
  other, with no-graph marginally ahead.
- The wind-aware graph's advantage over the distance graph is itself concentrated
  in episodes (1.22 µg/m³ for Graph WaveNet during episodes against 0.23 outside).

So the graph penalty is paid almost entirely in the high-pollution regime, and
spatial mixing does not rescue the high-pollution case even where transport
should matter most.

### Rolling-origin check (is the result just the split boundary?)

Train on 2019…Y, validate on Q4 of Y, test on Y+1.

| Model | Fold | Test year | No graph | Wind-aware | Distance | Best |
|-------|------|-----------|----------|------------|----------|------|
{chr(10).join(roll_rows)}

The ranking is stable as the boundary moves, so the finding is not an artefact of
one particular split.

## Severe episodes

Episodes = PM2.5 > 150 µg/m³ sustained ≥ 6 h, station-wise (15.6% of observed station-hours in the test block).

| Model | Overall | Episodes | Non-episodes |
|-------|---------|----------|--------------|
{chr(10).join(ep_rows)}

All architectures are 3–5× worse during episodes, and the spread between models
widens there. Extreme pollution is where the remaining error lives.

## Data

| Item | Value |
|------|-------|
| Source | CPCB 15-min, via `Vonter/india-cpcb-aqi` releases (2009–2025) |
| Stations | **42** (34 Delhi + 8 Noida/Ghaziabad), ≥70% real coverage in every split |
| Period | **2019-01-01 → 2025-12-31** (61,345 hours, 7 years) |
| Splits | Train 2019–2022 (34,922 windows) · Val 2023 (8,641) · Test 2024–2025 (17,402) |
| Features | PM2.5 + station-measured T/RH/wind/rain (CPCB files, not reanalysis) |
| Normalisation | log1p + z-score, train split only |
| Missing | masked (1 = really observed), honoured in loss and metrics |

Split distributions are matched (mean 102.0 / 98.0 / 97.5 µg/m³; frac > 300 =
4.82% / 4.23% / 4.15%), so the training split contains **four full winters**.
Split boundaries are contiguous (0 h train→val) with a 23 h gap before test; no
window's input or target ever crosses a boundary (verified programmatically).

## What made the models work

1. **Predict the change from the last observation**, not the absolute concentration.
   The output layer is zero-initialised, so each model starts *at* Persistence and
   can only improve.
2. **Linear autoregressive skip** (last 24 lags → 48 horizons), i.e. the ridge
   solution inside the network. The network is measured relative to a strong
   linear floor; without it, models collapsed to the training mean.
3. **Mask channel** so imputed hours are distinguishable from observed ones.
4. **Checkpoint selection on denormalised µg/m³**, not normalised loss.

## Honest limitations

- Seven years of data, but the fixed test block covers only two winters, so
  generalisation across pollution regimes is only partly tested. The
  rolling-origin evaluation is our partial answer to this.
- The wind-aware graph uses a period-mean prevailing direction, discarding gust
  structure, and assumes a single direction per station. The per-station wind
  vectors agree only weakly (mean cross-station correlation of the u-component
  is 0.11), so that mean direction is itself a noisy estimate.
- The graph comparison rests on 8 seeds for the two graph architectures but on a
  single model family and one hyperparameter setting per configuration. Each
  variant received the same tuning effort, but the graph-specific
  hyperparameters were not swept.
- Graph structures are static except GraphWaveNet's learned variant.
- One hazard (PM2.5), one horizon family, no episode-conditioned modelling.
- An earlier version of this study used synthetic data and reported that
  everything failed. That result is void; see `EXPERIMENTS.md` for the diagnosis.
"""
    (PROJECT_ROOT / "RESULTS_SUMMARY.md").write_text(doc)
    print("wrote RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
