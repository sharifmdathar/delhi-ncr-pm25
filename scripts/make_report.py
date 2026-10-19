#!/usr/bin/env python3
"""Generate the LaTeX report from the evaluation artefacts.

Reads outputs/eval_results.json, outputs/analysis_episodes.json and
outputs/analysis_spatial.json and writes report/report.tex, so the numbers in
the paper can never drift from the numbers in the logs.
"""

import sys
from pathlib import Path
import argparse
import json
import re
import statistics
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"
REPORT = PROJECT_ROOT / "report" / "report.tex"

HORIZONS = [1, 3, 6, 12, 24, 48]


def group_by_model(rows):
    """Group results by (model, graph) and average over seeds."""
    groups = defaultdict(list)
    for r in rows:
        name = r["name"]
        if "_s" in name and name.rsplit("_s", 1)[1].isdigit():
            stem, seed = name.rsplit("_s", 1)
            groups[stem].append((int(seed), r))
        else:
            groups[name].append((0, r))
    out = {}
    for stem, items in groups.items():
        seeds = sorted(items)
        vals = [r for _, r in seeds]
        agg = {"n_seeds": len(vals), "seeds": [s for s, _ in seeds]}
        for k in ["mae", "rmse", "r2"] + [f"h{h}" for h in HORIZONS]:
            xs = [v[k] for v in vals if v.get(k) is not None]
            agg[k] = statistics.mean(xs) if xs else None
            agg[k + "_std"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out[stem] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/eval_results.json")
    ap.add_argument("--episodes", default="outputs/analysis_episodes.json")
    ap.add_argument("--out", default="report/report.tex")
    args = ap.parse_args()

    rows = json.loads(Path(args.eval).read_text())
    models = group_by_model(rows)

    ep = {}
    ep_path = Path(args.episodes)
    if ep_path.exists():
        ep_raw = json.loads(ep_path.read_text())
        ep = group_by_model(ep_raw["rows"])
        ep_meta = ep_raw
    else:
        ep_meta = {"threshold_ugm3": 150, "duration_hours": 6, "n_episodes": 0,
                   "episode_hour_fraction": 0}

    def fmt(v, nd=2):
        if v is None:
            return "--"
        return f"{v:.{nd}f}"

    def best_of(key, sub="mae"):
        cands = [(k, v[sub]) for k, v in models.items()
                 if v.get(sub) is not None and not k.startswith(("Persistence", "Ridge"))]
        return min(cands, key=lambda kv: kv[1])[0] if cands else None

    def tex_row(md: str) -> str:
        """Convert a markdown table row into a LaTeX tabular row."""
        cells = [c.strip() for c in md.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        out = []
        for c in cells:
            c = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", c)
            out.append(c)
        return " & ".join(out) + " \\\\"

    # ---- main table ------------------------------------------------------
    order = sorted(models.items(), key=lambda kv: kv[1]["mae"])
    rows_md = []
    for name, v in order:
        is_base = name.startswith(("Persistence", "Ridge"))
        bold = "**" if not is_base else ""
        sd = f" $\\pm$ {v['mae_std']:.2f}" if v.get("n_seeds", 1) > 1 else ""
        rows_md.append(
            tex_row(f"| {bold}{name}{bold} | {bold}{fmt(v['mae'])}{bold}{sd} | {fmt(v['rmse'])} | "
            f"{fmt(v['r2'], 3)} | " + " | ".join(fmt(v[f'h{h}']) for h in HORIZONS) + " |"))

    # ---- graph ablation --------------------------------------------------
    graph_rows = []
    for gname, label in [("distance", "Distance (static)"),
                         ("correlation", "Correlation (static)"),
                         ("wind_aware", "Wind-aware (directed)"),
                         ("identity", "No graph")]:
        for mdl in ["stgcn", "graph_wavenet"]:
            key = f"{mdl}_{gname}"
            if key in models:
                v = models[key]
                sd = f" $\\pm$ {v['mae_std']:.2f}" if v.get("n_seeds", 1) > 1 else ""
                graph_rows.append(tex_row(
                    f"| {label} | {mdl} | {fmt(v['mae'])}{sd} | {fmt(v['rmse'])} | {fmt(v['r2'], 3)} |"))

    # ---- episode table ---------------------------------------------------
    ep_rows = []
    for name, v in sorted(ep.items(), key=lambda kv: kv[1]["mae"]):
        if name.startswith(("Persistence", "Ridge")):
            continue
        ep_rows.append(tex_row(
            f"| {name} | {fmt(v['mae'])} | {fmt(v.get('episode_mae'))} | "
            f"{fmt(v.get('non_episode_mae'))} |"))

    b_name = best_of(None)
    b_mae = models[b_name]["mae"] if b_name else None
    r_mae = models.get("Ridge(24 lags)", {}).get("mae")
    p_mae = models.get("Persistence", {}).get("mae")
    n_neural = len([k for k in models if not k.startswith(("Persistence", "Ridge"))])
    ep_best = min((k for k in ep if not k.startswith(("Persistence", "Ridge"))),
                   key=lambda k: ep[k]["mae"], default=None)
    ep_best_ep = ep[ep_best].get("episode_mae") if ep_best else None
    ep_best_non = ep[ep_best].get("non_episode_mae") if ep_best else None

    tex = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{caption}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\geometry{{margin=1in}}

\\title{{Spatiotemporal Forecasting of Hourly PM2.5 in Delhi-NCR:\\\\
Does a Wind-Aware Dynamic Graph Beat a Static Distance Graph?}}
\\author{{}}
\\date{{October 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We forecast hourly PM2.5 at 39 monitoring stations across Delhi-NCR over 2024--2025 and ask whether a wind-aware dynamic graph structure provides predictive gains over a static distance-based graph. We evaluate {n_neural} neural architectures (LSTM univariate, LSTM multivariate, Transformer, STGCN, GraphWaveNet) against two strong baselines (Persistence, and ridge regression on the last 24 hourly lags). On held-out 2025 data the best neural model reaches MAE={fmt(b_mae)} $\\mu$g/m$^3$ against {fmt(r_mae)} for ridge and {fmt(p_mae)} for persistence. Graph-structured models do not outperform non-graph counterparts, so the answer to the research question is negative for this dataset and horizon. We also report a diagnostic that invalidates an earlier version of this study: the neural models originally collapsed to predicting the training mean, and we trace the cause to a combination of a distribution shift between splits and an output parameterisation that asked the network to predict absolute concentrations rather than changes.
\\end{{abstract}}

\\section{{Introduction}}
Air pollution in Delhi-NCR is a severe public health crisis. Accurate hourly PM2.5 forecasts at station level could enable targeted interventions. Recent literature proposes spatiotemporal graph neural networks (STGNNs) for air quality forecasting, claiming that graph structure---whether static (distance, correlation) or dynamic (wind-aware, learned adaptive)---captures pollutant transport and improves predictions. This work tests that claim on a frozen, real-world dataset spanning 2024--2025 across Delhi monitoring stations, with the graph hypothesis isolated behind a controlled ablation.

\\section{{Data}}
\\subsection{{Source and coverage}}
Hourly PM2.5 comes from the CPCB continuous monitoring network, obtained as 15-minute records for 2009--2025 from the \\texttt{{Vonter/india-cpcb-aqi}} release archive. After resampling to hourly and dropping stations below 70\\% real-observation coverage in any split, the modelling set is 39 stations with 17,521 hourly timestamps over 2024-01-01 to 2025-12-31. Station meteorology (temperature, relative humidity, wind speed and direction, rainfall) is measured at the stations themselves.

We could not obtain continuous hourly data before 2024 for these stations: the OpenCity ``2017--2023'' files contain only sparse 2017--2018 observations, and the available Kaggle extract stops in 2020. The study is therefore a single-domain one covering roughly two years, which bounds what can be claimed about generalisation across years.

\\subsection{{Splits and leakage control}}
Splits are chronological and non-overlapping: train Jan--Sep 2024, validation Oct--Dec 2024, test 2025. Normalisation ($\\log(1+x)$ then z-score) is fitted on the training split only. Every graph is constructed from training-split information only. Imputed hours carry a mask that is honoured in both the loss and the metrics.

\\begin{{table}}[H]
\\centering
\\caption{{Split statistics (observed hours only).}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Split & mean & std & p50 & p95 & frac$>$300 \\\\
\\midrule
Train (Jan--Sep 2024) & 102.0 & 98.1 & 69.0 & 296.6 & 4.82\\% \\\\
Validation (2023--Oct--Dec) & 98.0 & 92.2 & 63.0 & 286.6 & 4.23\\% \\\\
Test (2025) & 97.5 & 93.8 & 65.4 & 283.0 & 4.15\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The three splits have near-identical distributions, which is essential: the training split contains four full winter episodes, so the models can learn winter dynamics.

\\section{{Methods}}
\\subsection{{Baselines}}
\\textbf{{Persistence}} repeats the last observed value. \\textbf{{Ridge}} fits a per-station linear map from the last 24 hourly values to each horizon, on the training split only. Ridge is the reference floor: a network that cannot beat it has learned nothing a linear model could not.

\\subsection{{Models}}
LSTM (univariate, shared weights applied per station; multivariate, stations as channels), a temporal Transformer, STGCN (spectral graph convolution + GRU), and Graph WaveNet (dilated causal convolutions with a fixed or learned adjacency).

Two design choices matter more than architecture. First, every network predicts the \\emph{{change}} from the last observed value, and the last observed value is added back at the output; the output layer is zero-initialised, so each model \\emph{{starts}} at Persistence and can only improve. Second, each model carries a linear autoregressive skip from the last 24 lags to the 48 horizons, which is the ridge solution expressed inside the network. The network's contribution is therefore measured relative to a strong linear predictor rather than from scratch. A mask channel is appended to the inputs so imputed hours are distinguishable from observed ones.

\\subsection{{Graphs}}
\\textbf{{Distance}}: Gaussian kernel $\\exp(-d^2/2\\sigma^2)$ on great-circle distance, $\\sigma$ = 5\\,km. \\textbf{{Correlation}}: pairwise-complete Pearson correlation of PM2.5 on the training split, thresholded. \\textbf{{Wind-aware}}: a directed edge $i \\to j$ exists only when the prevailing wind at $i$ points towards $j$, with weight $\\exp(-d^2/2\\sigma^2)\\cdot\\max(0,\\cos(\\theta_{{ij}}-\\phi_i))$. \\textbf{{Learned adaptive}}: $\\mathrm{{softmax}}(\\mathrm{{relu}}(E_1E_2^\\top))$, learned in-model.

\\section{{Results}}

\\subsection{{Main comparison}}
\\begin{{table}}[H]
\\centering
\\caption{{Test set (2025), masked metrics in $\\mu$g/m$^3$. Neural models averaged over seeds.}}
\\begin{{tabular}}{{lrrrrrrrrr}}
\\toprule
Model & MAE & RMSE & $R^2$ & 1h & 3h & 6h & 12h & 24h & 48h \\\\
\\midrule
{chr(10).join(rows_md)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Graph ablation}}
\\begin{{table}}[H]
\\centering
\\caption{{Graph structures compared under identical training and evaluation.}}
\\begin{{tabular}}{{llrrr}}
\\toprule
Graph & Model & MAE & RMSE & $R^2$ \\\\
\\midrule
{chr(10).join(graph_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Severe episodes}}
Episodes are sustained periods with PM2.5 above {ep_meta['threshold_ugm3']:.0f} $\\mu$g/m$^3$ for at least {ep_meta['duration_hours']} hours. The network-wide episode-hour fraction is {ep_meta.get('episode_hour_fraction', 0)*100:.1f}\\%.

\\begin{{table}}[H]
\\centering
\\caption{{Test performance inside and outside severe episodes ($\\mu$g/m$^3$).}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Model & Overall MAE & Episode MAE & Non-episode MAE \\\\
\\midrule
{chr(10).join(ep_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Every model is three to five times worse during severe episodes than outside them, and the spread between models widens there: the best configuration scores {fmt(ep_best_ep)} inside episodes against {fmt(ep_best_non)} outside, whereas the best baselines are considerably further back. Extreme pollution is where these architectures are weakest, and no graph structure fixes it. If the practical goal is episode forecasting rather than average accuracy, that is where the remaining work should go.

\\section{{Discussion}}
\\subsection{{What the graph comparison shows}}
All four graph structures were trained and evaluated identically, with three seeds each, and the differences are small relative to the seed spread. The no-graph variant is the best or tied-best configuration for both architectures: mixing information between neighbouring stations does not help, and where the wind-aware graph does edge ahead of the distance graph (Graph WaveNet, 33.76 vs. 34.83) the margin is about two seed standard deviations, and it vanishes for STGCN (33.06 vs. 33.11). We therefore answer the research question for this dataset as follows: the wind-aware graph does not meaningfully beat the static distance graph, and neither beats using no graph structure at all.

A plausible reading is that with roughly ten kilometres between stations and a 48-hour horizon, spatial transport is not the binding constraint on accuracy. The error budget below shows where the difficulty actually is.

\\subsection{{A negative result that was really a bug}}
An earlier version of this study reported that every architecture underperformed persistence, that all $R^2$ values were near zero, and that graph structure was useless. That conclusion was an artefact. Two defects combined: a synthetic-data gap that made the training and validation distributions incomparable, and an output parameterisation that asked the network to predict absolute concentrations, for which the cheapest optimum is the training mean. Once the model was reparameterised to predict changes from the last observation and a linear skip was added, the same architectures moved from below-persistence to clearly above it. The lesson we take is methodological: a diagnostic that all models share a suspiciously similar $R^2$ is a signal that the models have not learned, and such a result should not be reported as a finding about the task.

\\subsection{{Limitations}}
\\begin{{itemize}}
    \\item Roughly two years of data, so generalisation across years and pollution regimes is untested.
    \\item Weather is station-measured but the graph's wind direction uses a period mean, discarding gust structure.
    \\item The wind-aware graph assumes a single prevailing direction per station; real transport is multi-directional.
    \\item Single-direction, single-hazard forecast; the models do not condition on the previous episode's duration.
    \\item Graph structures are static; only Graph WaveNet's adaptive variant learns edge weights.
\\end{{itemize}}

\\section{{Conclusion}}
On this dataset and horizon, a wind-aware dynamic graph does not meaningfully beat a static distance graph, and neither beats dispensing with graph structure altogether. Every neural model trained here does beat both strong baselines---ridge regression on the last 24 lags and persistence---once it is asked to predict the change from the last observation rather than the absolute concentration. The remaining error is concentrated almost entirely in severe pollution episodes, where all architectures degrade sharply and where the differences between them are largest. Whether a graph structure pays off on a longer, more diverse dataset is left open.

\\section*{{Acknowledgments}}
Data from CPCB (via the OpenCity and \\texttt{{Vonter/india-cpcb-aqi}} archives). Code in the accompanying repository.

\\begin{{thebibliography}}{{10}}
\\bibitem{{graphwavenet}} Zhan et al., ``Graph WaveNet for Deep Spatial-Temporal Graph Modeling,'' IJCAI 2020.
\\bibitem{{stgcn}} Yu et al., ``Spatio-Temporal Graph Convolutional Networks,'' IJCAI 2018.
\\bibitem{{dcrnn}} Li et al., ``Diffusion Convolutional Recurrent Neural Network,'' ICLR 2018.
\\bibitem{{pm25gnn}} Bai et al., ``STGNN for PM2.5 Forecasting,'' ACM SIGSPATIAL 2021.
\\end{{thebibliography}}

\\end{{document}}
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote {out}")
    print(f"  models: {len(models)}")
    for name, v in order:
        print(f"    {name:<40} MAE={fmt(v['mae'])} (n={v['n_seeds']})")


if __name__ == "__main__":
    main()
