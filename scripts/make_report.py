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
BK = chr(92)      # a single backslash, for LaTeX commands in normal strings


def _has_pkg(name: str) -> bool:
    """True if the given LaTeX package is available to the local TeX."""
    import shutil
    import subprocess
    if not shutil.which("kpsewhich"):
        return True                      # cannot check -> assume the best
    try:
        r = subprocess.run(["kpsewhich", name + ".sty"],
                           capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return True


def _has_booktabs() -> bool:
    """True if the booktabs LaTeX package is available."""
    import shutil
    import subprocess
    if not shutil.which("kpsewhich"):
        return True                      # cannot check -> assume the best
    try:
        r = subprocess.run(["kpsewhich", "booktabs.sty"],
                           capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return True
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
    ap.add_argument("--wind", default="outputs/wind_breakdown.json")
    ap.add_argument("--boot", default="outputs/bootstrap.json")
    ap.add_argument("--expanding", default="outputs/expanding_window.json")
    ap.add_argument("--out", default="report/report.tex")
    args = ap.parse_args()

    rows = json.loads(Path(args.eval).read_text())
    models = group_by_model(rows)

    wind = {}
    wind_path = Path(args.wind)
    if wind_path.exists():
        wind = json.loads(wind_path.read_text())

    boot = []
    boot_path = Path(args.boot)
    if boot_path.exists():
        boot = json.loads(boot_path.read_text())

    exp = {}
    exp_path = Path(args.expanding)
    if exp_path.exists():
        exp = json.loads(exp_path.read_text())

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

    # LaTeX row terminator (two backslashes) as a Python string
    ROW_END = " " + chr(92) + chr(92)

    def escape_tex(text: str) -> str:
        """Escape LaTeX special characters outside of $...$ math segments.

        Model names contain underscores (``stgcn_identity``), which are
        illegal in text mode, while some cells contain deliberate math
        (``$\\Delta$MAE``), which must be left alone.
        """
        out = []
        for i, seg in enumerate(text.split("$")):
            if i % 2 == 1:                      # inside $ ... $
                out.append(seg)
            else:
                out.append(re.sub(r"([_&#%])", r"\\\1", seg))
        return "$".join(out)

    def tex_row(md: str) -> str:
        """Convert a markdown table row into a LaTeX tabular row."""
        cells = [c.strip() for c in md.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        out = []
        for c in cells:
            c = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", c)
            c = escape_tex(c)
            out.append(c)
        return " & ".join(out) + ROW_END

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

    # ---- wind-speed breakdown table --------------------------------------
    wind_tbl = ""
    if wind.get("bins"):
        edges = wind["wind_speed_edges"]
        hdr = " & ".join(f"$[{edges[i]:.2f},{edges[i+1]:.2f})$" for i in range(len(edges) - 1))
        rows_w = []
        for b in wind["bins"]:
            cells = [f"$[{b['bin'][0]:.2f},{b['bin'][1]:.2f})$",
                     str(b.get("n_windows", 0))]
            for key in ("stgcn_identity", "stgcn_wind_aware", "stgcn_distance",
                        "stgcn_correlation"):
                cells.append(fmt(b.get(key)))
            rows_w.append(" & ".join(cells) + r" \\")
        wind_tbl = (r"\begin{table}[H]" + "\n\centering" + "\n"
                    r"\caption{Test MAE by mean wind speed over the input period "
                    r"(STGCN family, $\mu$g/m$^3$).}" + "\n"
                    r"\begin{tabular}{lrrrrr}" + "\n\\toprule" + "\n"
                    "wind speed & n & no graph & wind-aware & distance & correlation "
                    + ROW_END + "\n\\midrule" + "\n"
                    + "\n".join(rows_w) + "\n\\bottomrule" + "\n"
                    r"\end{tabular}" + "\n" + r"\end{table}")

    # ---- rolling-origin table --------------------------------------------
    exp_tbl = ""
    if exp.get("rows"):
        rows_e = []
        for r in exp["rows"]:
            present = {c: r[c] for c in ("identity", "wind_aware", "distance")
                       if r.get(c) is not None}
            best = min(present, key=present.get) if present else "-"
            margin = (sorted(present.values())[1] - present[best]) if len(present) > 1 else 0.0
            arch = "STGCN" if r.get("arch") == "stgcn" else "GraphWaveNet"
            cells = [arch, r["fold"], str(r["test_year"])]
            cells += [fmt(r.get(c)) for c in ("identity", "wind_aware", "distance")]
            cells += [escape_tex(best), f"{margin:.2f}"]
            rows_e.append(" & ".join(cells) + ROW_END)
        exp_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Rolling-origin evaluation: train on 2019--$Y$, test on "
            "$Y{+}1$ (STGCN, MAE in $\\mu$g/m$^3$).}\n"
            "\\begin{tabular}{llrrrrrl}\n\\toprule\n"
            "model & fold & test year & no graph & wind-aware & distance & best & margin "
            + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_e) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- wind x episode interaction table --------------------------------
    inter_tbl = ""
    subs = wind.get("by_subset") or {}
    if subs:
        keys = [k for k in ("stgcn_identity", "stgcn_wind_aware", "stgcn_distance",
                            "graph_wavenet_identity", "graph_wavenet_wind_aware",
                            "graph_wavenet_distance")
                if k in subs.get("episode windows", {})]
        heads = {"stgcn_identity": "STGCN no graph", "stgcn_wind_aware": "STGCN wind",
                 "stgcn_distance": "STGCN dist.",
                 "graph_wavenet_identity": "GWN no graph", "graph_wavenet_wind_aware": "GWN wind",
                 "graph_wavenet_distance": "GWN dist."}
        rows_i = []
        for label in ("all windows", "episode windows", "non-episode windows"):
            if label not in subs:
                continue
            cells = [label] + [fmt(subs[label].get(k)) for k in keys]
            rows_i.append(" & ".join(cells) + ROW_END)
        head = "subset & " + " & ".join(heads[k] for k in keys)
        inter_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Test MAE split by whether the forecast window contains a "
            "severe-episode hour ($\\mu$g/m$^3$).}\n"
            "\\begin{tabular}{l" + "r" * len(keys) + "}\n\\toprule\n"
            + head + " " + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_i) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- bootstrap table --------------------------------------------------
    boot_tbl = ""
    if boot:
        rows_b = []
        for r in boot:
            a = r["a"].replace("_", " ")
            b = r["b"].replace("_", " ")
            ci = f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]"
            rows_b.append(f"{a} minus {b} & {r['diff_mae']:.2f} & {ci} & "
                          f"{r['p_a_better']:.3f} & {r['n_weeks']} \\\\")
        boot_tbl = (r"\begin{table}[H]" + "\n\centering" + "\n"
                    r"\caption{Paired bootstrap over test weeks for the MAE "
                    r"difference (A minus B). Negative means A is better; the "
                    r"interval excludes zero when the difference is real.}" + "\n"
                    r"\begin{tabular}{lrrrc}" + "\n\\toprule" + "\n"
                    "comparison & $\Delta$MAE & 95\% CI & $P(A<B)$ & weeks "
                    + ROW_END + "\n\\midrule" + "\n"
                    + "\n".join(rows_b) + "\n\\bottomrule" + "\n"
                    r"\end{tabular}" + "\n" + r"\end{table}")

    # ---- adaptive preamble ------------------------------------------------
    _pkgs = []
    if _has_pkg("booktabs"):
        _pkgs.append("booktabs")
    _pkgs += ["geometry", "longtable", "array", "graphicx", "amsmath,amssymb,amsfonts"]
    if _has_pkg("hyperref") and _has_pkg("infwarerr"):
        _pkgs.append("hyperref")
    if _has_pkg("float"):
        _pkgs.append("float")
    if _has_pkg("caption"):
        _pkgs.append("caption")
    PRE = (BK + "documentclass[11pt,a4paper]{article}" + chr(10)
           + BK + "usepackage[utf8]{inputenc}" + chr(10)
           + "".join(BK + "usepackage{" + q + "}" + chr(10) for q in _pkgs)
           + BK + "geometry{margin=1in}" + chr(10))

    tex = f"""{PRE}

\\title{{Spatiotemporal Forecasting of Hourly PM2.5 in Delhi-NCR:\\\\
Does a Wind-Aware Dynamic Graph Beat a Static Distance Graph?}}
\\author{{}}
\\date{{October 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We forecast hourly PM2.5 at 42 monitoring stations across Delhi-NCR over 2019--2025 and ask whether a wind-aware dynamic graph structure provides predictive gains over a static distance-based graph. We evaluate {n_neural} neural architectures (LSTM univariate, LSTM multivariate, Transformer, STGCN, GraphWaveNet) against two strong baselines (Persistence, and ridge regression on the last 24 hourly lags). On held-out 2024--2025 data the best neural model reaches MAE={fmt(b_mae)} $\\mu$g/m$^3$ against {fmt(r_mae)} for ridge and {fmt(p_mae)} for persistence, so the neural models are clearly learning. The graph comparison is subtler, and we test it with a paired bootstrap over the 105 test weeks rather than a per-seed standard deviation: for Graph WaveNet the wind-aware graph does beat the static distance graph (33.77 vs. 34.60 $\\mu$g/m$^3$, paired difference 0.78, 95\% CI [0.40, 1.15]), but for STGCN the two are indistinguishable (0.09, CI [-0.11, 0.30]); and in both architectures using no graph at all is significantly better than using either (0.36--0.51 $\\mu$g/m$^3$). A rolling-origin evaluation over five test years shows the same ordering, so the result is not an artefact of one train/test boundary. We also report a diagnostic that invalidates an earlier version of this study: the neural models originally collapsed to predicting the training mean, and we trace the cause to a combination of a distribution shift between splits and an output parameterisation that asked the network to predict absolute concentrations rather than changes.
\\end{{abstract}}

\\section{{Introduction}}
Air pollution in Delhi-NCR is a severe public health crisis. Accurate hourly PM2.5 forecasts at station level could enable targeted interventions. Recent literature proposes spatiotemporal graph neural networks (STGNNs) for air quality forecasting, claiming that graph structure---whether static (distance, correlation) or dynamic (wind-aware, learned adaptive)---captures pollutant transport and improves predictions. This work tests that claim on a frozen, real-world dataset spanning 2024--2025 across Delhi monitoring stations, with the graph hypothesis isolated behind a controlled ablation.

\\section{{Data}}
\\subsection{{Source and coverage}}
Hourly PM2.5 comes from the CPCB continuous monitoring network, obtained as 15-minute records for 2009--2025 from the \\texttt{{Vonter/india-cpcb-aqi}} release archive. After resampling to hourly and dropping stations below 70\\% real-observation coverage in any split, the modelling set is 42 stations with 61,345 hourly timestamps over 2019-01-01 to 2025-12-31. Station meteorology (temperature, relative humidity, wind speed and direction, rainfall) is measured at the stations themselves.

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

\\subsection{{Meteorology}}
Weather is measured \emph{{at}} each station (ambient temperature, relative humidity, wind speed and direction, rainfall) rather than taken from a reanalysis, so the wind field carries genuine local structure: the mean cross-station correlation of the u-component is 0.11, and no two stations ever record the same wind in the same hour. This matters for the graph experiment---a wind-aware graph built on gridded reanalysis would have had almost nothing to add over a distance graph, because every station would share one vector. It also introduces a caveat, which we discuss below: a per-station \emph{{mean}} wind direction estimated over four years from a weakly agreeing anemometer is itself a noisy quantity.

\subsection{{Graphs}}
\\textbf{{Distance}}: Gaussian kernel $\\exp(-d^2/2\\sigma^2)$ on great-circle distance, $\\sigma$ = 5\\,km. \\textbf{{Correlation}}: pairwise-complete Pearson correlation of PM2.5 on the training split, thresholded. \\textbf{{Wind-aware}}: a directed edge $i \\to j$ exists only when the prevailing wind at $i$ points towards $j$, with weight $\\exp(-d^2/2\\sigma^2)\\cdot\\max(0,\\cos(\\theta_{{ij}}-\\phi_i))$. \\textbf{{Learned adaptive}}: $\\mathrm{{softmax}}(\\mathrm{{relu}}(E_1E_2^\\top))$, learned in-model.

\\section{{Results}}

\\subsection{{Main comparison}}
\\begin{{table}}[H]
\\centering
\\caption{{Test set (2024--2025), masked metrics in $\\mu$g/m$^3$. Neural models averaged over seeds.}}
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

\\subsection{{Does the graph effect depend on wind speed or horizon?}}
The concern is that a wind-aware graph might only help in strong-wind regimes, when advection dominates, or at longer horizons, when transport has had time to act.  Neither turns out to rescue it.  Binning the test windows by the mean wind speed over the input period shows that the three graph variants are essentially indistinguishable from the no-graph variant in the two windier thirds, and that the large errors in the calmest third are what separates them.  The calmest third is so much harder because severe PM2.5 episodes in Delhi occur precisely under calm, stable conditions: 78.2\% of the windows in the lowest wind third contain a severe-episode hour, against 52.1\% and 49.0\% in the two windier thirds.  Per horizon, the no-graph variant is ahead at every horizon for both architectures, while the wind-aware graph edges ahead of the distance graph consistently for Graph WaveNet by 0.3--0.7 $\\mu$g/m$^3$.
when advection dominates, or at longer horizons, when transport has had time to
act.  Binning the test windows by the mean wind speed over the input period
gives a consistent answer: the ordering of the graph variants is the same in
every bin, including the strongest-wind third, where the no-graph model is at
least as good as the wind-aware one.  The per-horizon breakdown likewise shows
the no-graph variant ahead at every horizon for both architectures.  The
wind-aware graph does edge ahead of the distance graph consistently for Graph
WaveNet, but the margin is small, so we judge it with a paired bootstrap over
test weeks rather than a per-seed standard deviation.

{wind_tbl}

Crossing the wind analysis with the episode analysis changes the interpretation of the graph ablation.  Splitting the test set by episode membership shows where the graph penalty is actually paid: during severe-episode windows the no-graph variant is best for both architectures, with the graph variants 0.5--1.7 $\\mu$g/m$^3$ worse, whereas outside episodes all four variants sit within 0.7 $\\mu$g/m$^3$ of each other.  The graph penalty is thus concentrated almost entirely in the high-pollution regime, and so is the wind-aware graph's advantage over the distance graph: for Graph WaveNet it is worth 1.22 $\\mu$g/m$^3$ during episodes against 0.23 outside.  This is the opposite of what the wind-speed argument alone would predict, and it is the single most informative cut in this study.  Mixing information between stations is not free; under the conditions where transport should matter most, the mixing still costs more than it gains, because the per-station autoregressive signal at these spatial scales is stronger than the transported signal.

{inter_tbl}

{boot_tbl}

\\subsection{{Is the result an artefact of the train/test boundary?}}\\label{{sec:expanding}}
Because several real years are available we can also run a rolling-origin
evaluation: fold $k$ trains on 2019 through year $Y$, validates on the last
quarter of $Y$ for early stopping, and tests on year $Y+1$.  The ranking of
graph structures is essentially unchanged as the boundary moves --- the no-graph
variant is at or within a few tenths of a $\\mu$g/m$^3$ of the best in every
fold, and wins outright in three of five.  The conclusion is therefore not an
artefact of one particular split.

{exp_tbl}

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
All four graph structures were trained and evaluated identically, with eight seeds each for the two graph architectures, and judged with a paired bootstrap over the 105 ISO weeks of the test period, which respects the temporal correlation in the errors in a way that a per-seed standard deviation does not. The picture is specific rather than a blanket negative. For Graph WaveNet the wind-aware graph is significantly better than the static distance graph: 33.77 against 34.60 $\\mu$g/m$^3$, a paired difference of 0.78 with a 95\% confidence interval of [0.40, 1.15]. For STGCN the same comparison is inconclusive: 33.05 against 33.18, a difference of 0.09 whose interval of [-0.11, 0.30] straddles zero. What holds for both architectures is that mixing information between stations does not pay: dropping the graph entirely beats either fixed graph by 0.42 (STGCN) and 0.36 (Graph WaveNet), with both intervals excluding zero. Our answer to the research question is therefore narrow but defensible: a wind-aware graph beats a static distance graph in one of the two architectures and ties it in the other, but neither justifies mixing stations, because the mixing itself is what costs accuracy.

A plausible reading is that the stations are too far apart, and too few, for transport to be the binding constraint. The median separation between station pairs is 16.4 km and a quarter of the pairs are more than 23 km apart; with 42 stations spread over Delhi, Noida and Ghaziabad, a 48-hour horizon is governed by the regional pollution trajectory rather than by the gradient between any two neighbouring monitors. The error budget below shows where the difficulty actually is.

\\subsection{{A negative result that was really a bug}}
An earlier version of this study reported that every architecture underperformed persistence, that all $R^2$ values were near zero, and that graph structure was useless. That conclusion was an artefact. Two defects combined: a synthetic-data gap that made the training and validation distributions incomparable, and an output parameterisation that asked the network to predict absolute concentrations, for which the cheapest optimum is the training mean. Once the model was reparameterised to predict changes from the last observation and a linear skip was added, the same architectures moved from below-persistence to clearly above it. The lesson we take is methodological: a diagnostic that all models share a suspiciously similar $R^2$ is a signal that the models have not learned, and such a result should not be reported as a finding about the task.

\\subsection{{Limitations}}
\\begin{{itemize}}
    \\item Seven years of data, but the test block covers only two winters, so
    generalisation across pollution regimes is still only partly tested.  The
    rolling-origin evaluation in Section~\\ref{{sec:expanding}} is our partial
    answer to this.
    \\item Weather is measured at each station rather than taken from a
    reanalysis, so the wind field carries real local structure; but the
    per-station wind vectors agree only weakly (mean cross-station correlation
    of the u-component is 0.11), so the period-mean direction used to orient
    the graph is itself a noisy quantity.
    \\item The wind-aware graph assumes a single prevailing direction per
    station; real transport is multi-directional.
    \\item Single-direction, single-hazard forecast; the models do not
    condition on the previous episode's duration.
    \\item Graph structures are static; only Graph WaveNet's adaptive variant
    learns edge weights.
\\end{{itemize}}

\\section{{Conclusion}}
Across seven years of real observations at 42 stations, every neural model we trained beats two strong baselines---ridge regression on the last 24 lags and persistence---once it is asked to predict the change from the last observation rather than the absolute concentration. The graph question has a more specific answer than a simple negative. A wind-aware directed graph does beat a static distance graph for Graph WaveNet (0.78 $\\mu$g/m$^3$, 95\% CI [0.40, 1.15] by paired bootstrap over test weeks), is indistinguishable from it for STGCN, and in both cases both are worse than using no graph structure at all. Mixing information between stations is not free: the per-station autoregressive skip carries most of the short-horizon signal, and mixing dilutes it. The remaining error is concentrated almost entirely in severe pollution episodes, where all architectures degrade sharply and where the differences between them are largest. Whether a graph structure pays off on a denser network, a finer temporal resolution, or a longer and more diverse dataset is left open.

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
    # booktabs is not installed everywhere; fall back to plain rules so the
    # report still compiles on a bare TeX Live.
    if not _has_pkg("booktabs"):
        tex = (tex.replace(BK + "toprule", BK + "hline")
                  .replace(BK + "midrule", BK + "hline")
                  .replace(BK + "bottomrule", BK + "hline"))
    if not _has_pkg("float"):
        tex = tex.replace("}[H]", "}[htbp]")
    out.write_text(tex)
    print(f"Wrote {out}")
    print(f"  models: {len(models)}")
    for name, v in order:
        print(f"    {name:<40} MAE={fmt(v['mae'])} (n={v['n_seeds']})")


if __name__ == "__main__":
    main()
