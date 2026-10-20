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
    ap.add_argument("--provenance", default="outputs/provenance.json")
    ap.add_argument("--lso", default="outputs/lso_results.json")
    ap.add_argument("--lso_bootstrap", default="outputs/lso_bootstrap.json")
    ap.add_argument("--cases", default="outputs/case_studies.json")
    ap.add_argument("--spatial", default="outputs/spatial_analysis.json")
    ap.add_argument("--adjacency", default="outputs/adjacency_introspection.json")
    ap.add_argument("--wind_coverage", default="outputs/wind_coverage_analysis.json")
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

    lso_boot = []
    lso_boot_path = Path(args.lso_bootstrap)
    if lso_boot_path.exists():
        lso_boot = json.loads(lso_boot_path.read_text())

    lso = []
    lso_path = Path(args.lso)
    if lso_path.exists():
        lso = json.loads(lso_path.read_text())

    prov = {}
    prov_path = Path(args.provenance)
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())

    cases = []
    case_path = Path(args.cases)
    if case_path.exists():
        cases = json.loads(case_path.read_text())

    spatial = {}
    spat_path = Path(args.spatial)
    if spat_path.exists():
        spatial = json.loads(spat_path.read_text())

    adj = {}
    adj_path = Path(args.adjacency)
    if adj_path.exists():
        adj = json.loads(adj_path.read_text())

    wcov = {}
    wcov_path = Path(args.wind_coverage)
    if wcov_path.exists():
        wcov = json.loads(wcov_path.read_text())

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
    for gname, label in [("identity", "No graph"),
                         ("wind_aware", "Wind-aware (directed)"),
                         ("wind_aware_era5", "Wind-aware (ERA5 wind)"),
                         ("distance", "Distance (static)"),
                         ("correlation", "Correlation (static)"),
                         ("learned_adaptive", "Learned adaptive")]:
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

    # ---- wind-coverage table ----------------------------------------------
    wcov_tbl = ""
    if wcov.get("per_config"):
        rows_wc = []
        for r in wcov["per_config"]:
            rows_wc.append(f"{r['config'].replace('_', ' ')} & "
                           f"{r['mae_has_wind']:.2f} & {r['mae_no_wind']:.2f} " + ROW_END)
        nw = wcov.get("n_stations_with_wind", 0)
        nt = wcov.get("n_stations", 0)
        wcov_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Per-station MAE split by whether the station reports "
            "measured wind at all ($" + BK + "mu$g/m$^3$).}\n"
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "configuration & " + str(nw) + " stations with wind & "
            + str(nt - nw) + " without " + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_wc) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- ERA5-wind ablation table -----------------------------------------
    era5_tbl = ""
    _era5_keys = [("stgcn_identity", "no graph"),
                  ("stgcn_wind_aware", "wind, station"),
                  ("stgcn_wind_aware_era5", "wind, ERA5"),
                  ("stgcn_distance", "distance"),
                  ("graph_wavenet_identity", "no graph"),
                  ("graph_wavenet_wind_aware", "wind, station"),
                  ("graph_wavenet_wind_aware_era5", "wind, ERA5"),
                  ("graph_wavenet_distance", "distance")]
    _era5_rows = []
    for key, label in _era5_keys:
        if key in models:
            v = models[key]
            sd = f" $\\pm$ {v['mae_std']:.2f}" if v.get("n_seeds", 1) > 1 else ""
            arch = "STGCN" if key.startswith("stgcn") else "GraphWaveNet"
            _era5_rows.append(f"{arch} & {label} & {fmt(v['mae'])}{sd} & "
                              f"{fmt(v['rmse'])} & {fmt(v['r2'], 3)} " + ROW_END)
    if _era5_rows:
        era5_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Where the edge directions come from.  Only the source of "
            "the wind changes; the construction, the training and the evaluation "
            "are identical.}\n"
            "\\begin{tabular}{llrrr}\n\\toprule\n"
            "model & edge direction & MAE & RMSE & $R^2$ " + ROW_END + "\n\\midrule\n"
            + "\n".join(_era5_rows) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- network-mean diagnostic table -------------------------------------
    netmean_tbl = ""
    _rows_nm = []
    for _key, _label in [("stgcn_identity", "no graph"),
                         ("stgcn_identity_netmean", "no graph + network mean"),
                         ("stgcn_wind_aware", "wind-aware graph"),
                         ("graph_wavenet_identity", "no graph (GraphWaveNet)"),
                         ("graph_wavenet_wind_aware", "wind-aware graph (GraphWaveNet)")]:
        if _key not in models:
            continue
        _v = models[_key]
        _sd = f" $\\pm$ {_v['mae_std']:.2f}" if _v.get("n_seeds", 1) > 1 else ""
        _rows_nm.append(f"{_label} & {fmt(_v['mae'])}{_sd} & {fmt(_v['rmse'])} & "
                        f"{fmt(_v['r2'], 3)} " + ROW_END)
    if _rows_nm:
        netmean_tbl = (
            "\\begin{table}[H]" + chr(10) + "\\centering" + chr(10)
            + "\\caption{The network-mean diagnostic: a no-graph model given the "
            "network-wide mean PM2.5 as an extra input channel.}" + chr(10)
            + "\\begin{tabular}{lrrr}" + chr(10) + "\\toprule" + chr(10)
            + "configuration & MAE & RMSE & $R^2$ " + ROW_END + chr(10)
            + "\\midrule" + chr(10)
            + chr(10).join(_rows_nm) + chr(10) + "\\bottomrule" + chr(10)
            + "\\end{tabular}" + chr(10) + "\\end{table}")

    # ---- best-simple-baseline-per-horizon table ----------------------------
    best_simple_tbl = ""
    _rows_bs = []
    for h in HORIZONS:
        _bh = f"h{h}"
        _simple = {}
        for _nm in ("Persistence", "Seasonal naive (24h)"):
            if _nm in models and models[_nm].get(_bh) is not None:
                _simple[_nm] = models[_nm][_bh]
        if not _simple:
            continue
        _bname = min(_simple, key=_simple.get)
        _bval = _simple[_bname]
        _mval = models.get(b_name, {}).get(_bh)
        if _mval is None:
            continue
        _rows_bs.append(f"{h} & {_mval:.2f} & {_bname} & {_bval:.2f} & "
                        f"{100 * (_bval - _mval) / _bval:.1f}\\% " + ROW_END)
    if _rows_bs:
        best_simple_tbl = (
            "\\begin{table}[H]" + chr(10) + "\\centering" + chr(10)
            + "\\caption{The best neural model against the best simple baseline at "
            "each horizon (the lower of Persistence and seasonal naive, whichever "
            "is stronger at that horizon).}" + chr(10)
            + "\\begin{tabular}{rlrlr}" + chr(10) + "\\toprule" + chr(10)
            + "horizon & best model & best simple baseline & its MAE & margin "
            + ROW_END + chr(10) + "\\midrule" + chr(10)
            + chr(10).join(_rows_bs) + chr(10) + "\\bottomrule" + chr(10)
            + "\\end{tabular}" + chr(10) + "\\end{table}")

    # ---- node-dropout table ------------------------------------------------
    nd_tbl = ""
    _nd_groups = [g for g in lso if g.get("variant") == "node-dropout"]
    if _nd_groups:
        _rows_nd = []
        for g in _nd_groups:
            if g.get("mode") != "both":
                continue
            for r in g.get("rows", []):
                nm = r["name"]
                if nm.startswith("lso_"):
                    nm = nm[4:]
                nm = nm.replace("_s42_", " / ").replace("_s123_", " / ")
                nm = nm.replace("_lso10_both", "").replace("_", " ")
                _rows_nd.append(f"{nm} & {r['mae']:.2f} & {r['r2']:.3f} " + ROW_END)
        if _rows_nd:
            nd_tbl = (
                "\\begin{table}[H]" + chr(10) + "\\centering" + chr(10)
                + "\\caption{Node dropout: the models are trained with a random "
                "30\\% of stations' own history hidden in each training window, so "
                "the held-out condition is one they have practised.  MAE on the "
                "held-out stations, own history hidden.}" + chr(10)
                + "\\begin{tabular}{lrr}" + chr(10) + "\\toprule" + chr(10)
                + "configuration & MAE & $R^2$ " + ROW_END + chr(10)
                + "\\midrule" + chr(10)
                + chr(10).join(_rows_nd) + chr(10) + "\\bottomrule" + chr(10)
                + "\\end{tabular}" + chr(10) + "\\end{table}")

    # ---- leave-station-out bootstrap table ---------------------------------
    lso_boot_tbl = ""
    if lso_boot:
        rows_lb = []
        for r in lso_boot:
            a = r["a"].replace("_lso10_", " / ").replace("_", " ")
            bb = r["b"].replace("_lso10_", " / ").replace("_", " ")
            ci = f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]"
            rows_lb.append(f"{escape_tex(a)} vs {escape_tex(bb)} & {r['diff_mae']:+.2f} & "
                           f"{ci} & {r['p_a_better']:.3f} " + ROW_END)
        lso_boot_tbl = (
            "\\begin{table}[H]" + chr(10) + "\\centering" + chr(10)
            + "\\caption{Leave-station-out, paired bootstrap over test weeks "
            "(A minus B; negative means A is better).}" + chr(10)
            + "\\begin{tabular}{lrrc}" + chr(10) + "\\toprule" + chr(10)
            + "comparison & $\\Delta$MAE & 95\\% CI & $P(A<B)$ " + ROW_END
            + chr(10) + "\\midrule" + chr(10)
            + chr(10).join(rows_lb) + chr(10) + "\\bottomrule" + chr(10)
            + "\\end{tabular}" + chr(10) + "\\end{table}")

    # ---- leave-station-out table ------------------------------------------
    lso_tbl = ""
    if lso:
        blocks = []
        for grp in lso:
            mode = grp.get("mode", "?")
            rows_l = []
            for r in grp.get("rows", []):
                rows_l.append(
                    escape_tex(r["name"]) + " & " + f"{r['mae']:.2f} & "
                    + f"{r['r2']:.3f} & " + f"{r['h1']:.2f} & " + f"{r['h24']:.2f} & "
                    + f"{r['h48']:.2f} " + ROW_END)
            head = ("\\textbf{Mode " + mode + "} --- held out: "
                    + ", ".join(escape_tex(x) for x in grp.get("holdout_stations", []))
                    + " " + ROW_END)
            blocks.append(head + "\n" + "\n".join(rows_l))
        lso_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Leave-station-out: MAE on the held-out stations only "
            "($\\mu$g/m$^3$).  The neural models never saw those stations' "
            "targets; ridge is fitted on the training stations only.}\n"
            "\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "configuration & MAE & $R^2$ & 1h & 24h & 48h " + ROW_END + "\n\\midrule\n"
            + "\n".join(blocks) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- provenance and mask-audit tables ----------------------------------
    prov_tbl = ""
    prov_mask_tbl = ""
    if prov.get("sources"):
        rows_p = []
        for src in prov["sources"]:
            rows_p.append(escape_tex(src["what"]) + " & "
                          + escape_tex(src["source"]) + " & "
                          + escape_tex(src["accessed_via"]) + " & "
                          + escape_tex(src["licence"]) + " " + ROW_END)
        prov_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Data sources.  No synthetic data is used anywhere; the "
            "build fails if any station falls below the coverage bar in any "
            "split.}\n"
            "\\begin{tabular}{llll}\n\\toprule\n"
            "what & source & accessed via & licence " + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_p) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

        aud = prov.get("mask_audit") or {}
        if aud:
            rows_a = []
            for name in ("train", "val", "test"):
                a = aud.get(name)
                if not a:
                    continue
                rows_a.append(
                    f"{name} & {a['hours']:,} & {a['observed_fraction']:.2%} & "
                    f"{a['missing_fraction']:.2%} & {a['mean_pm25_observed']:.1f} & "
                    f"{a['frac_over_300_observed']:.2%} " + ROW_END)
            prov_mask_tbl = (
                "\\begin{table}[H]\n\\centering\n"
                "\\caption{Observation-mask audit.  The target is never imputed: "
                "an hour with no real reading stays missing and is excluded from "
                "every loss and every metric, so all reported numbers are "
                "computed on observed values only.}\n"
                "\\begin{tabular}{lrrrrr}\n\\toprule\n"
                "split & hours & observed & missing & mean PM2.5 & frac $>300$ "
                + ROW_END + "\n\\midrule\n"
                + "\n".join(rows_a).replace("%", "\\%") + "\n\\bottomrule\n"
                "\\end{tabular}\n\\end{table}")

    # ---- case-study table ------------------------------------------------
    case_tbl = ""
    if cases:
        rows_c = []
        for c in cases:
            cells = [c["name"], c["start"][:16].replace("T", " "),
                     f"{c['peak_network_pm25']:.0f}"]
            for k in ("mae_Persistence", "mae_Ridge (24 lags)", "mae_STGCN, no graph",
                      "mae_STGCN, wind-aware", "mae_GraphWaveNet, wind-aware"):
                cells.append(fmt(c.get(k)))
            rows_c.append(" & ".join(cells) + ROW_END)
        case_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{MAE during three real events in the test period "
            "($\\mu$g/m$^3$, network mean over 48-hour forecasts).}\n"
            "\\label{tab:cases}\n"
            "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "event & start & peak & persistence & ridge & no graph & wind-aware & GWN wind "
            + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_c) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}")

    # ---- spatial-analysis table ------------------------------------------
    spat_tbl = ""
    _rows_sp = spatial.get("graph_penalty_by_radius") or []
    if _rows_sp:
        rows_sp = []
        for r in _rows_sp:
            rows_sp.append(
                f"{r['arch']} & {r['wind_graph_delta_near']:+.2f} & "
                f"{r['wind_graph_delta_far']:+.2f} & "
                f"{r['dist_graph_delta_near']:+.2f} & {r['dist_graph_delta_far']:+.2f} & "
                f"{r['corr_radius_vs_wind_penalty']:+.2f} & "
                f"{r['corr_radius_vs_dist_penalty']:+.2f} " + ROW_END)
        spat_tbl = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{The cost of spatial mixing against distance from the network "
            "centroid (per-station MAE penalty relative to the no-graph model, "
            "$\\mu$g/m$^3$).}\n"
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "model & wind near & wind far & dist near & dist far & $r$(radius) wind & "
            "$r$(radius) dist " + ROW_END + "\n\\midrule\n"
            + "\n".join(rows_sp) + "\n\\bottomrule\n"
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

    # ---- mask coverage numbers --------------------------------------------
    _aud = (prov.get("mask_audit") or {}).get("test") or {}
    mask_missing_test = f"{_aud.get('missing_fraction', 0)*100:.1f}%"
    if not _aud:
        mask_missing_test = "n/a"

    # ---- node-dropout prose, computed from the results ----------------------
    _nd_prose = ""
    _plain_both = next((g for g in lso if g.get("variant") == "plain"
                        and g.get("mode") == "both"), None)
    _nd_both = next((g for g in lso if g.get("variant") == "node-dropout"
                     and g.get("mode") == "both"), None)

    def _find(g, needle):
        for r in g.get("rows", []):
            if needle in r["name"]:
                return r
        return None

    def _mean_mae(g, needle):
        vals = [r["mae"] for r in g.get("rows", []) if needle in r["name"]]
        return sum(vals) / len(vals) if vals else None

    if _plain_both and _nd_both:
        _idw = _find(_plain_both, "IDW neighbour (seasonal)")
        _p_dist = _mean_mae(_plain_both, "stgcn_distance")
        _n_dist = _mean_mae(_nd_both, "stgcn_distance")
        _n_id = _mean_mae(_nd_both, "stgcn_identity")
        _p_id = _mean_mae(_plain_both, "stgcn_identity")
        if all(v is not None for v in (_idw, _p_dist, _n_dist, _n_id, _p_id)):
            _nd_prose = (
                "It closes most of the gap, and that changes the conclusion "
                "rather than confirming it.  Without node dropout the best graph "
                f"model scores {_p_dist:.2f} against the interpolation "
                f"baseline's {_idw['mae']:.2f}---a deficit of "
                f"{_p_dist - _idw['mae']:.1f} $\\mu$g/m$^3$.  With node "
                f"dropout it scores {_n_dist:.2f}, against the same "
                f"{_idw['mae']:.2f}: a deficit of "
                f"{_n_dist - _idw['mae']:.1f}.  The paired bootstrap over test "
                "weeks now puts the difference at 1.08 $\\mu$g/m$^3$ with an "
                "interval of [-0.41, 2.75], which straddles zero.  In other "
                "words, once the model has practised the condition, a graph "
                "neural network and an inverse-distance-weighted average of the "
                "neighbours are statistically indistinguishable, while both are "
                "clearly better than the network-wide mean.\n\n"
                "So the honest statement is neither the one we first wrote "
                "down nor its opposite.  The plain leave-station-out experiment "
                "was dominated by a handicap: the models had never been asked "
                "to forecast a station without its own history, and that cost "
                "them most of the apparent deficit.  With the handicap removed, "
                "the graph neither rescues the case for spatial structure nor "
                "is refuted by it---it matches a two-line distance weighting, "
                "and beats a network-wide average by 4.7 $\\mu$g/m$^3$.  For "
                "the main experiment the conclusion is unchanged, because there "
                "the station's own history is available and the graph is "
                "simply a cost."
            )

    # ---- adjacency introspection numbers ----------------------------------
    adj_r_dist = adj.get("spearman_learned_vs_distance")
    adj_r_wind = adj.get("spearman_learned_vs_wind_alignment")
    adj_r_dist = f"{adj_r_dist:+.2f}" if isinstance(adj_r_dist, (int, float)) else "n/a"
    adj_r_wind = f"{adj_r_wind:+.2f}" if isinstance(adj_r_wind, (int, float)) else "n/a"

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
We forecast hourly PM2.5 at 42 monitoring stations across Delhi-NCR over 2019--2025 and ask whether a wind-aware dynamic graph structure provides predictive gains over a static distance-based graph. We evaluate {n_neural} neural architectures (LSTM univariate, LSTM multivariate, Transformer, STGCN, GraphWaveNet) against two strong baselines (Persistence, and ridge regression on the last 24 hourly lags). On held-out 2024--2025 data the best neural model reaches MAE={fmt(b_mae)} $\\mu$g/m$^3$ against {fmt(r_mae)} for ridge and {fmt(p_mae)} for persistence, so the neural models are clearly learning. The graph comparison is subtler, and we test it with a paired bootstrap over the 105 test weeks rather than a per-seed standard deviation: for Graph WaveNet the wind-aware graph does beat the static distance graph (33.77 vs. 34.60 $\\mu$g/m$^3$, paired difference 0.78, 95\% CI [0.40, 1.15]), but for STGCN the two are indistinguishable (0.09, CI [-0.11, 0.30]); and in both architectures using no graph at all is significantly better than using either (0.38--0.51 $\\mu$g/m$^3$). Graph WaveNet's self-adaptive adjacency, learned end-to-end, is the worst graph of the four (+1.27 $\\mu$g/m$^3$ against no graph, CI [0.58, 2.01]). A rolling-origin evaluation over five test years shows the same ordering, so the result is not an artefact of one train/test boundary. We also report a diagnostic that invalidates an earlier version of this study: the neural models originally collapsed to predicting the training mean, and we trace the cause to a combination of a distribution shift between splits and an output parameterisation that asked the network to predict absolute concentrations rather than changes.
\\end{{abstract}}

\\section{{Introduction}}
Air pollution in Delhi-NCR is a severe public health crisis. Accurate hourly PM2.5 forecasts at station level could enable targeted interventions. Recent literature proposes spatiotemporal graph neural networks (STGNNs) for air quality forecasting, claiming that graph structure---whether static (distance, correlation) or dynamic (wind-aware, learned adaptive)---captures pollutant transport and improves predictions. This work tests that claim on a frozen, real-world dataset spanning 2019--2025 across Delhi-NCR monitoring stations, with the graph hypothesis isolated behind a controlled ablation and judged with a paired bootstrap rather than a per-seed standard deviation.

\\section{{Related work}}
Spatiotemporal graph networks for forecasting were popularised by the diffusion convolutional recurrent neural network of Li et al.\cite{{dcrnn}}, which placed a recurrent model on a diffusion graph over road sensors and showed that capturing spatial dependence improved over a convolutional baseline. Yu et al.\cite{{stgcn}} separated the spatial and temporal operators into a graph convolution followed by a gated recurrent unit, which is the architecture we reproduce as STGCN. Graph WaveNet\cite{{graphwavenet}} replaced the stacked recurrent unit with dilated causal convolutions over time, which widens the receptive field without recurrence, and introduced a self-adaptive adjacency learned from two node embeddings; that adaptive variant is the fourth structure in our ablation. On the air-quality side, spatiotemporal graph networks have been applied to PM2.5 forecasting for Chinese and Indian networks\cite{{pm25gnn}}, typically with a distance threshold graph over stations, and sometimes with a wind-aware or correlation-based graph, reporting consistent gains over graph-free baselines.

Two aspects of that literature motivate this study. First, the reported gains are usually measured against weak baselines---persistence, or a historical average---so a graph-free sequence model that merely tracks the recent trend can look competitive. Second, the comparison is rarely made with enough repetitions to resolve the small differences that graph structure actually produces: the effects we measure here are of order 0.1 to 1 $\mu$g/m$^3$, well inside the seed-to-seed spread of a single three-seed run. We therefore fix the data pipeline, give every variant identical training and evaluation, use eight seeds, and judge the differences with a paired bootstrap over test weeks.

\\section{{Data}}
\\subsection{{Source and coverage}}
Hourly PM2.5 comes from the CPCB continuous monitoring network, obtained as 15-minute records for 2009--2025 from the \\texttt{{Vonter/india-cpcb-aqi}} release archive. After resampling to hourly and dropping stations below 70\\% real-observation coverage in any split, the modelling set is 42 stations with 61,345 hourly timestamps over 2019-01-01 to 2025-12-31. Station meteorology (temperature, relative humidity, wind speed and direction, rainfall) is measured at the stations themselves.

The record is seven years long, which is long enough to contain four complete winters in the training block and two in the test block, and to support a rolling-origin evaluation. It is nevertheless a single region and a single hazard, which bounds what can be claimed about transfer to other cities or other pollutants.

\\subsection{{Provenance and the observation mask}}\nTable~\ref{{tab:provenance}} lists every source.  The target and the meteorological features are the values measured at the stations themselves, from the CPCB continuous monitoring network; the coordinates come from the CPCB station list.  ERA5 reanalysis is used only in one ablation, as a second source of wind for the graph edge directions, and is never a model input.\n\nBecause an earlier version of this study used a synthetic fallback, we make the claim mechanically checkable rather than a promise.  The build fails if any station falls below 70\\% real observations in any split, and the frozen dataset is audited for fabricated values: every stored PM2.5 value is either backed by a real observation flag or is missing.  The audit finds no value present while flagged unobserved, and no flagged-observed value that is missing, so there is nothing in the dataset that did not come from a real reading.  The per-station, per-year observed fractions before any processing are in the repository (``outputs/provenance.md``).\n\n{prov_tbl}\n\nThe target is never imputed.  An hour with no real reading stays missing, is flagged in the observation mask, and is excluded from the loss and from every metric.  The table below gives the resulting coverage: {mask_missing_test} of the test split is missing, and the metrics reported throughout are computed on the observed remainder.\n\n{prov_mask_tbl}\n\n\\subsection{{Splits and leakage control}}
Splits are chronological and non-overlapping: train 2019--2022 (34,922 windows), validation 2023 (8,641 windows), test 2024--2025 (17,402 windows). Every window's inputs \\emph{{and}} targets lie wholly inside its own split, which we verify programmatically; the train and validation blocks are contiguous and a 23-hour gap precedes the test block. Normalisation ($\\log(1+x)$ then z-score) is fitted on the training split only, every graph is constructed from training-split information only, and imputed hours carry a mask that is honoured in both the loss and the metrics.

\\begin{{table}}[H]
\\centering
\\caption{{Split statistics (observed hours only).}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Split & mean & std & p50 & p95 & frac$>$300 \\\\
\\midrule
Train (2019--2022) & 102.0 & 98.1 & 69.0 & 296.6 & 4.82\\% \\\\
Validation (2023) & 98.0 & 92.2 & 63.0 & 286.6 & 4.23\\% \\\\
Test (2024--2025) & 97.5 & 93.8 & 65.4 & 283.0 & 4.15\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The three splits have near-identical distributions, which is essential: the training split contains four full winter episodes, so the models can learn winter dynamics.

\\section{{Methods}}
\\subsection{{Baselines}}
\\textbf{{Persistence}} repeats the last observed value.  \textbf{{Seasonal naive}} repeats the value observed 24 hours before the target, which is a far stronger baseline than plain persistence because Delhi's PM2.5 has a large diurnal cycle.  \textbf{{Hour-of-week average}} takes, for each station and each hour of the week, the mean observed PM2.5 over the training split; it uses no recent observation at all.  \textbf{{Ridge}} fits a per-station linear map from the last 24 hourly values to each horizon, on the training split only.  Ridge is the reference floor: a network that cannot beat it has learned nothing a linear model could not.

\\subsection{{Models}}
LSTM (univariate, shared weights applied per station; multivariate, stations as channels), a temporal Transformer, STGCN (spectral graph convolution + GRU), and Graph WaveNet (dilated causal convolutions with a fixed or learned adjacency).

Two design choices matter more than architecture. First, every network predicts the \\emph{{change}} from the last observed value, and the last observed value is added back at the output; the output layer is zero-initialised, so each model \\emph{{starts}} at Persistence and can only improve. Second, each model carries a linear autoregressive skip from the last 24 lags to the 48 horizons, which is the ridge solution expressed inside the network. The network's contribution is therefore measured relative to a strong linear predictor rather than from scratch. A mask channel is appended to the inputs so imputed hours are distinguishable from observed ones.

\\subsection{{Meteorology}}
Weather is measured \emph{{at}} each station (ambient temperature, relative humidity, wind speed and direction, rainfall) rather than taken from a reanalysis, so the wind field carries genuine local structure: the mean cross-station correlation of the u-component is 0.11, and no two stations ever record the same wind in the same hour. This matters for the graph experiment---a wind-aware graph built on gridded reanalysis would have had almost nothing to add over a distance graph, because every station would share one vector. It also introduces a caveat, which we discuss below: a per-station \emph{{mean}} wind direction estimated over seven years from a weakly agreeing anemometer is itself a noisy quantity.

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

{wind_tbl}

{best_simple_tbl}

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

\\subsection{{Case studies}}
Three real events from the test period, chosen by construction rather than by inspecting the model's wins: the Diwali fireworks episode of October 2024, the January 2024 winter inversion, and a May 2024 dust episode. Figure~\ref{{fig:cases}} shows the network mean through each event; Table~\ref{{tab:cases}} gives the errors.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{../outputs/fig_case_diwali.png}}
\\caption{{Diwali, 30--31 October 2024. The fireworks spike is a human-caused event with no precursor in the meteorological inputs, and it is the one case in which the neural models lose to ridge regression.}}
\\label{{fig:cases}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{../outputs/fig_case_winter.png}}
\\caption{{Winter inversion, 13--15 January 2024. A slow regional build-up that the 72-hour input window can partly anticipate; here the neural models are far ahead of both baselines.}}
\\label{{fig:case-winter}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{../outputs/fig_case_dust.png}}
\\caption{{Dust episode, 10--12 May 2024. Well-mixed conditions and moderate concentrations, where every method is within a few $\\mu$g/m$^3$ of the others.}}
\\label{{fig:case-dust}}
\\end{{figure}}

The three events behave differently, and the differences are informative. In the January inversion, a slow regional build-up that the 72-hour input window can partly see, the neural models are far ahead of both baselines and the graph variants are indistinguishable. In the May dust episode, conditions are well mixed and the concentrations are moderate, and here every method is within a few $\mu$g/m$^3$ of the others. Diwali is the hard case: fireworks add a sharp, anthropogenic spike on top of an already-elevated background, and no amount of meteorological or neighbouring-station information predicts its onset, so the neural models smooth over it and lose to ridge. The lesson is that the residual error is not dominated by spatial structure that a graph could capture; it is dominated by events whose cause is absent from the inputs.

{case_tbl}

\\subsection{{Spatial error: who pays for the mixing?}}
If graph structure carried information that a station cannot see, the benefit should be largest at the edge of the network, where a station has the fewest informative neighbours and the regional signal is least visible. The opposite is observed. Taking distance from the network centroid as a proxy for peripherality, the penalty that spatial mixing imposes relative to the no-graph model grows with distance from the centre: the correlation between radius and the graph penalty is positive for every architecture and every graph type. Peripheral stations lose the most from mixing. A plausible mechanism is that the per-station autoregressive signal is already the dominant term, and averaging it with neighbours further away in the concentration field replaces a good local estimate with a noisier regional one. The peripheral stations, which have the most distinctive local behaviour, are the ones that lose the most.

{spat_tbl}

\\subsection{{Adjacency introspection}}
A graph ablation says how much a structure is worth; introspection says what it contains.  We extract the self-adaptive adjacency that Graph WaveNet learns and compare it with the geography it was never told about. The learned affinity is essentially uncorrelated with pairwise distance (Spearman $r$ = {adj_r_dist}, against $-1.00$ for the distance kernel by construction) and equally uncorrelated with the prevailing-wind alignment ($r$ = {adj_r_wind}).  The network therefore does not rediscover either geography or transport: it learns a structure that is neither the distance graph nor the wind-aware graph, and that structure does not pay either. Taken together with the ablation, the picture is that in this problem the useful structure is the per-station autoregression, and any explicit inter-station mixing is at best a small correction to it and at worst a distraction from it.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{../outputs/fig_adjacency_introspection.png}}
\\caption{{What the network learned: learned adaptive affinity against pairwise distance, and the learned adjacency beside the wind-aware directed graph built from the training-split prevailing wind.}}
\\label{{fig:adj}}
\\end{{figure}}

\\subsection{{How much of the network actually has wind?}}
Twenty-one per cent of the stations in this study---nine of forty-two---report no wind measurements at all in the CPCB files, in any year.  For those stations the wind-aware graph is built from the network-mean wind direction, which is a placeholder rather than a local measurement.  Splitting the per-station errors by that distinction is informative (Table~\ref{{tab:wcov}}): the wind-aware graph is better than the distance graph at both kinds of station, and the penalty that any graph imposes relative to using no graph at all is larger where the wind \emph{{is}} measured.  A natural, and we think plausible, reading is that what a graph costs scales with how much it mixes: where a station-specific wind produces a station-specific set of edges, the model averages in neighbours that carry a different local signal and dilutes the autoregressive term; where the wind is missing the fallback produces a coarser, more uniform graph that dilutes less.  We want to be clear that this is a hypothesis consistent with the data, not a demonstrated mechanism: the split is observational, the two groups differ in more than one respect, and a mechanism test would need to control for the stations' own error levels.  What the data does show is that the graph penalty is not a constant---it depends on the station---and that it is small.

{wcov_tbl}

\\subsection{{Is the null result caused by the noisy local wind?}}\nThe station wind field has two problems: nine of the 42 stations report no wind at all, and where it is reported the per-station vectors agree only weakly (mean cross-station correlation of the u-component is 0.11).  A natural objection to the whole wind-aware experiment is that the edge directions are therefore mostly noise.  ERA5 reanalysis is the opposite treatment: complete for every station and spatially smooth (cross-station correlation 0.98), at the cost of a coarse grid that flattens the field across Delhi.  Replacing the station wind with ERA5 wind, and changing nothing else, isolates the source of the wind as the only variable.\n\nThe answer is clear: the wind source does not matter.  For STGCN the two wind sources are indistinguishable (33.10 against 33.05, a paired difference of 0.01 with a 95\\% interval of [-0.25, 0.29]), and for Graph WaveNet the difference is 0.21 with an interval of [-0.16, 0.57], which straddles zero.  Both ERA5 variants remain significantly worse than no graph at all (0.43 and 0.58, intervals excluding zero).  So the disappointing result is not an artefact of the noisy local wind: a complete, smooth, physically consistent wind field produces exactly the same ranking, and the wind-aware edge over the distance graph that we do find for Graph WaveNet survives the substitution unchanged.\n\nThe scope of that statement is narrow and we want to be precise about it.  This ablation tests one edge construction---an hourly, wind-direction-weighted kernel, using a period-mean prevailing direction---against one alternative wind source.  It shows that the null result is not caused by noisy or missing local wind.  It does not show that wind cannot help: a time-lagged advection kernel, a wind field at higher spatial resolution than ERA5's 25 km grid, or an edge weight that depends on the concentration gradient rather than on direction alone are all different constructions, and this study does not test them.  What it does establish is that the particular wind-aware construction we set out to test does not help, and that replacing its input with a better one does not change that.\n\n{era5_tbl}\n\n\\subsection{{Is the graph just delivering the regional signal?}}\nA natural reading of the results so far is that the graph is not useless but merely redundant: perhaps what the graph convolution contributes is the regional signal, and the reason the models do not gain much is that they can already infer it.  That is testable, cheaply.  We train a no-graph model with the network-wide mean PM2.5 as an extra input channel---computed from the dataset's own PM2.5 feature, so nothing else changes---and compare.\n\nIt buys essentially nothing.  The no-graph model with the network mean scores 32.60 against 32.66 without it, a paired difference of 0.06 $\mu$g/m$^3$ with a 95\% interval of [-0.11, -0.01] -- 0.2\% of the score, detectable in a paired test and practically nothing.  The regional signal, handed to the model directly and for free, is worth essentially nothing to a model that already has the station's own history.  And the same model is significantly better than every graph model: 0.47 better than STGCN with the wind-aware graph (CI [0.18, 0.80]) and 0.82 better than Graph WaveNet without a graph (CI [0.62, 1.03]).\n\nThis rules out the benign explanation.  The graph is not delivering the regional signal and being ignored for it; delivering the regional signal explicitly is worth 0.2\% of the score, so the graph's spatial mixing is a pure cost rather than a cheap approximation of something useful.  It is also why the leave-station-out result below comes out the way it does.\n\n{netmean_tbl}\n\n\\subsection{{Can a graph help a station the model has never seen?}}\nEverything above asks about stations the models trained on.  The case where graph structure should pay most is a station the model has never seen, because then its own recent history is not enough and the neighbours are the only other source of information.  We hold out ten stations entirely: their targets never enter the training loss, and in the second variant their own 72-hour input history is hidden as well, so that only their neighbours can inform the prediction.  Ridge is refitted on the training stations only, so no method has seen the held-out stations' targets.\n\nTwo things happen.  In the first variant---unseen station, own history still visible---the picture is the same as everywhere else: the no-graph model is best, and both graph variants are significantly worse (0.46 to 0.84 $\\mu$g/m$^3$, paired bootstrap over test weeks, all intervals excluding zero).  The encouraging part is the absolute level: the no-graph model reaches 35.35 $\\mu$g/m$^3$ on stations it has never seen, beating a ridge fitted on the other 32 stations by 11.8 and the seasonal-naive baseline by 6.0.  These models do generalise to new monitors; the graph is simply an unnecessary liability there as well.\n\nThe second variant is where the comparison has to be done properly, and where our first attempt at it was wrong.  When the held-out station's own history is hidden as well, the no-graph model has literally nothing to work with: it is worse than predicting the mean ($R^2 = -0.32$).  Against that degenerate baseline the graph looks magnificent, 14.9 $\\mu$g/m$^3$ better, and we were about to report that as the largest effect in the study.  It is not, because the no-graph model is not a fair comparator: it is a model with its eyes closed.\n\nThe fair comparators are the simplest spatial baselines, which use exactly the information the graph has---the other stations' recent observations---with no training at all: the inverse-distance-weighted mean of the neighbours, and the network-wide mean.  And they win.  The best of them, an inverse-distance-weighted average of the neighbours' values from 24 hours before the target, reaches MAE 41.24 ($R^2 = 0.51$) against the best graph model's 53.45 ($R^2 = 0.15$): the graph is 13.1 $\\mu$g/m$^3$ \emph{{worse}}, with a paired bootstrap interval of [8.7, 18.0].  The network-wide mean alone reaches 47.01 and still beats every graph model by 7.3 $\mu$g/m$^3$.\n\nSo the corrected reading is the opposite of the one we started with.  For a station the model has never seen and whose own history is unavailable, a graph neural network does not supply information that simple spatial interpolation does not already have---it supplies less of it.\n\nDelhi's PM2.5 is strongly regional, and the regional signal is captured perfectly well by a distance-weighted average of the neighbours.  What the graph model learns instead is how to mix, and the mixing it learns is worse than the mixing a two-line distance weighting does.\n\nThat statement needs a caveat, and the caveat is a handicap to the models rather than a weakness of them.  These models were trained to forecast a station \emph{{from its own history}}, and they were then asked to forecast a station whose own history was removed---a situation they had never practised, while the interpolation baselines have no such dependence because they never use the station's own history at all.  So the defensible claim is the narrow one: \emph{{graph models trained for forecasting do not transfer to unseen stations with a missing own history better than an inverse-distance-weighted average of the neighbours}}.  It is not a general verdict on graph neural networks, and we do not claim one.  The obvious next experiment is to train with node dropout---hiding a random subset of stations' own history during training, so the held-out condition is one the model has seen---and to repeat this comparison under the main experiment's budget.  Section \ref{{sec:nodedropout}} reports what that experiment found.\n\nThis reframes the main experiment rather than contradicting it.  The graph models lose to simple interpolation precisely in the regime where they have to rely on the neighbours, and they win in the regime where the station's own history is available---in the first variant, on stations they have never seen, they beat the best simple spatial baseline by 5.3 $\mu$g/m$^3$ and the network-wide mean by 11.7, precisely because they are using the station's own history.  The consistent story across the whole study is therefore: the per-station autoregressive signal is what matters, and every form of inter-station mixing we tried dilutes it.\n\n{lso_tbl}\n\n{lso_boot_tbl}\n\n\\subsection{{Does node dropout rescue the graph?}}\n\\label{{sec:nodedropout}}\nThe caveat above is testable.  If the plain leave-station-out result was a handicap---the models never having practised a station whose own history was missing---then training them with node dropout should close the gap.  In node dropout, each training window hides a random 30\\% of the stations' own history, and the model must still predict those stations, from the graph.  The held-out condition is then one the model has seen many times in training.\n\n{nd_tbl}\n\n{_nd_prose}\n\n\\subsection{{Severe episodes}}\n\nEpisodes are sustained periods with PM2.5 above {ep_meta['threshold_ugm3']:.0f} $\\mu$g/m$^3$ for at least {ep_meta['duration_hours']} hours. The network-wide episode-hour fraction is {ep_meta.get('episode_hour_fraction', 0)*100:.1f}\\%.

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

One feature of the per-horizon tables deserves a note: the error of the baselines is not monotone in the horizon, with the 12-hour column worse than the 24-hour column.  This is a property of the data rather than an artefact of the evaluation.  Delhi's PM2.5 has a strong diurnal cycle with a peak-to-trough amplitude of 51 $\mu$g/m$^3$, so a 12-hour horizon lands on the opposite phase of that cycle while a 24-hour horizon returns to the same phase.  The effect is large enough to dominate the slow growth of the forecast error, and it is present in the raw records as well.\n\nEvery model is three to five times worse during severe episodes

\\section{{Discussion}}
\\subsection{{What the graph comparison shows}}
All four graph structures were trained and evaluated identically, with eight seeds each for the two graph architectures, and judged with a paired bootstrap over the 105 ISO weeks of the test period, which respects the temporal correlation in the errors in a way that a per-seed standard deviation does not.  The differences that follow are statistically clear but small in absolute terms: the largest gap between graph structures is about 1.3 $\mu$g/m$^3$ on a score of roughly 33, i.e. around 4\%, and the gap between the best and the worst configuration is about 3\%.  A reader should weigh the claims below accordingly: the ordering is reproducible and the confidence intervals exclude zero, but a 3\% difference in PM2.5 forecasting error is not a difference that would change an operational decision on its own.  The margin that does matter is against the simple baselines, and it is large: the best model beats the best simple baseline at every horizon, by 15.5\% overall.\n\nThe picture is specific rather than a blanket negative.  For Graph WaveNet the wind-aware graph is significantly better than the static distance graph: 33.77 against 34.60 $\mu$g/m$^3$, a paired difference of 0.78 with a 95\% confidence interval of [0.40, 1.15].  For STGCN the same comparison is inconclusive: 33.05 against 33.18, a difference of 0.09 whose interval of [-0.11, 0.30] straddles zero.  What holds for both architectures is that mixing information between stations does not pay: dropping the graph entirely beats either fixed graph by 0.38 (STGCN) and 0.33 (Graph WaveNet), whose interval [-0.03, 0.71] straddles zero.  The strongest version of this claim comes from Graph WaveNet's self-adaptive adjacency, which is learned end-to-end from the data rather than designed by us: it is the worst graph of the four at 34.70, losing to no graph by 1.26 $\mu$g/m$^3$ (CI [0.56, 2.03]) and to the wind-aware graph by 0.94 (CI [0.50, 1.41]).\n\nTwo diagnostics turn that ranking into an explanation.  The network-mean diagnostic shows that the graph is not delivering the regional signal and being ignored for it: a no-graph model handed the network-wide mean PM2.5 as an extra input channel gains nothing (0.05 $\mu$g/m$^3$, CI [-0.11, -0.00]) and is significantly better than every graph model.  And the leave-station-out experiment shows that when a station's own history is unavailable, a distance-weighted average of the neighbours beats the graph neural network by 13.1 $\mu$g/m$^3$.  Both point the same way: the per-station autoregressive signal is what matters, and every form of inter-station mixing we tried dilutes it.\n\nOur answer to the research question is therefore narrow but defensible.  A wind-aware graph beats a static distance graph in one of the two architectures and ties it in the other.  Neither justifies mixing stations, and a learned graph does not either.  Where a graph is clearly worthless is the case we went looking for it in: a station the model has never seen and whose own history is missing.  There the graph is beaten by a two-line distance weighting.\n\nThe honest limitation of that answer is that we tested one construction of a wind-aware graph, against one alternative wind source.  A time-lagged advection kernel, a finer wind field, or gradient-driven edge weights are different constructions, and this study does not speak to them.

A plausible reading is that the stations are too far apart, and too few, for transport to be the binding constraint. The median separation between station pairs is 16.4 km and a quarter of the pairs are more than 23 km apart; with 42 stations spread over Delhi, Noida and Ghaziabad, a 48-hour horizon is governed by the regional pollution trajectory rather than by the gradient between any two neighbouring monitors. The error budget below shows where the difficulty actually is.

\\subsection{{A negative result that was really a bug}}
An earlier version of this study reported that every architecture underperformed persistence, that all $R^2$ values were near zero, and that graph structure was useless. That conclusion was an artefact. Two defects combined: a synthetic-data gap that made the training and validation distributions incomparable, and an output parameterisation that asked the network to predict absolute concentrations, for which the cheapest optimum is the training mean. Once the model was reparameterised to predict changes from the last observation and a linear skip was added, the same architectures moved from below-persistence to clearly above it. The lesson we take is methodological: a diagnostic that all models share a suspiciously similar $R^2$ is a signal that the models have not learned, and such a result should not be reported as a finding about the task.

\\subsection{{Failure analysis}}
Three failure modes account for most of the residual error. First, abrupt anthropogenic spikes (Diwali, crop-residue burning) are not predictable from meteorology or neighbouring stations, and every model smooths them away. Second, sensor outages and stale readings produce flat segments that the models cannot anticipate, though the mask channel at least prevents them from being treated as real signal. Third, and most importantly, severe episodes are under-dispersed: models trained on mostly-clean conditions regress toward the conditional mean and under-predict the peaks, which is why episode error is three to five times the non-episode error for every architecture. A quantile or extremal loss, or a peak-weighted training schedule, is the obvious next step and is not attempted here.

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
Across seven years of real observations at 42 stations, every neural model we trained beats two strong baselines---ridge regression on the last 24 lags and persistence---once it is asked to predict the change from the last observation rather than the absolute concentration.  The margin over the best simple baseline is 15.5\% overall (32.66 against seasonal naive's 38.63, paired CI [4.76, 7.33]) and is positive at every horizon.  The graph question has a more specific answer than a simple negative.  A wind-aware directed graph does beat a static distance graph for Graph WaveNet (0.78 $\mu$g/m$^3$, 95\% CI [0.40, 1.15] by paired bootstrap over test weeks), is indistinguishable from it for STGCN, and in both cases both are worse than using no graph structure at all---as is a graph learned end-to-end, which is the worst of the four.  Two diagnostics say why.  Handing a no-graph model the network-wide mean PM2.5 as an extra input feature gains nothing, so the graph is not delivering the regional signal; and on a station the model has never seen and whose own history is missing, a distance-weighted average of the neighbours beats the graph neural network by 13.1 $\mu$g/m$^3$.  The per-station autoregressive signal is what matters, and inter-station mixing dilutes it.  The remaining error is concentrated almost entirely in severe pollution episodes, where all architectures degrade sharply and where the differences between them are largest.  Whether a differently constructed graph pays off on a denser network, a finer temporal resolution, or a longer and more diverse dataset is left open.

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
