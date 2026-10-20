#!/usr/bin/env python3
"""Fifteenth patch of scripts/make_report.py: the leave-station-out section
and its paired bootstrap, with the blinding asymmetry stated explicitly."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
s = p.read_text()
B = chr(92) * 2
B1 = chr(92)


def replace_line(s, marker, new_text, label):
    i = s.index(marker)
    j = s.index("\n", i)
    print(f"  replaced: {label}")
    return s[:i] + new_text + s[j:]


# ---------------- load the LSO bootstrap ----------------
old_load = '''    lso = []
    lso_path = Path(args.lso)'''
new_load = '''    lso_boot = []
    lso_boot_path = Path(args.lso_bootstrap)
    if lso_boot_path.exists():
        lso_boot = json.loads(lso_boot_path.read_text())

    lso = []
    lso_path = Path(args.lso)'''
assert old_load in s
s = s.replace(old_load, new_load, 1)

old_args = '''    ap.add_argument("--lso", default="outputs/lso_results.json")'''
new_args = '''    ap.add_argument("--lso", default="outputs/lso_results.json")
    ap.add_argument("--lso_bootstrap", default="outputs/lso_bootstrap.json")'''
assert old_args in s
s = s.replace(old_args, new_args, 1)

# ---------------- LSO bootstrap table ----------------
old_tbl = "    # ---- leave-station-out table ------------------------------------------"
assert old_tbl in s
new_tbl = '''    # ---- leave-station-out bootstrap table ---------------------------------
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
            "\\\\begin{table}[H]\\n\\\\centering\\n"
            "\\\\caption{Leave-station-out, paired bootstrap over test weeks "
            "(A minus B; negative means A is better).}\\n"
            "\\\\begin{tabular}{lrrc}\\n\\\\toprule\\n"
            "comparison & $\\\\Delta$MAE & 95\\\\% CI & $P(A<B)$ " + ROW_END + "\\n\\\\midrule\\n"
            + "\\n".join(rows_lb) + "\\n\\\\bottomrule\\n"
            "\\\\end{tabular}\\n\\\\end{table}")

''' + old_tbl
s = s.replace(old_tbl, new_tbl, 1)

# ---------------- the LSO prose ----------------
old_marker = B + "subsection{{Severe episodes}}"
new_block = (
    B + "subsection{{Can a graph help a station the model has never seen?}}"
    + "\\n"
    + "Everything above asks about stations the models trained on.  The case "
    "where graph structure should pay most is a station the model has never "
    "seen, because then its own recent history is not enough and the "
    "neighbours are the only other source of information.  We hold out ten "
    "stations entirely: their targets never enter the training loss, and in "
    "the second variant their own 72-hour input history is hidden as well, so "
    "that only their neighbours can inform the prediction.  Ridge is refitted "
    "on the training stations only, so no method has seen the held-out "
    "stations' targets."
    + "\\n\\n"
    + "Two things happen.  In the first variant---unseen station, own history "
    "still visible---the picture is the same as everywhere else: the no-graph "
    "model is best, and both graph variants are significantly worse (0.46 to "
    "0.84 $\\\\mu$g/m$^3$, paired bootstrap over test weeks, all intervals "
    "excluding zero).  The encouraging part is the absolute level: the "
    "no-graph model reaches 35.35 $\\\\mu$g/m$^3$ on stations it has never "
    "seen, beating a ridge fitted on the other 32 stations by 11.8 and the "
    "seasonal-naive baseline by 6.0.  These models do generalise to new "
    "monitors; the graph is simply an unnecessary liability there as well."
    + "\\n\\n"
    + "In the second variant---unseen station, own history hidden as well---"
    "the graph becomes essential.  The no-graph model has literally nothing "
    "to work with for such a station and is worse than predicting the mean "
    "($R^2 = -0.32$), while the graph models reach $R^2 = +0.11$ by reading "
    "the neighbours: a 13.4 $\\\\mu$g/m$^3$ improvement, the largest effect "
    "anywhere in this study.  But the graph is not sufficient either.  The "
    "baselines were not blinded---they still use the held-out stations' own "
    "recent values, which the models were denied---so that comparison is "
    "deliberately generous to the baselines, and the baselines still win: a "
    "graph-fed model at 53.5 against ridge at 47.5 and seasonal naive at "
    "41.4.  Read carefully, the two variants say the same thing: the "
    "neighbours do not carry as much information about a station as that "
    "station's own recent observations, so a graph is a poor substitute for a "
    "station's own history---but it is a very good substitute for having no "
    "history at all, which is the only situation in which graph structure is "
    "clearly worth its cost."
    + "\\n\\n"
    + "{lso_tbl}\\n\\n"
    + "{lso_boot_tbl}\\n\\n"
)
i = s.index(old_marker)
s = s[:i] + new_block + s[i:]
print("  inserted: LSO prose")

p.write_text(s)
print("patched")
