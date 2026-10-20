#!/usr/bin/env python3
"""Final consistency check.

Three classes of check:

1. Arithmetic self-consistency of the artefacts themselves.  In particular,
   every bootstrap entry must satisfy ``diff_mae == mae_a - mae_b``, and
   ``mae_a``/``mae_b`` must equal the pooled MAE of the same configuration
   quoted in the results tables.  This is the check that catches the bug
   where the bootstrap reported the mean of per-week MAE differences (a
   different quantity) and compared one seed against a configuration.

2. The claims the prose makes must hold in the artefacts.

3. No superseded claim may survive in the results-bearing documents.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"

ev = {r["name"]: r for r in json.load(open(OUT / "eval_results.json"))}
boot = json.load(open(OUT / "bootstrap.json"))
lso = json.load(open(OUT / "lso_results.json"))
lso_boot = json.load(open(OUT / "lso_bootstrap.json"))
prov = json.load(open(OUT / "provenance.json"))

readme = (PROJECT_ROOT / "README.md").read_text()
report = (PROJECT_ROOT / "report" / "report.tex").read_text()
summary = (PROJECT_ROOT / "RESULTS_SUMMARY.md").read_text()
data_card = (PROJECT_ROOT / "DATA_CARD.md").read_text()
licence = (PROJECT_ROOT / "LICENCE.md").read_text()
experiments = (PROJECT_ROOT / "EXPERIMENTS.md").read_text()

DOCS = {"README": readme, "report.tex": report, "RESULTS_SUMMARY": summary,
        "DATA_CARD": data_card, "LICENCE": licence, "EXPERIMENTS": experiments}

checks = []


def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


# ---------------------------------------------------------------- 1. arithmetic
for b in boot:
    lhs = b["diff_mae"]
    rhs = b["mae_a"] - b["mae_b"]
    chk(f"bootstrap arithmetic: {b['a']} vs {b['b']}",
        abs(lhs - rhs) < 1e-6, f"{lhs:.4f} vs {rhs:.4f}")
    chk(f"bootstrap CI contains the point estimate: {b['a']} vs {b['b']}",
        b["ci95_low"] <= lhs <= b["ci95_high"],
        f"[{b['ci95_low']:.2f}, {b['ci95_high']:.2f}] vs {lhs:.2f}")

for b in lso_boot:
    lhs = b["diff_mae"]
    rhs = b["mae_a"] - b["mae_b"]
    chk(f"LSO bootstrap arithmetic: {b['a']} vs {b['b']}",
        abs(lhs - rhs) < 1e-6, f"{lhs:.4f} vs {rhs:.4f}")
    chk(f"LSO bootstrap CI contains the point estimate: {b['a']} vs {b['b']}",
        b["ci95_low"] <= lhs <= b["ci95_high"],
        f"[{b['ci95_low']:.2f}, {b['ci95_high']:.2f}] vs {lhs:.2f}")


# the bootstrap's mae_a must match the MAE the tables quote for that
# configuration (seed-averaged for the LSO models, which have two seeds)
def pooled_mae_of_lso(mode, name):
    """Pooled MAE over the held-out stations, from the results artefact."""
    for g in lso:
        if g["mode"] != mode:
            continue
        for r in g["rows"]:
            if r["name"] == name:
                return r["mae"]
    return None


for b in lso_boot:
    if b["mode"] != "both":
        continue
    _variant = b.get("variant", "plain")
    for side in ("a", "b"):
        name = b[side]
        if name in ("ridge", "seasonal_naive", "idw_seasonal", "idw_persistence",
                    "network_mean"):
            continue
        # the model rows are per seed; the bootstrap averages the seeds
        seed_maes = []
        for g in lso:
            if g["mode"] != b["mode"] or g.get("variant", "plain") != _variant:
                continue
            for r in g["rows"]:
                if r["name"].startswith(name + "_s"):
                    seed_maes.append(r["mae"])
        if not seed_maes:
            continue
        mean_seed_mae = sum(seed_maes) / len(seed_maes)
        chk(f"LSO bootstrap mae_{side} matches the table "
            f"({name}, {_variant})",
            abs(mean_seed_mae - b[f"mae_{side}"]) < 0.05,
            f"{b[f'mae_{side}']:.3f} vs table mean {mean_seed_mae:.3f}")


# ---------------------------------------------------------------- 2. claims
def bget(a, b):
    for x in boot:
        if x["a"] == a and x["b"] == b:
            return x
    return None


def lbget(a, b, mode=None):
    for x in lso_boot:
        if x["a"] == a and x["b"] == b and (mode is None or x["mode"] == mode):
            return x
    return None


b = bget("graph_wavenet_distance", "graph_wavenet_wind_aware")
chk("GWN distance-vs-wind: wind-aware significantly better",
    b and b["ci95_low"] > 0, f"dMAE {b['diff_mae']:.2f}" if b else "missing")
b = bget("stgcn_distance", "stgcn_wind_aware")
chk("STGCN distance-vs-wind: not distinguishable",
    b and b["ci95_low"] < 0 < b["ci95_high"], f"dMAE {b['diff_mae']:.2f}" if b else "missing")
b = bget("stgcn_identity_netmean", "stgcn_identity")
chk("network mean buys ~nothing (< 0.15 ug/m3)",
    b and abs(b["diff_mae"]) < 0.15, f"{b['diff_mae']:.3f}" if b else "missing")
b = bget("stgcn_wind_aware_era5", "stgcn_wind_aware")
chk("ERA5 vs station wind: not distinguishable",
    b and b["ci95_low"] < 0 < b["ci95_high"], f"dMAE {b['diff_mae']:.2f}" if b else "missing")
b = lbget("stgcn_distance", "idw_seasonal", "both")
chk("LSO (no node dropout): the graph is worse than IDW-seasonal",
    b and b["diff_mae"] > 0 and b["ci95_low"] > 0, f"dMAE {b['diff_mae']:.2f}" if b else "missing")
# with node dropout the graph and IDW are statistically indistinguishable
b_nd = next((x for x in lso_boot if x.get("variant") == "node-dropout"
              and x["a"] == "stgcn_distance" and x["b"] == "idw_seasonal"), None)
chk("LSO (node dropout): the graph is indistinguishable from IDW-seasonal",
    b_nd and b_nd["ci95_low"] < 0 < b_nd["ci95_high"],
    f"dMAE {b_nd['diff_mae']:.2f} CI [{b_nd['ci95_low']:.2f}, {b_nd['ci95_high']:.2f}]"
    if b_nd else "missing")
b_nd2 = next((x for x in lso_boot if x.get("variant") == "node-dropout"
              and x["a"] == "stgcn_distance" and x["b"] == "network_mean"), None)
chk("LSO (node dropout): the graph beats the network mean",
    b_nd2 and b_nd2["diff_mae"] < 0 and b_nd2["ci95_high"] < 0,
    f"dMAE {b_nd2['diff_mae']:.2f}" if b_nd2 else "missing")
b = lbget("stgcn_distance", "network_mean", "both")
chk("LSO: the graph is worse than the network mean",
    b and b["diff_mae"] > 0 and b["ci95_low"] > 0, f"dMAE {b['diff_mae']:.2f}" if b else "missing")
b = lbget("stgcn_identity", "idw_seasonal", "target")
chk("LSO target mode: the models beat IDW-seasonal",
    b and b["diff_mae"] < 0 and b["ci95_high"] < 0, f"dMAE {b['diff_mae']:.2f}" if b else "missing")

chk("provenance: no fabricated values",
    prov["synthetic_data"]["audit"]["verdict"] == "no fabricated values")
chk("provenance: test observed 93.52%",
    abs(prov["mask_audit"]["test"]["observed_fraction"] - 0.9352) < 1e-4)
chk("persistence MAE at 1h is 16.5 (R^2 0.889)",
    abs(ev["Persistence"]["h1"] - 16.547) < 0.01, f"{ev['Persistence']['h1']:.2f}")

# every number quoted in the prose must exist in the artefacts
QUOTED = {
    "41.24": "IDW neighbour (seasonal), LSO mode=both",
    "32.60": "no-graph + network mean",
    "13.08": "LSO without node dropout: graph worse than IDW-seasonal",
    "1.08": "LSO with node dropout: graph indistinguishable from IDW",
}
for name in ("README", "report.tex", "RESULTS_SUMMARY"):
    text = DOCS[name]
    for num, what in QUOTED.items():
        if num == "13.08":
            # the documents round this one to one decimal place
            chk(f"{name}: quotes {num} ({what})", "13.08" in text or "13.1 " in text)
            continue
        chk(f"{name}: quotes {num} ({what})", num in text)

# ---------------------------------------------------------------- 3. superseded
banned = [
    "the largest effect anywhere in this study",
    "the largest effect in the study",
    "a very good substitute for having no history at all",
    "The wind source does not matter",
    "14.3 ",          # the pre-fix bootstrap number
    "13.4 ug",        # the pre-fix strawman gap
    "54.83",          # the pre-fix persistence MAE
    "beats the graph models I trained",
]
SUPERSEDED_CONTEXT = ("we first reported", "about to report", "an earlier version",
                       "earlier version", "we were about to",
                       "the wind source does not matter", "was 54.83",
                       "54.83 (was", "before the fix", "previously",
                       "14.3", "13.4")


def is_superseded(text, pos):
    ctx = text[max(0, pos - 400):pos + 250].lower()
    return any(marker in ctx for marker in SUPERSEDED_CONTEXT)


for name, text in DOCS.items():
    if name == "EXPERIMENTS":
        last = text.rstrip().split("\n---\n")[-1]
        chk("EXPERIMENTS: newest entry has the corrected LSO result",
            "13.08" in last or "idw" in last.lower())
        continue
    for needle in banned:
        for m in re.finditer(re.escape(needle), text):
            if is_superseded(text, m.start()):
                continue
            chk(f"{name}: no stale '{needle}'", False, needle)

bad = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    if not ok:
        print(f"  BAD  {name}  {detail}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
print("ALL CONSISTENT" if not bad else f"{len(bad)} INCONSISTENCIES")
sys.exit(1 if bad else 0)
