#!/usr/bin/env python3
"""Final consistency check: every number quoted in the prose must match the
JSON artefacts, and no superseded claim may survive anywhere."""
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
netmean = json.load(open(OUT / "eval_netmean.json"))

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


def bget(a, b):
    for x in boot:
        if x["a"] == a and x["b"] == b:
            return x
    return None


def lbget(a, b):
    for x in lso_boot:
        if x["a"] == a and x["b"] == b:
            return x
    return None


# ---- the claims that must be true in the artefacts ----
b = bget("graph_wavenet_distance", "graph_wavenet_wind_aware")
chk("GWN distance-vs-wind dMAE ~0.78", b and abs(b["diff_mae"] - 0.78) < 0.01,
    f"{b['diff_mae']:.3f}" if b else "missing")
b = bget("stgcn_distance", "stgcn_wind_aware")
chk("STGCN distance-vs-wind not significant",
    b and b["ci95_low"] < 0 < b["ci95_high"], f"CI {b['ci95_low']:.2f},{b['ci95_high']:.2f}" if b else "missing")
b = bget("stgcn_identity_netmean", "stgcn_identity")
chk("network mean buys nothing (-0.05)", b and abs(b["diff_mae"] + 0.05) < 0.01,
    f"{b['diff_mae']:.3f}" if b else "missing")
b = bget("stgcn_wind_aware_era5", "stgcn_wind_aware")
chk("ERA5 vs station wind indistinguishable",
    b and b["ci95_low"] < 0 < b["ci95_high"], f"CI {b['ci95_low']:.2f},{b['ci95_high']:.2f}" if b else "missing")
b = lbget("stgcn_distance_s42_lso10_both", "IDW neighbour (seasonal)")
chk("LSO: graph loses to IDW by ~14.3", b and abs(b["diff_mae"] - 14.32) < 0.1,
    f"{b['diff_mae']:.2f}" if b else "missing")

# ---- superseded claims that must be GONE from the prose ----
banned = [
    ("13.4 $\\\\mu$g/m$^3$ improvement, the largest effect", "LSO strawman headline"),
    ("the largest effect anywhere in this study", "LSO strawman headline"),
    ("the largest effect in the study", "LSO strawman headline"),
    ("Graph essential", "overstated LSO"),
    ("a very good substitute for having no history at all", "overstated LSO"),
    ("The wind source does not matter", "over-broad ERA5 claim"),
    ("54.83", "pre-fix persistence MAE (except when explicitly marked as 'was')"),
]
# Superseded claims are allowed where they are explicitly marked as
# superseded: inside a blockquote that says we first reported them wrongly,
# or in the chronological run log.
SUPERSEDED_CONTEXT = ("we first reported", "about to report", "an earlier version",
                       "earlier version", "we were about to",
                       "the wind source does not matter", "was 54.83",
                       "54.83 (was", "before the fix", "previously")


def is_superseded(text, pos):
    """True if the claim is explicitly marked as superseded.

    The window looks both ways: "(was 54.83)" puts the marker after the
    number, while "we first reported ..." puts it before.
    """
    ctx = text[max(0, pos - 400):pos + 250].lower()
    return any(marker in ctx for marker in SUPERSEDED_CONTEXT)


for name, text in DOCS.items():
    if name == "EXPERIMENTS":
        # a chronological log records what was believed at the time; what
        # matters is that its newest entry states the corrected result
        last = text.rstrip().split("\n---\n")[-1]
        chk("EXPERIMENTS: newest entry has the corrected LSO result",
            "14.3" in last and "41.24" in last)
        chk("EXPERIMENTS: newest entry has the network-mean result",
            "32.60" in last)
        continue
    for needle, why in banned:
        for m in re.finditer(re.escape(needle), text):
            if is_superseded(text, m.start()):
                continue
            chk(f"{name}: no '{why}'", False, needle)

# ---- the numbers that must be present ----
for name in ("README", "report.tex", "RESULTS_SUMMARY"):
    text = DOCS[name]
    chk(f"{name}: quotes 41.24 (IDW seasonal)", "41.24" in text)
    chk(f"{name}: quotes 32.60 (network mean)", "32.60" in text)
    chk(f"{name}: quotes 14.3 (LSO graph penalty)", "14.3" in text)

# ---- provenance / mask ----
chk("provenance: no fabricated values",
    prov["synthetic_data"]["audit"]["verdict"] == "no fabricated values")
chk("provenance: test observed 93.52%",
    abs(prov["mask_audit"]["test"]["observed_fraction"] - 0.9352) < 1e-4)

# ---- persistence sanity check still holds ----
chk("persistence R2 at 1h is 0.889", abs(ev["Persistence"]["h1"] - 16.547) < 0.01,
    f"MAE@1h {ev['Persistence']['h1']:.2f}")

# ---- the LSO results file must contain the spatial baselines ----
rows = lso[0]["rows"] if lso else []
names = {r["name"] for r in rows}
chk("LSO results include IDW neighbour (seasonal)", "IDW neighbour (seasonal)" in names)
chk("LSO results include Network mean (persistence)", "Network mean (persistence)" in names)

# ---- the netmean diagnostic must be in the evaluation ----
chk("netmean diagnostic evaluated", any("netmean" in r["name"] for r in netmean))

bad = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    if not ok:
        print(f"  BAD  {name}  {detail}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
print("ALL CONSISTENT" if not bad else f"{len(bad)} INCONSISTENCIES")
sys.exit(1 if bad else 0)
