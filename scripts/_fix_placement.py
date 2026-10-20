#!/usr/bin/env python3
"""Fix two placement errors in the report:

1. the node-dropout prose placeholder is named nd_prose, not _nd_prose;
2. the leave-station-out caveat was inserted in the middle of a paragraph,
   orphaning its last two sentences.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
p = PROJECT_ROOT / "scripts" / "make_report.py"
s = p.read_text()
NL = chr(92) + "n"

# ---- 1. the placeholder name -------------------------------------------
old = "{nd_prose}" + NL + NL
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "{_nd_prose}" + NL + NL, 1)
print("  fixed: nd_prose placeholder")

# ---- 2. move the caveat to the end of the paragraph --------------------
caveat_start = "That statement needs a caveat, and the caveat is a handicap to the "
i = s.index(caveat_start)
j = s.index("Section " + chr(92) + chr(92) + "ref{{sec:nodedropout}} reports what that experiment found.", i)
j = s.index("\n", j)
caveat = s[i:j + 1]
# remove it from where it is
s = s[:i] + s[j + 1:]
# re-insert it after the paragraph it belongs to
anchor = ("and the mixing it learns is worse than the mixing a two-line distance "
          "weighting does.")
k = s.index(anchor) + len(anchor)
s = s[:k] + NL + NL + caveat + s[k:]
print("  fixed: caveat placement")

p.write_text(s)
print("patched")
