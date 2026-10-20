#!/usr/bin/env python3
"""Move the leave-station-out caveat to the end of its paragraph."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
p = PROJECT_ROOT / "scripts" / "make_report.py"
s = p.read_text()
NL = chr(92) + "n"

start_marker = "That statement needs a caveat, and the caveat is a handicap to the "
end_marker = "and the mixing it learns is worse than the mixing a two-line distance "

i = s.index(start_marker)
j = s.index(end_marker, i)
j = s.index("\n", j)
caveat = s[i:j + 1]
print("  caveat length:", len(caveat))

# remove it from where it is
s = s[:i] + s[j + 1:]

# re-insert it after the paragraph it belongs to
k = s.index(end_marker) + len(end_marker)
s = s[:k] + NL + NL + caveat + s[k:]
print("  moved the caveat to the end of its paragraph")

p.write_text(s)
print("patched")
