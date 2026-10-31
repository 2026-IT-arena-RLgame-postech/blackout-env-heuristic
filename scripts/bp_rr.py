"""Round robin: python scripts/bp_rr.py NAME 'a;b;c' [seeds]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bp_loop as L
from blackout_env.heuristics.params import resolve_descriptor
name, names = sys.argv[1], sys.argv[2].split(";")
seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 48
d = {n: resolve_descriptor(n) for n in names}
rows = L.play(name, list(d.values()), list(d.values()), seeds, 70_000_000)
t = L.table(rows)
out = ["row-vs-col mean margin (+- se)"]
for a in names:
    cells = []
    for b in names:
        v = np.array(list(t[d[a]][d[b]].values())) if d[b] in t[d[a]] else np.array([np.nan])
        cells.append(f"{v.mean():+6.1f}±{v.std(ddof=1)/np.sqrt(len(v)):.1f}" if a != b else "   --    ")
    out.append(f"{a:6s} " + " ".join(cells) + f"  avg {np.nanmean([np.mean(list(t[d[a]][d[b]].values())) for b in names if b != a]):+.1f}")
L.log("\n".join(out))
