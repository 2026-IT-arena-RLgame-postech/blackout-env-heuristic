"""R4 (docs §54) per-opponent table: paired differences (same opponent, seed, side; SE over seeds) and margins
for the P21e / P20 search arms in reports/r4_search/sim_*.  Run from the repo root: python reports/r4_search/scripts/table.py
"""
import json, numpy as np
from pathlib import Path
rows = {}
for d in Path("reports/r4_search").glob("sim_*"):
    for f in d.glob("matches_*.jsonl"):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x); rows.setdefault((d.name[4:] + ":" + r["tag"], r["opponent"]), {})[(r["seed"], r["side"])] = r
def paired(a, b):
    ks = sorted(set(a) & set(b))
    by = {}
    for k in ks: by.setdefault(k[0], []).append(a[k]["margin"] - b[k]["margin"])
    v = np.array([np.mean(x) for x in by.values()])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)
opps = ["v31", "v30", "p21e", "p20", "p17", "v28", "v27", "v25", "p14b"]
specs = [("p21e:o4f_r21", "p21e:plain"), ("p21e:o4f_r21", "p20:o4f_r20"), ("p21e:o4f_r21", "p21e:o4f_r20"), ("p21e:o4f_r20", "p21e:plain"), ("p21e:plain", "p20:plain")]
print("| opp | " + " | ".join(f"{a}-{b}" for a, b in specs) + " | margins o4f_r21 / plain21 / p20o4f_r20 / p20plain |")
pooled = {s: [] for s in specs}
for o in opps:
    cells = []
    for s in specs:
        a, b = rows.get((s[0], o)), rows.get((s[1], o))
        if a and b:
            m, se, n = paired(a, b); cells.append(f"{m:+.1f}±{se:.1f} ({n})"); pooled[s].append((m, se))
        else: cells.append("-")
    mg = " / ".join(f"{np.mean([r['margin'] for r in rows[(t, o)].values()]):+.1f}" if (t, o) in rows else "-" for t in ["p21e:o4f_r21", "p21e:plain", "p20:o4f_r20", "p20:plain"])
    print(f"| {o} | " + " | ".join(cells) + f" | {mg} |")
print("| pooled | " + " | ".join(f"{np.mean([m for m, _ in v]):+.1f}±{np.sqrt(sum(se**2 for _, se in v))/len(v):.1f} ({len(v)})" for v in pooled.values()) + " |")
for t in ["p21e:o4f_r21", "p21e:plain", "p20:o4f_r20", "p20:plain"]:
    d = rows[(t, "v31")]; m = [r["margin"] for r in d.values()]
    print(t, "v31 n", len(d) // 2, "win", round(np.mean([1 if x > 0 else .5 if x == 0 else 0 for x in m]), 3))
