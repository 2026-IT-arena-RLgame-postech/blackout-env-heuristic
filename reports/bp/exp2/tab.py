"""usage: tab.py dir [dir..]  -- per-candidate margin table with names (cands sets + exploit registry)"""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "reports/bp/exp2")
from cands import SETS
from blackout_env.train.distill.exploit_bp import EXPLOITERS
from blackout_env.train.distill.ood_bp import OODBP
names = {d: n for n, d, _ in EXPLOITERS}
names.update({d: "ood:" + n for n, d, _, _ in OODBP})
names["B6"] = "B6"
for s in SETS.values():
    names.update({d: n for n, d in s.items()})
d = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f + "/matches.jsonl"):
        r = json.loads(l); d[r["ours"]].append(r)
rows = []
for o, v in d.items():
    m = np.array([r["margin"] for r in v]); sd = np.array([r["side"] for r in v])
    rows.append((m.mean(), names.get(o, o[:60]), len(m), m.std(ddof=1) / len(m) ** .5, (m > 0).mean(), m[sd == 0].mean(), m[sd == 1].mean()))
for mm, o, n, se, w, a, b in sorted(rows, reverse=True):
    print(f"{mm:+6.1f}±{se:4.1f} n={n:3d} win {w:.2f} A{a:+5.0f} B{b:+5.0f}  {o}")
