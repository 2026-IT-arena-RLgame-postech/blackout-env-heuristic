"""tab6.py dir [dir..]: margin table with candidate names from c6 sets."""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "reports/bp/exp6")
from c6 import SETS
from blackout_env.train.distill.exploit_bp import EXPLOITERS
names = {d: n for n, d, _ in EXPLOITERS}; names["B6"] = "B6"
for s in SETS.values(): names.update({d: n for n, d in s.items()})
d = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f + "/matches.jsonl"):
        r = json.loads(l); d[r["ours"]].append(r)
rows = []
for o, v in d.items():
    m = np.array([r["margin"] for r in v]); sd = np.array([r["side"] for r in v])
    h = np.mean([r["hunter_s"] for r in v]); oh = np.mean([r["opp_hunter_s"] for r in v])
    rows.append((m.mean(), names.get(o, o[:50]), len(m), m.std(ddof=1) / len(m) ** .5, (m > 0).mean(), m[sd == 0].mean(), m[sd == 1].mean(), h, oh))
for mm, o, n, se, w, a, b, h, oh in sorted(rows, reverse=True):
    print(f"{mm:+6.1f}±{se:4.1f} n={n:3d} win {w:.2f} A{a:+5.0f} B{b:+5.0f} Hs {h:5.1f}/{oh:5.1f}  {o}")
