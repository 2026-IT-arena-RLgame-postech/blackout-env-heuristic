"""paired.py ref_name dir.. : paired (same seed+side) margin difference of every candidate vs the reference candidate"""
import json, sys, numpy as np
sys.path.insert(0, "reports/bp/exp2")
from cands import SETS
names = {}
for s in SETS.values(): names.update({d: n for n, d in s.items()})
ref = sys.argv[1]
M = {}
for f in sys.argv[2:]:
    for l in open(f + "/matches.jsonl"):
        r = json.loads(l); M.setdefault(names.get(r["ours"], r["ours"][:30]), {})[(r["seed"], r["side"])] = r["margin"]
R = M[ref]
for k, v in sorted(M.items()):
    ks = [x for x in v if x in R]
    d = np.array([v[x] - R[x] for x in ks])
    same = np.mean(d == 0)
    print(f"{k:14s} n={len(d)} diff {d.mean():+6.1f}±{d.std(ddof=1)/len(d)**.5:4.1f} identical {same:.2f}")
