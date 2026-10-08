import json, sys
from collections import defaultdict
import numpy as np

base = "/Users/mac/project/26rl/blackout-env/reports/anyangle_targets"
new_id, old_id = sys.argv[1], sys.argv[2]


def load(ids):
    rows = {}
    for line in (l for i in ids.split(",") for l in open(f"{base}/{i}/matches.jsonl")):
        r = json.loads(line)
        if r.get("student") is not None:
            continue
        rows[(r["opponent"], r["seed"], r["swap"])] = r["learner_score"] - r["opponent_score"]
    return rows


new, old = load(new_id), load(old_id)
by = defaultdict(list)
mn, mo = defaultdict(list), defaultdict(list)
for k, v in new.items():
    mn[k[0]].append(v)
for k, v in old.items():
    mo[k[0]].append(v)
for k in new.keys() & old.keys():
    by[k[0]].append(new[k] - old[k])
allg = []
print(f"{'opponent':26s} {'n':>3s} {'new':>14s} {'old':>14s} {'new-old':>14s}")
se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))
for opp in sorted(by):
    g = np.array(by[opp]); allg += list(g)
    a, b = np.array(mn[opp]), np.array(mo[opp])
    print(f"{opp:26s} {len(g):3d} {a.mean():+7.1f}±{se(a):4.1f}  {b.mean():+7.1f}±{se(b):4.1f}  {g.mean():+7.1f}±{se(g):4.1f}")
allg = np.array(allg)
# opponent-mean of per-opponent gaps, se by pooled
print(f"{'all':26s} {len(allg):3d} {'':14s} {'':14s} {allg.mean():+7.1f}±{se(allg):4.1f}")
