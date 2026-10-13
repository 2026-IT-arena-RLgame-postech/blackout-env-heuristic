"""Paired search - plain margins (same seed, side) vs V32.1; SE over seeds (sides averaged)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "reports/official_submission/strength_v321"
arm = {a: {(r["seed"], r["side"]): r for f in glob.glob(f"{d}/{a}_w*.jsonl") for r in map(json.loads, open(f))}
       for a in ("plain", "search")}
for a, rows in arm.items():
    m = np.array([r["margin"] for r in rows.values()])
    print(f"{a:7s} n={len(m):3d} margin {m.mean():+.1f} ±{m.std(ddof=1) / np.sqrt(len(m)):.1f}  win {np.mean(m > 0):.3f} "
          f"loss {np.mean(m < 0):.3f}  tracker fixes {sum(sum(v for k, v in r['tracker'].items() if k.endswith('_fix') or k == 'score_override') for r in rows.values())}")
keys = sorted(set(arm["plain"]) & set(arm["search"]))
by = defaultdict(list)
for k in keys:
    by[k[0]].append(arm["search"][k]["margin"] - arm["plain"][k]["margin"])
ps = np.array([np.mean(v) for v in by.values()])
if len(ps) > 1:
    print(f"paired search - plain: {ps.mean():+.1f} ±{ps.std(ddof=1) / np.sqrt(len(ps)):.1f} over {len(ps)} seeds ({len(keys)} games)")
s = [r for r in arm["search"].values()]
if s:
    print(f"search cost: {np.mean([r['search_s'] for r in s]):.0f} s/match, {np.mean([r['search_s'] / max(r['searches'], 1) for r in s]):.2f} s/root, "
          f"wall {np.mean([r['wall_s'] for r in s]):.0f} s/match")
