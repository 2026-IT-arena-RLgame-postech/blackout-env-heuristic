"""Q12 (docs §46): per opponent, V31 vs V30 on the same (seed, side): first Hunter at decision 99, second race
w/t/l, Hunter-seconds (ours / theirs), match length.  python .../mech.py <matches.jsonl ...>"""
import json
import sys
from collections import defaultdict

import numpy as np

BASE = "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
CAND = BASE + ",threat_radius=0.13,camp_engage_radius=6.5"
rows = [json.loads(x) for f in sys.argv[1:] for x in open(f) if x.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r


def wtl(v):
    a = np.array([x["second_ours"] for x in v], float)
    b = np.array([x["second_opp"] for x in v], float)
    a[a < 0], b[b < 0] = 1e9, 1e9
    return f"{int((a < b).sum())}/{int((a == b).sum())}/{int((a > b).sum())}"


for o in sorted({o for _, o in by}):
    b, c = by.get((BASE, o), {}), by.get((CAND, o), {})
    k = sorted(set(b) & set(c))
    if not k:
        continue
    B, C = [b[x] for x in k], [c[x] for x in k]
    f = lambda v, key: np.mean([x[key] for x in v])  # noqa: E731
    print(f"{o[:40]:40s} n {len(k):4d} | first@99 ours {np.mean([x['first_ours'] == 99 for x in C]):.3f} "
          f"(V30 {np.mean([x['first_ours'] == 99 for x in B]):.3f}) | 2nd race w/t/l V31 {wtl(C)} V30 {wtl(B)} | "
          f"Hunter s ours {f(C, 'hunter_s'):5.1f} vs {f(B, 'hunter_s'):5.1f}, theirs {f(C, 'opp_hunter_s'):5.1f} vs "
          f"{f(B, 'opp_hunter_s'):5.1f} | ticks {f(C, 'ticks'):5.0f} vs {f(B, 'ticks'):5.0f} | margin {f(C, 'margin'):+5.1f} vs {f(B, 'margin'):+5.1f}")
