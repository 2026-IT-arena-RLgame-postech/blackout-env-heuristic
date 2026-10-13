"""Per opponent: mean log-likelihood per move / top-1 of every candidate net, and the posterior's pick at 5 / 30 / 150 s."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/opp_model_fit")
for f in sorted(d.glob("fit_*.jsonl")):
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    names = list(rows[0]["ll_per_move"])
    ll = {n: np.mean([r["ll_per_move"][n] for r in rows]) for n in names}
    t1 = {n: np.mean([r["top1"][n] for r in rows]) for n in names}
    picks = defaultdict(lambda: defaultdict(int))
    conf = defaultdict(list)
    for r in rows:
        for at, p in r["posterior_at"].items():
            best = max(p, key=p.get)
            picks[at][best] += 1
            conf[at].append(p[best])
    order = sorted(names, key=lambda n: -ll[n])
    print(f"{f.stem:22s} n={len(rows):2d} " + "  ".join(f"{n} {ll[n]:+.3f}/{t1[n]:.2f}" for n in order))
    for at in sorted(picks, key=int):
        print(f"{'':26s}posterior @{int(at) // 25:>3d}s: " + ", ".join(f"{n} x{c}" for n, c in sorted(picks[at].items(), key=lambda x: -x[1]))
              + f"  (mean max-p {np.mean(conf[at]):.2f})")
