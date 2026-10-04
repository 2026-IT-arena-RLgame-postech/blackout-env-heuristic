"""Q7 gate summary (docs/policy_v2_design_ko.md §36): paired Δ (candidate - base on the same seed and side, SE over
seeds with both sides averaged) per opponent, pooled over the opponent pool (mean of per-opponent Δ; SE from the
per-seed pooled Δ), worst opponent, and the pre-set decision rule (adopt |Δ| >= 3 SE, reject |Δ| < 1 SE at >= 256
seeds, else more seeds; no regression beyond 1 SE vs any opponent).

    python examples/q7_gate_report.py <matches.jsonl ...> --base 'strategic_v27:reserve_stage=1.5' [--pool a;b;...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--base", default="strategic_v27:reserve_stage=1.5")
    p.add_argument("--pool", default="strategic_v27:reserve_stage=1.5;strategic_v25;strategic_v23;p14b;p16;p11;"
                                     "strategic_v27:reserve_stage=1.0;perturb:eps=0.1:strategic_v27:reserve_stage=1.5")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rows = [json.loads(x) for f in a.files for x in open(f) if x.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r["margin"]
    pool = a.pool.split(";")
    opps = sorted({o for _, o in by}, key=lambda o: (o not in pool, pool.index(o) if o in pool else 0, o))
    cands = sorted({c for c, _ in by if c != a.base})
    lines = []
    for c in cands:
        lines.append(f"== {c}  - {a.base}")
        pooled = defaultdict(list)
        stats = {}
        for o in opps:
            d, b = by.get((c, o), {}), by.get((a.base, o), {})
            keys = sorted(set(d) & set(b))
            if not keys:
                continue
            per = defaultdict(list)
            for k in keys:
                per[k[0]].append(d[k] - b[k])
            ps = {s: np.mean(v) for s, v in per.items()}
            x = np.array(list(ps.values()))
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            stats[o] = (m, se, len(x))
            if o in pool:
                for s, v in ps.items():
                    pooled[s].append(v)
            flag = "REGRESS>1SE" if m < -se else ""
            lines.append(f"   vs {o:52s} {m:+7.2f} ± {se:5.2f}  ({len(x)} seeds, z {m / se if se > 0 else 0:+5.1f}) {flag}")
        full = [np.mean(v) for v in pooled.values() if len(v) == len([o for o in pool if o in stats])]
        if full:
            x = np.array(full)
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            inpool = {o: s for o, s in stats.items() if o in pool}
            worst = min(inpool, key=lambda o: inpool[o][0])
            n = min(s[2] for s in inpool.values())
            verdict = ("adopt" if abs(m) >= 3 * se and all(s[0] >= -s[1] for s in inpool.values()) else
                       "reject" if abs(m) < se and n >= 256 else "more seeds")
            lines.append(f"   POOL ({len(inpool)} opponents, {len(x)} seeds): {m:+.2f} ± {se:.2f}; worst {worst} "
                         f"{inpool[worst][0]:+.2f} ± {inpool[worst][1]:.2f}; rule -> {verdict}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
