"""R4 search gate report (docs §54; after reports/r3_search/scripts/gate_report.py): paired differences between (base, tag) arms across the
reports/r4_search/sim_<base> row sets -- same opponent, seed, side; SE over seeds (sides averaged);
pooled = mean over opponents, SE = sqrt(sum SE^2) / n (as blind_search_eval.py report).

    python reports/r4_search/scripts/gate_report.py p21e:o4f_r21-p21e:plain p21e:o4f_r21-p20:o4f_r20 ...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "examples"))
from blind_search_eval import OPPONENTS, paired  # noqa: E402

rows = defaultdict(dict)
for d in ROOT.glob("sim_*"):
    base = d.name[4:]
    for f in d.glob("matches_*.jsonl"):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                rows[(f"{base}:{r['tag']}", r["opponent"])][(r["seed"], r["side"])] = r
arms = sorted({a for a, _ in rows})
opps = [o for o in OPPONENTS if any((a, o) in rows for a in arms)]
out = ["margins (mean over rows; n = seeds)", f"{'arm':18s} " + " ".join(f"{o:>11s}" for o in opps)]
for a in arms:
    out.append(f"{a:18s} " + " ".join(f"{np.mean([r['margin'] for r in rows[(a, o)].values()]):+7.1f}/{len(rows[(a, o)]) // 2:<3d}"
                                      if (a, o) in rows else f"{'':>11s}" for o in opps))
out += ["", "paired a - b (same opponent, seed, side), pooled over common opponents"]
summary = {}
for spec in sys.argv[1:]:
    a, b = spec.split("-", 1)
    ds = {o: paired(rows[(a, o)], rows[(b, o)]) for o in opps if (a, o) in rows and (b, o) in rows}
    ds = {o: v for o, v in ds.items() if v[2] > 1}
    if not ds:
        continue
    pm = float(np.mean([v[0] for v in ds.values()]))
    ps = float(np.sqrt(sum(v[1] ** 2 for v in ds.values())) / len(ds))
    worst = min(ds.items(), key=lambda kv: kv[1][0])
    out.append(f"  {a} - {b}: pooled {pm:+.1f}±{ps:.1f} ({pm / ps:+.1f} SE), worst {worst[0]} {worst[1][0]:+.1f}±{worst[1][1]:.1f}")
    out.append("     " + "  ".join(f"{o} {v[0]:+.1f}±{v[1]:.1f}/{v[2]}" for o, v in ds.items()))
    summary[spec] = {"pooled": pm, "se": ps, "by_opponent": {o: {"diff": v[0], "se": v[1], "seeds": v[2]} for o, v in ds.items()}}
txt = "\n".join(out)
print(txt)
(ROOT / "gate_report.txt").write_text(txt + "\n")
(ROOT / "gate_summary.json").write_text(json.dumps(summary, indent=1))
