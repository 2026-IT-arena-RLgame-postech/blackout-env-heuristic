"""Paired (by seed and side) margin diff of each angle-quant setting vs q0 (reports/rl/angle_probe/q<Q>__<opp>__<lo>.json)."""
import glob, json, os, re, sys
import numpy as np

D = os.path.join(os.path.dirname(__file__), "angle_probe")
SHORT = {"hunter_quota=4,hunter_mode=hunt": "v30_h4hunt", "solo_sentry=true,hunter_mode=hunt": "v30_hunt",
         "exposure_cost=0.2": "v30_reckless", "rush_hunters=5": "rush_all"}


def short(spec):
    for k, v in SHORT.items():
        if k in spec:
            return v
    return "early_rush" if spec == "strategic_early_rush" else os.path.basename(spec).replace(".pt", "")


def load(q):
    out = {}
    for f in glob.glob(f"{D}/{'q' + q if q.lstrip('-').isdigit() else q}__*.json"):
        r = json.load(open(f))
        d = out.setdefault(short(r["opp"]), {})
        for s, a, b in zip(r["seeds"], r["margin_side0"], r["margin_side1"]):
            d[(s, 0)], d[(s, 1)] = a, b
    return out


base = load("0")
qs = sys.argv[1:] or ["-1", "32", "64", "128"]
opps = list(base)
print(f"{'opp':14s} {'q0':>12s} " + " ".join(f"{'q' + q + '-q0':>12s}" for q in qs))
for o in opps + ["mean"]:
    cells = []
    for q in qs:
        d = load(q)
        if o == "mean":
            ks = [(oo, k) for oo in opps if oo in d for k in base[oo] if k in d[oo]]
            x = np.array([np.mean([d[oo][k] - base[oo][k] for oo in opps if oo in d and k in d[oo]]) for k in base[opps[0]]])
        else:
            if o not in d:
                cells.append(f"{'n/a':>12s}"); continue
            x = np.array([d[o][k] - base[o][k] for k in base[o] if k in d[o]])
        cells.append(f"{x.mean():+6.1f}±{x.std(ddof=1) / np.sqrt(len(x)):4.1f}")
    b = np.mean([np.mean(list(base[oo].values())) for oo in opps]) if o == "mean" else np.mean(list(base[o].values()))
    print(f"{o:14s} {b:+12.1f} " + " ".join(cells))
