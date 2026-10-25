"""Ablation fast-eval analysis: per-opponent margin/win per tag, paired diffs arm - control (seed-paired).
Usage: .venv/bin/python reports/rl/abl_analysis.py [arm:control ...]   (files reports/rl/abl_eval/<tag>__<opp>__<lo>.json)"""
import glob, json, os, sys
import numpy as np

D = os.path.join(os.path.dirname(__file__), "abl_eval")
OPPS = ["p24c_v321", "p22_v31", "V32.1"]


def load(tag, opp):
    fs = sorted(glob.glob(f"{D}/{tag}__{opp}__*.json"), key=lambda f: int(f.rsplit("__", 1)[1][:-5]))
    if not fs:
        return None
    s0, s1 = [], []
    for f in fs:
        d = json.load(open(f)); s0 += d["margin_side0"]; s1 += d["margin_side1"]
    return np.array(s0 + s1, float)


win = lambda x: (np.sign(x) + 1) / 2
tags = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{D}/*.json")})
print("tag            " + " | ".join(f"{o:>16s}" for o in OPPS) + " | h2h vs ctrl")
for t in tags:
    row = []
    for o in OPPS:
        x = load(t, o)
        row.append(f"{x.mean():+5.1f}±{x.std(ddof=1)/len(x)**.5:.1f}/{100*win(x).mean():4.1f}" if x is not None else " " * 16)
    h = [os.path.basename(f).split("__")[1] for f in glob.glob(f"{D}/{t}__*.json") if os.path.basename(f).split("__")[1] not in OPPS]
    hs = ""
    if h:
        x = load(t, h[0]); hs = f"{h[0]}: {x.mean():+.1f}±{x.std(ddof=1)/len(x)**.5:.1f} win {100*win(x).mean():.1f}%"
    print(f"{t:14s} " + " | ".join(row) + " | " + hs)
pairs = [a.split(":") for a in sys.argv[1:]]
for a, b in pairs:
    out = []
    for o in OPPS:
        x, y = load(a, o), load(b, o)
        if x is None or y is None or len(x) != len(y):
            out.append(f"{o}: n/a"); continue
        d = x - y; dw = win(x) - win(y); n = len(d)
        out.append(f"{o}: {d.mean():+.1f}±{d.std(ddof=1)/n**.5:.1f} ({100*dw.mean():+.1f}pp)")
    xs = [load(a, o) for o in OPPS]; ys = [load(b, o) for o in OPPS]
    if all(v is not None for v in xs + ys):
        d = np.mean([x - y for x, y in zip(xs, ys)], 0)
        out.append(f"avg3: {d.mean():+.1f}±{d.std(ddof=1)/len(d)**.5:.1f}")
    print(f"{a} - {b}: " + " | ".join(out))
