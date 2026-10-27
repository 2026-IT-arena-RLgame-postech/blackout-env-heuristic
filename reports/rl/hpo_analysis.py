"""HPO curve analysis: avg4 margin per checkpoint per trial, and paired diffs trial - base (seed and side paired).
Speed = curve mean (iter_50/100/200 + final restricted to the first 128 seeds); final = 256 seeds.
Usage: python reports/rl/hpo_analysis.py [base_trial]   (files reports/rl/hpo_eval/hpo_<trial>_<ck>__<opp>__<lo>.json)"""
import glob, json, os, sys
import numpy as np

D = os.path.join(os.path.dirname(__file__), "hpo_eval")
OPPS = ["p24c_v321", "p22_v31", "V32.1", "V33"]
CKS = ["iter_50", "iter_100", "iter_200", "final"]


def load(trial, ck, opp, n=None):
    fs = sorted(glob.glob(f"{D}/hpo_{trial}_{ck}__{opp}__*.json"), key=lambda f: int(f.rsplit("__", 1)[1][:-5]))
    if not fs:
        return None
    s0, s1 = [], []
    for f in fs:
        d = json.load(open(f)); s0 += d["margin_side0"]; s1 += d["margin_side1"]
    if n:
        s0, s1 = s0[:n], s1[:n]
    return np.array(s0 + s1, float)


def avg4(trial, ck, n=None):
    xs = [load(trial, ck, o, n) for o in OPPS]
    return None if any(x is None for x in xs) else np.mean(xs, 0)


ms = lambda d: f"{d.mean():+5.1f}±{d.std(ddof=1)/len(d)**.5:.1f}"
base = sys.argv[1] if len(sys.argv) > 1 else "base"
trials = sorted({os.path.basename(f)[4:].split("_iter_")[0].split("_final")[0] for f in glob.glob(f"{D}/hpo_*.json")})
trials = [base] + [t for t in trials if t != base]

print("avg4 margin (absolute)\ntrial        " + " ".join(f"{c:>12s}" for c in CKS))
for t in trials:
    print(f"{t:12s} " + " ".join(f"{ms(x):>12s}" if (x := avg4(t, c)) is not None else " " * 12 for c in CKS))

print("\npaired vs", base, "(avg4 diff; curve = mean over 4 ckpts on first 128 seeds)")
print("trial        " + " ".join(f"{c:>12s}" for c in CKS) + "        curve | final per opp")
for t in trials[1:]:
    row, cur = [], []
    for c in CKS:
        x, y = avg4(t, c), avg4(base, c)
        row.append(f"{ms(x - y):>12s}" if x is not None and y is not None and len(x) == len(y) else " " * 12)
        x, y = avg4(t, c, 128), avg4(base, c, 128)
        if x is not None and y is not None:
            cur.append(x - y)
    cs = ms(np.mean(cur, 0)) if len(cur) == len(CKS) else "  (partial)"
    po = []
    for o in OPPS:
        x, y = load(t, "final", o), load(base, "final", o)
        if x is not None and y is not None and len(x) == len(y):
            po.append(f"{o.split('_')[0]} {(x - y).mean():+.1f}")
    print(f"{t:12s} " + " ".join(row) + f" {cs:>12s} | " + ", ".join(po))
