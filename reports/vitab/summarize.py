"""Summarise reports/vitab/sim/matches.jsonl: per student x opponent win/margin/SE, per arm (mean of 2 training seeds), A-paired deltas, h2h."""
import json, sys, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
OPP = {"net:" : ""}
def name(o): return {"p24c": "P24c", "p22": "P22", "p5a": "P5a", "strategic_v19": "V19"}.get(o, o)
d = collections.defaultdict(dict)  # (ours, opp) -> {(seed, side): row}
for r in rows: d[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
# restrict every opponent to the (seed, side) cells that all six students played
STU = ["As0","As1","B1s0","B1s1","B2s0","B2s1"]
for o in {k[1] for k in d if k[0] in STU}:
    if not all((st, o) in d for st in STU): continue
    common = set.intersection(*[set(d[(st, o)]) for st in STU])
    common = {k for k in common if (k[0], 1 - k[1]) in common}
    for st in STU: d[(st, o)] = {k: v for k, v in d[(st, o)].items() if k in common}
def per_seed(cell, key):  # seed-level value = mean over sides
    s = collections.defaultdict(list)
    for (sd, _), r in cell.items(): s[sd].append(key(r))
    return {k: float(np.mean(v)) for k, v in s.items()}
se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
win = lambda r: 1.0 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0.0
opps = [o for o in ["p24c", "p22", "p5a", "strategic_v19"] if any(k[1] == o for k in d)]
arms = {"A": ["As0", "As1"], "B1": ["B1s0", "B1s1"], "B2": ["B2s0", "B2s1"]}
out = []
out.append("### Per student (margin = our score - opp score, mean +- SE over seeds [sides averaged]; win rate with draws 1/2)\n")
out.append("| student | " + " | ".join(name(o) for o in opps) + " |\n|---|" + "---|" * len(opps))
for st in [s for a in arms.values() for s in a]:
    c = []
    for o in opps:
        cell = d.get((st, o), {})
        if not cell: c.append("-"); continue
        m = per_seed(cell, lambda r: r["margin"]); w = per_seed(cell, win)
        c.append(f"{np.mean(list(m.values())):+.1f}+-{se(list(m.values())):.1f}, win {np.mean(list(w.values())):.3f} (n={len(cell)})")
    out.append(f"| {st} | " + " | ".join(c) + " |")
out.append("\n### Per arm (mean of the 2 training seeds; SE = from the two student means and, in brackets, pooled-seed SE)\n")
out.append("| arm | " + " | ".join(name(o) for o in opps) + " | mean of opps |\n|---|" + "---|" * (len(opps) + 1))
armval = {}
for a, sts in arms.items():
    c, tot = [], []
    for o in opps:
        sm = []
        for st in sts:
            cell = d.get((st, o), {})
            if cell: sm.append(list(per_seed(cell, lambda r: r["margin"]).values()))
        if len(sm) < 2: c.append("-"); continue
        means = [np.mean(x) for x in sm]
        # SE of arm mean: independent student/seed sampling variance
        s = np.sqrt(sum(se(x) ** 2 for x in sm)) / 2
        armval[(a, o)] = (np.mean(means), s); tot.append(np.mean(means))
        c.append(f"{np.mean(means):+.1f}+-{s:.1f}")
    out.append(f"| {a} | " + " | ".join(c) + f" | {np.mean(tot):+.1f} |")
out.append("\n### Arm minus A, paired by (opponent, seed): mean over the 2 training seeds, SE over seeds\n")
out.append("| arm | " + " | ".join(name(o) for o in opps) + " |\n|---|" + "---|" * len(opps))
for a in ("B1", "B2"):
    c = []
    for o in opps:
        diffs = []
        for sa, sb in zip(arms[a], arms["A"]):
            x, y = d.get((sa, o), {}), d.get((sb, o), {})
            ps, qs = per_seed(x, lambda r: r["margin"]), per_seed(y, lambda r: r["margin"])
            diffs.append([ps[k] - qs[k] for k in ps if k in qs])
        if not all(diffs): c.append("-"); continue
        n = min(len(x) for x in diffs)
        pooled = np.mean([x[:n] for x in diffs], axis=0)
        c.append(f"{np.mean(pooled):+.1f}+-{se(pooled):.1f} (s0 {np.mean(diffs[0]):+.1f}, s1 {np.mean(diffs[1]):+.1f})")
    out.append(f"| {a}-A | " + " | ".join(c) + " |")
h = [(k, v) for k, v in d.items() if k[1] in ("B1s0","B1s1","B2s0","B2s1") ]
out.append("\n### Head to head (row student vs column student; margin from the row student's view, sides averaged)\n")
out.append("| match | margin | win rate |\n|---|---|---|")
for (a, b), cell in sorted(h):
    m = per_seed(cell, lambda r: r["margin"]); w = per_seed(cell, win)
    out.append(f"| {a} vs {b} | {np.mean(list(m.values())):+.1f}+-{se(list(m.values())):.1f} | {np.mean(list(w.values())):.3f} (n={len(cell)}) |")
print("\n".join(out))
