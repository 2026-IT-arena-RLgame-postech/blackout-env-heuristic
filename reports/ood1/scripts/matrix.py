"""OOD1 (docs §56): win-margin matrix + cycle test from v27_eval rows (any ours/opponent orientation).
M[a][b] = a's margin vs b (seed-level mean, both sides averaged; SE over seeds), from rows ours=a/opp=b or (negated)
ours=b/opp=a, merged per (seed, side).  Edge a > b when M[a][b] > 2 SE.  Prints the matrix, every directed 3-cycle
and the best responses to each column.
    python matrix.py reports/ood1/{screen,screen2,screen2b,screen_fu2,br}/matches.jsonl"""
import itertools, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from blackout_env.train.distill.ood_suite import C3_NOOP, OOD26, TEACHER_OF, V30, V31  # noqa: E402

NAME = {n: n for n in ("p20", "p21e", "p17", "p14b")}
NAME.update({d: n for n, d, _, _ in OOD26})
NAME.update({V30: "V30", C3_NOOP: "V30", V31: "V31", "strategic_v25": "V25", TEACHER_OF["p17"]: "V28",
             "strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve": "V29",
             "strategic_v27:reserve_stage=1.5": "V27", "strategic_v23": "V23", "strategic_v19": "v19",
             V30 + ",hunter_quota=4,hunter_mode=hunt": "h4hunt", V30 + ",hunter_mode=hunt": "hunt",
             V30 + ",hunter_quota=4": "h4", "strategic_early_rush": "early_rush", "strategic_fortress": "fortress",
             "strategic_raider": "raider", "strategic_duelist": "duelist", "strategic_cargo_hunter": "cargo_hunter"})
SET = ["V31", "V30", "h4hunt", "hunt", "h4", "p20", "p21e", "V25", "p17"]

cell = defaultdict(dict)  # (a, b) -> {(seed, side_of_a): margin of a}
for f in sys.argv[1:]:
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        a, b = NAME.get(r["ours"]), NAME.get(r["opponent"])
        if a is None or b is None or a == b:
            continue
        cell[(a, b)][(r["seed"], r["side"])] = r["margin"]
        cell[(b, a)][(r["seed"], 1 - r["side"])] = -r["margin"]


def stat(a, b):
    d = cell.get((a, b))
    if not d:
        return None
    per = defaultdict(list)
    for (s, _), m in d.items():
        per[s].append(m)
    x = np.array([np.mean(v) for v in per.values()])
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"), len(x)


names = SET + sorted({a for a, _ in cell if a not in SET and stat(a, "h4hunt") and stat(a, "h4hunt")[0] > 0})
print("row vs column margin (seeds); '>' = beyond 2 SE")
print(f"{'':12s}" + "".join(f"{b:>13s}" for b in names))
for a in names:
    out = f"{a:12s}"
    for b in names:
        s = stat(a, b) if a != b else None
        out += f"{'':>13s}" if s is None else f"{s[0]:+6.1f}±{s[1]:3.1f}{'>' if s[0] > 2 * s[1] else ('<' if s[0] < -2 * s[1] else ' '):1s}".rjust(13)
    print(out)
print()
beats = {(a, b) for a in names for b in names if a != b and stat(a, b) and stat(a, b)[0] > 2 * stat(a, b)[1]}
cyc = [c for c in itertools.permutations(names, 3) if c[0] == min(c) and (c[0], c[1]) in beats and (c[1], c[2]) in beats
       and (c[2], c[0]) in beats]
print("3-cycles (A>B>C>A, each beyond 2 SE):", "; ".join(" > ".join(c) + f" > {c[0]}" for c in cyc) or "none")
all_names = sorted({a for a, _ in cell})
for col in ("h4hunt", "h4", "hunt"):
    br = sorted(((stat(a, col), a) for a in all_names if a != col and stat(a, col)), key=lambda x: -x[0][0])[:6]
    print(f"best responses to {col}: " + ", ".join(f"{a} {s[0]:+.1f}±{s[1]:.1f}" for s, a in br))
for row in ("h4hunt", "h4"):
    rr = sorted(((stat(row, b), b) for b in all_names if b != row and stat(row, b)), key=lambda x: x[0][0])
    print(f"{row} as candidate: " + ", ".join(f"{b} {s[0]:+.1f}±{s[1]:.1f}" for s, b in rr))
