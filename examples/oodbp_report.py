"""OOD-BP report: per-opponent margin / win rate / worst cases of candidates against the post-patch suite
(blackout_env/train/distill/ood_bp.py), rows from ``examples/v27_eval.py play --opponent oodbp --hz 50``.

Per (candidate, opponent), paired by (seed, side): margin mean +- SE (SE over seeds, both sides averaged first), win and
loss rate (a draw counts half to neither), ``cat`` = share of matches lost by more than ``--cat`` (20), ``p10`` = mean
of the worst 10 % matches, and with ``--base`` the paired difference candidate - base (+- SE).  Per candidate: mean
margin over the suite (and per group), worst opponent, pooled cat, and the *weak spots*: opponents where the candidate
loses > ``--weak-loss`` of its matches, has margin < 0, or trails ``--base`` by > 5 at 2 SE.  The ``crushed`` group
is read by win rate (it is expected to be won by a wide margin).

    PYTHONPATH=. .venv/bin/python examples/oodbp_report.py --out reports/bp/ood/base --base B6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from blackout_env.train.distill.ood_bp import OODBP, short  # noqa: E402

GROUP = {d: g for _, d, g, _ in OODBP}
ORDER = [d for _, d, _, _ in OODBP]


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def per_seed(d: dict, f) -> list[float]:
    by = defaultdict(list)
    for k, r in d.items():
        by[k[0]].append(f(k, r))
    return [float(np.mean(v)) for v in by.values()]


def cname(c: str) -> str:
    return Path(c[4:]).stem if c.startswith("net:") else c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="dir(s) with matches.jsonl, ','-separated (merged)")
    p.add_argument("--base", default="", help="reference candidate (name as in the rows): adds cand - base")
    p.add_argument("--cat", type=float, default=20.0)
    p.add_argument("--min-n", type=int, default=8)
    p.add_argument("--weak-loss", type=float, default=0.35)
    args = p.parse_args()

    by = defaultdict(dict)
    for out in args.out.split(","):
        for line in (Path(out) / "matches.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    cands = sorted({c for c, _ in by}, key=lambda c: (c != args.base, c))
    opps = [o for o in ORDER if any((c, o) in by for c in cands)] + sorted({o for _, o in by} - set(ORDER))

    st = {}
    for c in cands:
        for o in opps:
            d = by.get((c, o), {})
            if len(d) < args.min_n:
                continue
            m = np.array([r["margin"] for r in d.values()], float)
            s = {"n": len(m), "m": float(np.mean(per_seed(d, lambda k, r: r["margin"]))),
                 "se": se(per_seed(d, lambda k, r: r["margin"])), "win": float((m > 0).mean() + 0.5 * (m == 0).mean()),
                 "loss": float((m < 0).mean() + 0.5 * (m == 0).mean()), "cat": float((m < -args.cat).mean()),
                 "p10": float(np.sort(m)[:max(1, len(m) // 10)].mean())}
            b = by.get((args.base, o)) if args.base and c != args.base else None
            if b:
                keys = [k for k in d if k in b]
                if len(keys) >= args.min_n:
                    x = per_seed({k: d[k] for k in keys}, lambda k, r: r["margin"] - b[k]["margin"])
                    s["gB"], s["gBse"] = float(np.mean(x)), se(x)
            st[(c, o)] = s

    L = [f"OOD-BP report: {args.out}  (cat = margin < -{args.cat:g}; p10 = mean of worst 10 % matches)", ""]
    for c in cands:
        L.append(f"== {cname(c)}")
        L.append(f"{'opponent':16s} {'grp':8s} {'n':>4s} {'margin':>13s} {'win':>5s} {'loss':>5s} {'cat':>5s} {'p10':>7s}"
                 + (f" {'- ' + cname(args.base):>14s}" if args.base else ""))
        for o in opps:
            s = st.get((c, o))
            if not s:
                continue
            line = (f"{short(o)[:16]:16s} {GROUP.get(o, '-'):8s} {s['n']:4d} {s['m']:+7.1f}±{s['se']:4.1f} {s['win']:5.2f} "
                    f"{s['loss']:5.2f} {s['cat']:5.2f} {s['p10']:+7.1f}")
            if "gB" in s:
                line += f" {s['gB']:+7.1f}±{s['gBse']:4.1f}"
            L.append(line)
        L.append("")

    L.append("per candidate (suite without the `crushed` group; crushed reported as win rate):")
    L.append(f"{'candidate':22s} {'opps':>4s} {'mean':>7s} {'worst opponent':>24s} {'cat':>5s}  per-group mean  |  crushed win  |  weak spots")
    for c in cands:
        ss = {o: st[(c, o)] for o in opps if (c, o) in st and GROUP.get(o) not in (None, "crushed")}
        cr = [st[(c, o)]["win"] for o in opps if (c, o) in st and GROUP.get(o) == "crushed"]
        if not ss:
            continue
        w = min(ss, key=lambda o: ss[o]["m"])
        groups = defaultdict(list)
        for o, s in ss.items():
            groups[GROUP[o]].append(s["m"])
        weak = []
        for o, s in ss.items():
            why = []
            if s["loss"] > args.weak_loss:
                why.append(f"loss {s['loss']:.2f}")
            if s["m"] < 0:
                why.append(f"margin {s['m']:+.1f}")
            if "gB" in s and s["gB"] < -5 and s["gB"] < -2 * s["gBse"]:
                why.append(f"vs base {s['gB']:+.1f}")
            if why:
                weak.append(f"{short(o)} ({', '.join(why)})")
        L.append(f"{cname(c)[:22]:22s} {len(ss):4d} {np.mean([s['m'] for s in ss.values()]):+7.1f} "
                 f"{short(w)[:14]:>14s} {ss[w]['m']:+6.1f} {np.mean([s['cat'] for s in ss.values()]):5.2f}  "
                 + " ".join(f"{g}:{np.mean(v):+.0f}" for g, v in sorted(groups.items()))
                 + (f"  |  {np.mean(cr):.2f}" if cr else "  |  -"))
        L[-1] += "  |  " + ("; ".join(weak) or "-")
    txt = "\n".join(L)
    print(txt)
    (Path(args.out.split(",")[0]) / "oodbp_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
