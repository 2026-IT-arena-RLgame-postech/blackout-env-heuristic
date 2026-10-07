"""OOD-26 report (docs/policy_v2_design_ko.md §56): metrics of examples/v27_eval.py rows against the OOD-26 suite.

Per (candidate, opponent), paired by (seed, side):
  margin mean ± SE (SE over seeds, both sides averaged first), win rate, ``cat`` = share of matches lost by more than
  ``--cat`` (20) points, ``catT`` = share lost by more than 20 on a (seed, side) the candidate's *teacher* won,
  ``h99`` = share whose first own Hunter came at decision 99, ``-T`` = candidate minus its teacher (TEACHER_OF in
  ood_suite.py, or ``--teacher name=descriptor``), ``-base`` = candidate minus ``--base`` (the gate's champion).
Per candidate: worst-opponent margin, mean margin over the suite, pooled cat / catT, the opponents where it trails its
teacher by > 5 at 2 SE, and (with --base) the OOD non-regression rule of §56:

  R1 worst-opponent margin: cand >= base - 5            R2 pooled catT rate: cand <= base + 2 points (%)
  R3 no opponent with cand - base < -5 at > 2 SE        R4 h99 rate on every opponent >= base - 0.10

    PYTHONPATH=. .venv/bin/python examples/ood_report.py --out reports/ood1/screen --base p20
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

from blackout_env.train.distill.ood_suite import OOD26, TEACHER_OF, V30, V31, short  # noqa: E402

GROUP = {d: g for _, d, g, _ in OOD26}
CAND = {V30: "V30", V31: "V31", TEACHER_OF["p17"]: "V28", "strategic_v25": "V25", TEACHER_OF["p16"]: "V27"}


def cn(c: str) -> str:
    return CAND.get(c, c)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def per_seed(d: dict, f) -> list[float]:
    by = defaultdict(list)
    for k, r in d.items():
        v = f(k, r)
        if v is not None:
            by[k[0]].append(v)
    return [float(np.mean(v)) for v in by.values()]


def paired(d: dict, base: dict) -> tuple[float, float, int]:
    x = per_seed({k: r for k, r in d.items() if k in base}, lambda k, r: r["margin"] - base[k]["margin"])
    return (float(np.mean(x)), se(x), len(x)) if x else (float("nan"), float("nan"), 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="dir(s) with matches.jsonl, ','-separated (merged)")
    p.add_argument("--base", default="", help="gate champion: adds cand - base and the R1-R4 rule")
    p.add_argument("--teacher", action="append", default=[], help="name=descriptor (extra TEACHER_OF entries)")
    p.add_argument("--cat", type=float, default=20.0)
    p.add_argument("--min-n", type=int, default=8)
    args = p.parse_args()
    teacher_of = dict(TEACHER_OF)
    teacher_of.update(dict(t.split("=", 1) for t in args.teacher))

    by = defaultdict(dict)
    for out in args.out.split(","):
        path = Path(out) / "matches.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    cands = sorted({c for c, _ in by}, key=lambda c: (c not in teacher_of, c))
    opps = [d for _, d, _, _ in OOD26 if any((c, d) in by for c in cands)]
    opps += sorted({o for _, o in by} - set(opps))

    stat = {}
    for c in cands:
        t = teacher_of.get(c)
        for o in opps:
            d = by.get((c, o), {})
            if len(d) < args.min_n:
                continue
            ms = per_seed(d, lambda k, r: r["margin"])
            v = list(d.values())
            tb = by.get((t, o), {}) if t else {}
            won = [k for k in d if k in tb and tb[k]["margin"] > 0]
            s = {"n": len(d), "m": float(np.mean(ms)), "se": se(ms), "win": float(np.mean([r["margin"] > 0 for r in v])),
                 "cat": float(np.mean([r["margin"] < -args.cat for r in v])),
                 "h99": float(np.mean([r["first_ours"] == 99 for r in v])),
                 "catT_num": sum(d[k]["margin"] < -args.cat for k in won), "catT_den": len(won)}
            if tb:
                s["gT"], s["gTse"], s["gTn"] = paired(d, tb)
            if args.base and c != args.base and (args.base, o) in by:
                s["gB"], s["gBse"], s["gBn"] = paired(d, by[(args.base, o)])
            stat[(c, o)] = s

    L = []
    L.append(f"OOD-26 report: {args.out}  (cat = margin < -{args.cat:g}; catT = cat on a (seed, side) the teacher won)")
    L.append("")
    hdr = f"{'candidate':10s} {'opponent':14s} {'grp':6s} {'n':>4s} {'margin':>12s} {'win':>5s} {'cat':>5s} {'catT':>9s} {'h99':>5s} {'- teacher':>13s}"
    if args.base:
        hdr += f" {'- ' + args.base:>13s}"
    for c in cands:
        L.append(hdr)
        for o in opps:
            s = stat.get((c, o))
            if not s:
                continue
            line = (f"{cn(c)[:10]:10s} {short(o)[:14]:14s} {GROUP.get(o, '-'):6s} {s['n']:4d} {s['m']:+6.1f}±{s['se']:4.1f} {s['win']:5.2f} "
                    f"{s['cat']:5.2f} {s['catT_num']:3d}/{s['catT_den']:<4d} {s['h99']:5.2f}")
            line += f" {s['gT']:+6.1f}±{s['gTse']:4.1f}" if "gT" in s else f" {'':13s}"
            if "gB" in s:
                line += f" {s['gB']:+6.1f}±{s['gBse']:4.1f}"
            L.append(line)
        L.append("")

    L.append("per candidate (over the OOD-26 opponents it played):")
    L.append(f"{'candidate':10s} {'opps':>4s} {'mean':>6s} {'worst':>22s} {'cat':>5s} {'catT':>10s} {'collapse vs teacher (gap < -5 at 2 SE)'}")
    summ = {}
    for c in cands:
        ss = {o: stat[(c, o)] for o in opps if (c, o) in stat and o in GROUP}
        if not ss:
            continue
        worst = min(ss, key=lambda o: ss[o]["m"])
        num = sum(s["catT_num"] for s in ss.values())
        den = sum(s["catT_den"] for s in ss.values())
        cat = float(np.mean([s["cat"] for s in ss.values()]))
        col = [f"{short(o)} {s['gT']:+.1f}" for o, s in sorted(ss.items(), key=lambda kv: kv[1].get("gT", 0))
               if "gT" in s and s["gT"] < -5 and s["gT"] < -2 * s["gTse"]]
        summ[c] = {"worst": ss[worst]["m"], "worst_o": worst, "catT": num / den if den else float("nan"), "ss": ss}
        L.append(f"{cn(c)[:10]:10s} {len(ss):4d} {np.mean([s['m'] for s in ss.values()]):+6.1f} "
                 f"{short(worst)[:14]:>14s} {ss[worst]['m']:+6.1f}  {cat:5.3f} {num:4d}/{den:<5d} {'; '.join(col) or '-'}")
    L.append("")

    if args.base and args.base in summ:
        b = summ[args.base]
        L.append(f"OOD non-regression rule vs {args.base} (§56): R1 worst >= base-5, R2 catT <= base+2 %, "
                 f"R3 no opponent < -5 at 2 SE, R4 h99 >= base-0.10 everywhere")
        for c, s in summ.items():
            if c == args.base:
                continue
            common = [o for o in s["ss"] if o in b["ss"]]
            r1 = s["worst"] >= b["worst"] - 5
            r2 = not (s["catT"] > b["catT"] + 0.02)
            bad = [f"{short(o)} {s['ss'][o]['gB']:+.1f}±{s['ss'][o]['gBse']:.1f}" for o in common
                   if "gB" in s["ss"][o] and s["ss"][o]["gB"] < -5 and s["ss"][o]["gB"] < -2 * s["ss"][o]["gBse"]]
            h99 = [f"{short(o)} {s['ss'][o]['h99']:.2f}<{b['ss'][o]['h99']:.2f}" for o in common
                   if s["ss"][o]["h99"] < b["ss"][o]["h99"] - 0.10]
            ok = r1 and r2 and not bad and not h99
            L.append(f"  {cn(c):10s} {'PASS' if ok else 'FAIL'}  R1 worst {s['worst']:+.1f} ({short(s['worst_o'])}) vs {b['worst']:+.1f} "
                     f"{'ok' if r1 else 'X'} | R2 catT {100 * s['catT']:.1f}% vs {100 * b['catT']:.1f}% {'ok' if r2 else 'X'} | "
                     f"R3 {'; '.join(bad) or 'ok'} | R4 {'; '.join(h99) or 'ok'}")
        L.append("")

    rows = [(c, o, s) for (c, o), s in stat.items() if "gT" in s and o in GROUP]
    if rows:
        L.append("ranked student - teacher gaps (worst first, |gap| > 2 SE):")
        for c, o, s in sorted(rows, key=lambda x: x[2]["gT"]):
            if s["gT"] < -2 * s["gTse"]:
                L.append(f"  {cn(c):10s} vs {short(o):14s} {s['gT']:+6.1f}±{s['gTse']:4.1f}  (margin {s['m']:+.1f}, "
                         f"catT {s['catT_num']}/{s['catT_den']}, h99 {s['h99']:.2f})")
    txt = "\n".join(L)
    print(txt)
    (Path(args.out.split(",")[0]) / "ood_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
