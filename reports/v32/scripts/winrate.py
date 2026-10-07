"""V32 (docs §58.6): paired win-rate / catastrophic-loss table of a candidate vs a base on the same (seed, side),
from every matches.jsonl under reports/v32/* and reports/ood1/* (a (ours, opponent, seed, side) row counted once).
win = 1 / 0.5 / 0 for margin > / = / < 0; cat = margin < -20.  SE over seeds (both sides averaged first).
Weak-opponent rule (§58.6): an opponent both arms beat at win >= 0.9 counts as a regression only if the paired
win-rate drop is > 2 SE and > 2 %p, or the cat rate rises by > 2 SE.
    python reports/v32/scripts/winrate.py [--cand V32] [--base V31] [--min-n 128]"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from blackout_env.heuristics.v32_four_hunters import V31_PARAMS, V32  # noqa: E402
from blackout_env.train.distill.ood_suite import short  # noqa: E402

V31 = "strategic_v30:" + V31_PARAMS
NAMES = {"V32": V32, "V31": V31}


def se_seed(vals):
    """vals: {(seed, side): x} -> mean, SE over seeds (sides averaged)."""
    per = defaultdict(list)
    for (s, _), x in vals.items():
        per[s].append(x)
    m = np.array([np.mean(v) for v in per.values()])
    return float(np.mean(list(vals.values()))), float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="V32")
    ap.add_argument("--base", default="V31")
    ap.add_argument("--min-n", type=int, default=128)
    a = ap.parse_args()
    cand, base = NAMES.get(a.cand, a.cand), NAMES.get(a.base, a.base)
    rows = {}
    for f in sorted(glob.glob(str(REPO / "reports/v32/*/matches.jsonl")) + glob.glob(str(REPO / "reports/ood1/*/matches.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ours") in (cand, base):
                rows[(r["ours"], r["opponent"], r["seed"], r["side"])] = r["margin"]
    by = defaultdict(lambda: defaultdict(dict))
    for (o, op, s, sd), m in rows.items():
        by[op][(s, sd)][o] = m
    w = lambda m: 1.0 if m > 0 else 0.5 if m == 0 else 0.0  # noqa: E731
    out = []
    for op, d in by.items():
        pr = {k: v for k, v in d.items() if cand in v and base in v}
        if len(pr) < a.min_n:
            continue
        wc, _ = se_seed({k: w(v[cand]) for k, v in pr.items()})
        wb, _ = se_seed({k: w(v[base]) for k, v in pr.items()})
        dw, sdw = se_seed({k: w(v[cand]) - w(v[base]) for k, v in pr.items()})
        dc, sdc = se_seed({k: float(v[cand] < -20) - float(v[base] < -20) for k, v in pr.items()})
        mc, _ = se_seed({k: v[cand] for k, v in pr.items()})
        mb, _ = se_seed({k: v[base] for k, v in pr.items()})
        dm, sdm = se_seed({k: v[cand] - v[base] for k, v in pr.items()})
        weak = wc >= 0.9 and wb >= 0.9
        reg = (dw < -0.02 and dw < -2 * sdw) or (dc > 2 * sdc and dc > 0)
        verdict = ("REGRESSION" if reg else "ok (weak)") if weak else ("margin-regr" if dm < -1 and dm < -sdm else "")
        out.append((dw, short(op) if op != V31 else "V31", len(pr), wc, wb, dw, sdw, dc, sdc, mc, mb, dm, sdm, verdict))
    out.sort(key=lambda x: x[0])
    print(f"{a.cand} vs {a.base}: paired over (seed, side); SE over seeds")
    print(f"{'opponent':16s} {'n':>5s} {'win':>6s} {'base':>6s} {'Δwin':>13s} {'Δcat':>13s} {'margin':>7s} {'base':>7s} {'Δmargin':>12s}")
    for _, name, n, wc, wb, dw, sdw, dc, sdc, mc, mb, dm, sdm, v in out:
        print(f"{name:16s} {n:5d} {wc:6.3f} {wb:6.3f} {100 * dw:+6.1f}±{100 * sdw:4.1f}%p {100 * dc:+6.1f}±{100 * sdc:4.1f}%p "
              f"{mc:+7.1f} {mb:+7.1f} {dm:+6.1f}±{sdm:4.1f}  {v}")


if __name__ == "__main__":
    main()
