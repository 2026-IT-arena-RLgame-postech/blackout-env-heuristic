"""QZ: paired (seed, side) margin difference between two blind_search_eval / v27_eval row sets, per opponent.

    python reports/qz_quant/scripts/paired.py search reports/qz_quant/search_p20 reports/qz_quant/search_p20c
    python reports/qz_quant/scripts/paired.py plain reports/qz_quant/plain64 p20 p20c
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def load(paths):
    return [json.loads(x) for p in paths for x in Path(p).read_text().splitlines() if x.strip()]


def report(a, b, label):
    """a, b: {(opp, seed, side): row}; prints b - a per opponent (SE over seeds, sides averaged first)."""
    opps = sorted({k[0] for k in a})
    for o in opps:
        keys = sorted(k for k in a if k[0] == o and k in b)
        per = defaultdict(list)
        for k in keys:
            per[k[1]].append(b[k]["margin"] - a[k]["margin"])
        d = [np.mean(v) for v in per.values()]
        same = np.mean([a[k]["margin"] == b[k]["margin"] for k in keys])
        line = (f"{label} vs {o[:40]:40s} n={len(keys):4d} A {np.mean([a[k]['margin'] for k in keys]):+6.1f} "
                f"B {np.mean([b[k]['margin'] for k in keys]):+6.1f}  B-A {np.mean(d):+5.1f} ± {se(d):4.1f} "
                f"({len(d)} seeds), same margin {same:.2f}")
        if "search_s" in a[keys[0]]:
            def cost(rows):
                ch = {(r["chunk_wall_s"], r["search_s"]): r for r in rows}.values()
                return (sum(r["chunk_wall_s"] for r in ch), sum(r["search_s"] for r in ch), sum(r["chunk_searches"] for r in ch),
                        sum(r["chunk_n"] for r in ch))
            for nm, rows in (("A", [a[k] for k in keys]), ("B", [b[k] for k in keys])):
                w, s, r, n = cost(rows)
                line += f"\n    {nm}: wall {w / n:.1f} s/game, search {s / r:.3f} s/root, {r / n:.1f} roots/game"
        print(line)


def main():
    if sys.argv[1] == "search":
        a, b = ({(r["opponent"], r["seed"], r["side"]): r for r in load(Path(d).glob("matches_*.jsonl"))} for d in sys.argv[2:4])
        report(a, b, "search")
    else:
        rows = load([Path(sys.argv[2]) / "matches.jsonl"])
        a = {(r["opponent"], r["seed"], r["side"]): r for r in rows if r["ours"] == sys.argv[3]}
        b = {(r["opponent"], r["seed"], r["side"]): r for r in rows if r["ours"] == sys.argv[4]}
        report(a, b, "plain")


if __name__ == "__main__":
    main()
