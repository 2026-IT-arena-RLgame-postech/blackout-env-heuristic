"""Rank candidates in one or more econ_stats ``matches.jsonl`` dirs: per opponent, mean margin / win rate and the
paired diff vs ``--base`` (same seed, both sides averaged first), and the mean paired diff over all opponents.

    PYTHONPATH=. .venv/bin/python examples/heur_rank.py --base V32.1 reports/heur_search/g0 reports/heur_search/g0_mac
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


_REF = {}


def short(name: str, width: int = 0) -> str:
    s = name.split("/")[-2] if name.startswith("net:") else name
    if s.startswith(("strategic_v32:", "strategic_v33:")):  # show only what differs from V32.1
        from blackout_env.heuristics.params import resolve_descriptor, split_descriptor
        if not _REF:
            _REF.update(split_descriptor(resolve_descriptor("V32.1"))[1])
        kv = split_descriptor(s)[1]
        s = "~" + ",".join(f"{k}={v}" for k, v in kv.items() if _REF.get(k) != v)
    return s if not width or len(s) <= width else s[:width - 3] + "..."


def load(dirs):
    by = defaultdict(dict)
    for d in dirs:
        p = Path(d) / "matches.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    return by


def paired(d, base, key=lambda r: r["margin"]):
    per = defaultdict(list)
    for k in d:
        if k in base:
            per[k[0]].append(key(d[k]) - key(base[k]))
    x = np.array([np.mean(v) for v in per.values()])
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)) if len(x) > 1 else (np.nan, np.nan, len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--base", required=True)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--top", type=int, default=0)
    a = p.parse_args()
    by = load(a.dirs)
    opps = sorted({k[1] for k in by})
    cands = sorted({k[0] for k in by})
    rows = []
    for c in cands:
        diffs, cells = [], []
        for o in opps:
            d = by.get((c, o))
            if not d:
                cells.append(f"{'-':>22s}")
                continue
            m = np.mean([r["margin"] for r in d.values()])
            w = np.mean([(np.sign(r["margin"]) + 1) / 2 for r in d.values()])
            base = by.get((a.base, o), {})
            mu, se, n = paired(d, base) if c != a.base else (0.0, 0.0, 0)
            if c != a.base and np.isfinite(mu):
                diffs.append((mu, se))
            cells.append(f"{m:+6.1f} {w:.2f} {mu:+5.1f}±{se:3.1f}")
        avg = np.mean([x[0] for x in diffs]) if diffs else (0.0 if c == a.base else np.nan)
        ase = np.sqrt(np.sum([x[1] ** 2 for x in diffs])) / len(diffs) if diffs else 0.0
        rows.append((avg, ase, c, cells, len(diffs)))
    rows.sort(key=lambda r: -np.nan_to_num(r[0], nan=-1e9))
    print(f"{'candidate':{a.width}s} {'avgΔ':>11s} " + " ".join(f"{short(o, 22):>22s}" for o in opps))
    print(f"{'':{a.width}s} {'':>11s} " + " ".join(f"{'margin win Δbase':>22s}" for _ in opps))
    for avg, ase, c, cells, nd in rows[: a.top or None]:
        print(f"{short(c, a.width):{a.width}s} {avg:+5.1f}±{ase:3.1f}({nd}) " + " ".join(cells))


if __name__ == "__main__":
    main()
