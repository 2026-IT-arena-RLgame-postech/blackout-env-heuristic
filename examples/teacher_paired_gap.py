"""Paired margin gap of a teacher (heuristic) against a baseline teacher on the same matches.

Reads distill_heldout_eval.py ``matches.jsonl`` files (teacher-only rows: ``student`` is null),
pairs candidate and baseline by (opponent, seed, swap) and prints, per opponent and overall, both
margins (learner_score - opponent_score) and the paired gap ± its standard error over pairs.
Several files per side are merged; a match key played more than once is averaged.

  python examples/teacher_paired_gap.py --candidate reports/teacher_v2/v22/matches.jsonl \\
      --baseline reports/anyangle_targets/strategic_v21_anyangle/matches.jsonl reports/d9_vs_students/matches.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(paths: list[Path], teacher: str | None) -> dict[tuple, float]:
    """(opponent, seed, swap) -> margin; a key played more than once (repeat runs) is averaged."""
    rows = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("student") is not None or (teacher and r.get("teacher") != teacher):
                continue
            rows[(r["opponent"], r["seed"], r["swap"])].append(r["learner_score"] - r["opponent_score"])
    return {key: float(np.mean(values)) for key, values in rows.items()}


def se(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate", nargs="+", type=Path, required=True)
    p.add_argument("--baseline", nargs="+", type=Path, required=True)
    p.add_argument("--candidate-teacher", default=None, help="keep only rows of this --teacher")
    p.add_argument("--baseline-teacher", default="act:strategic_v21_anyangle")
    p.add_argument("--opponents", nargs="*", default=None, help="restrict (and order) the opponents")
    args = p.parse_args()
    new, old = load(args.candidate, args.candidate_teacher), load(args.baseline, args.baseline_teacher)
    gaps, mn, mo = defaultdict(list), defaultdict(list), defaultdict(list)
    for key in new.keys() & old.keys():
        gaps[key[0]].append(new[key] - old[key])
        mn[key[0]].append(new[key])
        mo[key[0]].append(old[key])
    opponents = args.opponents or sorted(gaps)
    print(f"{'opponent':44s} {'n':>3s} {'candidate':>13s} {'baseline':>13s} {'gap':>13s}")
    pooled, per_opponent = [], []
    for opp in opponents:
        if opp not in gaps:
            continue
        g = gaps[opp]
        pooled += g
        per_opponent.append(np.mean(g))
        print(f"{opp:44s} {len(g):3d} {np.mean(mn[opp]):+7.1f}±{se(mn[opp]):4.1f} {np.mean(mo[opp]):+7.1f}±{se(mo[opp]):4.1f} "
              f"{np.mean(g):+7.1f}±{se(g):4.1f}")
    if pooled:
        print(f"{'all (pooled pairs)':44s} {len(pooled):3d} {'':13s} {'':13s} {np.mean(pooled):+7.1f}±{se(pooled):4.1f}")


if __name__ == "__main__":
    main()
