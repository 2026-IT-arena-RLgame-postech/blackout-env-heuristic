"""
Heuristic policy identity and strength, for weighting behaviour cloning by how good the
demonstrator is.

Rows store a small integer policy index (POLICY_IDS order, -1 = not a heuristic / unknown) so a
dataset can be split or weighted by who played it. The BC weight of a policy is its expected
score against an average opponent of the collection mixture under the full-pool Elo fit
(reports/elo_active_20260916, SE 42-65), normalised so the mixture-weighted mean weight is 1 --
the BC loss keeps its overall scale while V19 (Elo 2059) counts about 2x and V1 (1192) about 0.25x.
"""

from __future__ import annotations

import numpy as np

# Full-pool Bradley-Terry fit, 64 s truncated matches (reports/elo_active_20260916/summary.txt).
ELO_20260916 = {
    "strategic_v19": 2059, "strategic_v17": 1832, "strategic_v18": 1831, "strategic_v12": 1542,
    "strategic_v8": 1529, "strategic_v9": 1519, "strategic_v4": 1500, "strategic_v16": 1490,
    "strategic_v15": 1482, "strategic_v7": 1480, "strategic_v13": 1472, "strategic_v10": 1472,
    "strategic_v4_near": 1462, "strategic_v3": 1456, "strategic_v11": 1443, "strategic_v6": 1390,
    "strategic_v5": 1317, "strategic_v2": 1314, "strategic_v14": 1220, "strategic_v1": 1192,
}

POLICY_IDS: tuple[str, ...] = tuple(sorted(ELO_20260916, key=lambda p: (len(p), p)))
UNKNOWN_POLICY = -1


def policy_index(policy_id: str | None) -> int:
    return POLICY_IDS.index(policy_id) if policy_id in POLICY_IDS else UNKNOWN_POLICY


def bc_weight_table(mixture_weights: dict[str, float]) -> np.ndarray:
    """[len(POLICY_IDS) + 1] BC weight per policy index; the last entry (index -1) is 1.0 for rows
    with no recorded policy, so older datasets clone exactly as before."""
    names = [p for p in mixture_weights if p in ELO_20260916 and mixture_weights[p] > 0]
    share = np.array([mixture_weights[p] for p in names], dtype=np.float64)
    share /= share.sum()
    elo = np.array([ELO_20260916[p] for p in names], dtype=np.float64)
    reference = float((share * elo).sum())
    expected = {p: 1.0 / (1.0 + 10.0 ** ((reference - ELO_20260916[p]) / 400.0)) for p in ELO_20260916}
    scale = float(sum(s * expected[p] for s, p in zip(share, names)))
    table = np.ones(len(POLICY_IDS) + 1, dtype=np.float32)
    for i, p in enumerate(POLICY_IDS):
        table[i] = expected[p] / scale
    return table
