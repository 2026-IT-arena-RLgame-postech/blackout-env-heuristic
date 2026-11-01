"""Heuristic-teacher labels for PPO's early-iteration distillation term (``PPOConfig.teach``).

``TeachLabeler`` steps a (stateful) heuristic teacher on the learner's own team in every game the learner plays, on the
same live ``BatchSim`` state and at the same decisions (a DAgger-style relabelling of the learner's own states), and turns
its 5 world moves into the learner's canonical-frame action index (K direction bins, STAND = K), exactly like the
distillation labels (``collect._direction`` + ``mirror_bin`` for team B).  With heuristic workers it is a
``ParallelHeuristic`` (``start`` ships the states before the network forward, ``finish`` collects after it, so the teacher
overlaps the GPU); without, the in-process ``HeuristicPolicy``.
"""

from __future__ import annotations

import numpy as np

from blackout_env.model.direction_bins import mirror_bin, vector_to_bin
from blackout_env.sim.batch import HeuristicPolicy
from blackout_env.train.distill.act_teacher import ZERO_MOVE


def moves_to_bins(mv: np.ndarray, team_b: np.ndarray, k: int, stand: bool = True) -> np.ndarray:
    """World moves [n, 5, 2] -> canonical action indices [n, 5] int8 (bin 2 pi i / k; team B's view mirrored; zero move -> STAND = k)."""
    b = vector_to_bin(mv, k)
    b = np.where(np.asarray(team_b)[:, None], mirror_bin(b, k), b)
    zero = np.linalg.norm(mv, axis=-1) < ZERO_MOVE
    return np.where(zero, k if stand else b, b).astype(np.int8)


class TeachLabeler:
    def __init__(self, descriptor: str, seed: int, workers=None):
        if workers is not None:
            from blackout_env.rl.hpool import ParallelHeuristic

            self.pol = ParallelHeuristic(descriptor, seed, workers)
        else:
            self.pol = HeuristicPolicy(descriptor, seed)

    def start(self, bsim, idx, tm: int):
        """Begin labelling games ``idx`` for team ``tm``."""
        if hasattr(self.pol, "start"):
            return self.pol.start(bsim, idx, np.full(len(idx), tm))
        return self.pol.moves(bsim, idx, (tm,), None)[tm]

    def finish(self, h, tm: int, width: int) -> np.ndarray:
        """-> [n, 5] int8 labels for a learner head of ``width`` outputs (K bins [+ STAND])."""
        mv = self.pol.finish(h) if hasattr(self.pol, "finish") else h
        k = width - (width % 2)
        return moves_to_bins(mv, np.full(len(mv), tm == 1), k, stand=width != k)

    def forget(self, games) -> None:
        from blackout_env.rl.grpo import forget

        forget(self.pol, games)
