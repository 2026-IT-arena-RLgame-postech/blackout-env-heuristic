"""How a distilled student turns its 8 direction logits into a move, selectable per match.

* ``iqn`` -- ``MyPolicy`` as trained: IQN quantile fractions drawn at random every forward, argmax
  of their mean.  At D1 v5, tau noise alone changed the argmax on ~26% of the ticks where the
  student reversed;
* ``fixed_tau`` -- the same argmax over fixed, evenly spaced quantile fractions: deterministic;
* ``tie_mean`` -- fixed tau, and every direction within ``TIE_MARGIN`` logits of the best counts:
  the move is their mean, like the teacher following a label *set* (a straight and a diagonal
  move that both make progress), unless they point apart, then the best alone.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from blackout_env.model.my_model import N_QUANTILES_DEFAULT, MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy

INFERENCE = ("iqn", "fixed_tau", "tie_mean")
TIE_MARGIN = 1.0  # logits; e^-1 = the tied move has at least ~37% of the best one's probability
_CANCELLING = 0.5  # |mean of the tied unit vectors| below this = tied moves point apart


class _FixedTau(nn.Module):
    """MyModel evaluated at fixed quantile fractions (i + 0.5) / n."""

    def __init__(self, net: MyModel, n: int = N_QUANTILES_DEFAULT):
        super().__init__()
        self.net = net
        self.register_buffer("tau", (torch.arange(n, dtype=torch.float32) + 0.5) / n)

    def forward(self, graphic, team_state, agent_states, **_):
        return self.net(graphic, team_state, agent_states, tau=self.tau.expand(graphic.shape[0], -1))


class _TieMeanPolicy(MyPolicy):
    def act(self, obs):
        moves = {}
        for agent, row in self.action_values(obs).items():
            tied = row >= row.max() - TIE_MARGIN
            mean = DIRECTION_VECTORS[tied].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            moves[agent] = (mean / norm).astype(np.float32) if norm >= _CANCELLING else DIRECTION_VECTORS[int(np.argmax(row))]
        return moves


def make_student(net: MyModel, inference: str = "iqn") -> MyPolicy:
    net.eval()
    if inference == "iqn":
        return MyPolicy(net)
    if inference == "fixed_tau":
        return MyPolicy(_FixedTau(net))
    if inference == "tie_mean":
        return _TieMeanPolicy(_FixedTau(net))
    raise ValueError(f"unknown inference {inference!r}; choose from {INFERENCE}")
