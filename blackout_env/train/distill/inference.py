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

import warnings

import numpy as np
import torch
from torch import nn

from blackout_env.model.my_model import N_QUANTILES_DEFAULT, MyModel
from blackout_env.model.my_policy import ACTION_VECTORS, DIRECTION_VECTORS, STAND, MyPolicy
from blackout_env.train.distill.dataset import N_TEAM

INFERENCE = ("iqn", "fixed_tau", "tie_mean")
TIE_MARGIN = 1.0  # logits; e^-1 = the tied move has at least ~37% of the best one's probability
_CANCELLING = 0.5  # |mean of the tied unit vectors| below this = tied moves point apart


class _OwnQ(nn.Module):
    """MyModel's Q-values for the own team's rows only (0-4 of a canonical batch)."""

    def __init__(self, net: MyModel):
        super().__init__()
        self.net = net

    def forward(self, graphic, team_state, agent_states, tau):
        return self.net(graphic, team_state, agent_states, tau=tau, q_units=N_TEAM)[0]


class _FastNet(nn.Module):
    """``MyModel`` for MyPolicy: ``_OwnQ`` under inference_mode, traced at batch 1 on first use.

    ``tau`` = None draws fresh quantile fractions every call (IQN as trained); a tensor fixes
    them.  Returns ``(q_values,)`` so MyPolicy's ``q_values, *_ = net(...)`` still unpacks."""

    def __init__(self, net: MyModel, tau: torch.Tensor | None = None, n_quantiles: int = N_QUANTILES_DEFAULT):
        super().__init__()
        self.own = _OwnQ(net.eval())
        self.fixed_tau, self.n_quantiles = tau, n_quantiles
        self._traced = None

    def forward(self, graphic, team_state, agent_states, **_):
        batch = graphic.shape[0]
        tau = self.fixed_tau.expand(batch, -1) if self.fixed_tau is not None else torch.rand(batch, self.n_quantiles)
        with torch.inference_mode():
            if batch != 1:  # the trace is batch-1 only
                return (self.own(graphic, team_state, agent_states, tau),)
            if self._traced is None:
                with warnings.catch_warnings():  # deprecation, and GroupNorm's shape check (batch 1 is fixed)
                    warnings.simplefilter("ignore")
                    self._traced = torch.jit.trace(self.own, (graphic, team_state, agent_states, tau), check_trace=False)
            return (self._traced(graphic, team_state, agent_states, tau),)


def fixed_tau(n: int = N_QUANTILES_DEFAULT) -> torch.Tensor:
    """Evenly spaced quantile fractions (i + 0.5) / n, shape [1, n]."""
    return ((torch.arange(n, dtype=torch.float32) + 0.5) / n).unsqueeze(0)


class _TieMeanPolicy(MyPolicy):
    def act(self, obs):
        moves = {}
        for agent, row in self.action_values(obs).items():
            if int(np.argmax(row)) == STAND:  # a --stand-action student choosing to stay
                moves[agent] = ACTION_VECTORS[STAND]
                continue
            row = row[:len(DIRECTION_VECTORS)]
            tied = row >= row.max() - TIE_MARGIN
            mean = DIRECTION_VECTORS[tied].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            moves[agent] = (mean / norm).astype(np.float32) if norm >= _CANCELLING else DIRECTION_VECTORS[int(np.argmax(row))]
        return moves


def load_net(state: dict) -> MyModel:
    """A MyModel shaped like ``state``: its width, and 8 or 9 (--stand-action) outputs."""
    net = MyModel(hidden_size=state["token_type_emb.weight"].shape[1], n_actions=state["q_head.value_head.1.weight"].shape[0])
    net.load_state_dict(state)
    return net


def make_student(net: MyModel, inference: str = "iqn") -> MyPolicy:
    net.eval()
    if inference == "iqn":
        return MyPolicy(_FastNet(net))
    if inference == "fixed_tau":
        return MyPolicy(_FastNet(net, tau=fixed_tau()))
    if inference == "tie_mean":
        return _TieMeanPolicy(_FastNet(net, tau=fixed_tau()))
    raise ValueError(f"unknown inference {inference!r}; choose from {INFERENCE}")
