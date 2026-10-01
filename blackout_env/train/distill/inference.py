"""How a distilled student turns its direction logits into a move, selectable per match.

The logits are 8 compass directions or, for a K-bin student (P4, ``--action-bins``), K heading
bins (model/direction_bins.py), with STAND last when there is one; K is read off the net's output
width, so every mode below and the traced ``_FastNet`` work at any K unchanged.

* ``iqn`` -- ``MyPolicy`` as trained: IQN quantile fractions drawn at random every forward, argmax
  of their mean.  At D1 v5, tau noise alone changed the argmax on ~26% of the ticks where the
  student reversed;
* ``fixed_tau`` -- the same argmax over fixed, evenly spaced quantile fractions: deterministic;
* ``tie_mean`` -- fixed tau, and every direction within ``TIE_MARGIN`` logits of the best counts:
  the move is their mean, like the teacher following a label *set* (a straight and a diagonal
  move that both make progress), unless they point apart, then the best alone;
* ``arc_mean`` -- fixed tau; the best bin and its neighbours on either side, as long as each stays
  within ``ARC_MARGIN`` logits of the best (one contiguous arc, never a second peak): the move is
  their probability-weighted circular mean.  Meant for K-bin students, whose neighbouring bins
  share the mass of one heading: the mean is smoother than the arg max, which jitters between
  them.  At K = 8 it is ``tie_mean`` restricted to adjacent directions.

A temporal student (``MyModel(temporal=True)``) works in every mode: ``_FastNet`` then takes and
returns the memory, ``MyPolicy`` carries it across ticks (one memory per team, so a mirror match
against itself is two independent players) and clears it in ``reset()`` -- call that at every
match start, as ``play_labelled_match`` does.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import nn

from blackout_env.model.my_model import N_QUANTILES_DEFAULT, MyModel
from blackout_env.model.direction_bins import bin_vectors, bins_of_width
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.distill.dataset import N_TEAM

INFERENCE = ("iqn", "fixed_tau", "tie_mean", "arc_mean")
TIE_MARGIN = 1.0  # logits; e^-1 = the tied move has at least ~37% of the best one's probability
ARC_MARGIN = 1.0  # logits: arc_mean's bins
_CANCELLING = 0.5  # |mean of the tied unit vectors| below this = tied moves point apart


class _OwnQ(nn.Module):
    """MyModel's Q-values for the own team's rows only (0-4 of a canonical batch)."""

    def __init__(self, net: MyModel):
        super().__init__()
        self.net = net

    def forward(self, graphic, team_state, agent_states, tau):
        return self.net(graphic, team_state, agent_states, tau=tau, q_units=N_TEAM)[0]


class _OwnQStep(nn.Module):
    """``_OwnQ`` for a temporal MyModel: one tick, memory in and out as plain tensors (traceable)."""

    def __init__(self, net: MyModel):
        super().__init__()
        self.net = net

    def forward(self, graphic, team_state, agent_states, tau, kda_state, conv_state):
        out = self.net(graphic, team_state, agent_states, tau=tau, q_units=N_TEAM, memory=(kda_state, conv_state))
        return out[0], out[-1][0], out[-1][1]


class _FastNet(nn.Module):
    """``MyModel`` for MyPolicy: ``_OwnQ`` under inference_mode, traced at batch 1 on first use.

    ``tau`` = None draws fresh quantile fractions every call (IQN as trained); a tensor fixes
    them.  Returns ``(q_values,)`` so MyPolicy's ``q_values, *_ = net(...)`` still unpacks.

    A temporal net: ``forward(..., memory=None)`` returns ``(q_values, memory)`` (None = zeros,
    the start of a match); the batch-1 trace takes and returns the memory tensors."""

    def __init__(self, net: MyModel, tau: torch.Tensor | None = None, n_quantiles: int = N_QUANTILES_DEFAULT):
        super().__init__()
        net = net.eval()
        self.is_temporal = bool(getattr(net, "is_temporal", False))
        self.own = _OwnQStep(net) if self.is_temporal else _OwnQ(net)
        self.fixed_tau, self.n_quantiles = tau, n_quantiles
        self._traced = None

    def forward(self, graphic, team_state, agent_states, memory=None, **_):
        batch = graphic.shape[0]
        tau = self.fixed_tau.expand(batch, -1) if self.fixed_tau is not None else torch.rand(batch, self.n_quantiles)
        if self.is_temporal:
            return self._step(graphic, team_state, agent_states, tau, memory)
        with torch.inference_mode():
            if batch != 1:  # the trace is batch-1 only
                return (self.own(graphic, team_state, agent_states, tau),)
            if self._traced is None:
                with warnings.catch_warnings():  # deprecation, and GroupNorm's shape check (batch 1 is fixed)
                    warnings.simplefilter("ignore")
                    self._traced = torch.jit.trace(self.own, (graphic, team_state, agent_states, tau), check_trace=False)
            return (self._traced(graphic, team_state, agent_states, tau),)

    def _step(self, graphic, team_state, agent_states, tau, memory):
        batch = graphic.shape[0]
        with torch.inference_mode():
            if memory is None:
                memory = self.own.net.initial_memory(batch, device=graphic.device)
            args = (graphic, team_state, agent_states, tau, *memory)
            if batch != 1:
                q, *memory = self.own(*args)
                return q, tuple(memory)
            if self._traced is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._traced = torch.jit.trace(self.own, args, check_trace=False)
            q, *memory = self._traced(*args)
            return q, tuple(memory)


def fixed_tau(n: int = N_QUANTILES_DEFAULT) -> torch.Tensor:
    """Evenly spaced quantile fractions (i + 0.5) / n, shape [1, n]."""
    return ((torch.arange(n, dtype=torch.float32) + 0.5) / n).unsqueeze(0)


_ZERO = np.zeros(2, dtype=np.float32)


class _TieMeanPolicy(MyPolicy):
    def act(self, obs):
        moves = {}
        for agent, row in self.action_values(obs).items():
            bins, _ = bins_of_width(len(row))
            if int(np.argmax(row)) == bins:  # a --stand-action student choosing to stay
                moves[agent] = _ZERO.copy()
                continue
            vectors = bin_vectors(bins)
            row = row[:bins]
            tied = row >= row.max() - TIE_MARGIN
            mean = vectors[tied].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            moves[agent] = (mean / norm).astype(np.float32) if norm >= _CANCELLING else vectors[int(np.argmax(row))]
        return moves


def arc_mean_move(row: np.ndarray, margin: float = ARC_MARGIN) -> np.ndarray:
    """``arc_mean`` for one unit's logits: STAND -> the zero move, else the circular mean of the
    contiguous arc around the best bin (see the module docstring)."""
    bins, _ = bins_of_width(len(row))
    best = int(np.argmax(row))
    if best == bins:
        return _ZERO.copy()
    logits = row[:bins].astype(np.float64)
    floor = logits[best] - margin
    arc = [best]
    for step in (1, -1):  # grow each side while it stays near the best (at most all the way round)
        k = best
        while len(arc) < bins:
            k = (k + step) % bins
            if logits[k] < floor or k in arc:
                break
            arc.append(k)
    weights = np.exp(logits[arc] - logits[best])
    mean = (bin_vectors(bins)[arc].astype(np.float64) * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:  # a full circle of ties: no heading wins
        return bin_vectors(bins)[best].copy()
    return (mean / norm).astype(np.float32)


class _ArcMeanPolicy(MyPolicy):
    def act(self, obs):
        return {agent: arc_mean_move(row) for agent, row in self.action_values(obs).items()}


def load_net(state: dict, temporal: bool | None = None) -> MyModel:
    """A MyModel shaped like ``state``: its width, its outputs (8 or K bins, + STAND), and temporal
    memory iff ``state`` has ``temporal.`` keys (``temporal=None``).  ``temporal=True`` also
    accepts a stateless state: the warm start (identity-initialised memory, the same Q-values);
    see ``MyModel.from_state_dict``."""
    return MyModel.from_state_dict(state, temporal=temporal)


def make_student(net: MyModel, inference: str = "iqn") -> MyPolicy:
    net.eval()
    if inference == "iqn":
        return MyPolicy(_FastNet(net))
    if inference == "fixed_tau":
        return MyPolicy(_FastNet(net, tau=fixed_tau()))
    if inference == "tie_mean":
        return _TieMeanPolicy(_FastNet(net, tau=fixed_tau()))
    if inference == "arc_mean":
        return _ArcMeanPolicy(_FastNet(net, tau=fixed_tau()))
    raise ValueError(f"unknown inference {inference!r}; choose from {INFERENCE}")
