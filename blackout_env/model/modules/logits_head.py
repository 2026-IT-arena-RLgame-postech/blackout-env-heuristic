"""
Plain per-unit policy logits (P5, docs/policy_v2_design_ko.md §10): the IQN head without tau.

``IQNHead`` evaluates ``value_head(state * e(tau))`` once per sampled quantile fraction tau, where
``e(tau) = ReLU(tau_proj(cos(pi * i * tau)))``, and the distillation student reads the mean over
32 of them.  For distillation nothing needs the quantile axis, so this head keeps the same
``value_head`` (RMSNorm-SwiGLU block, then the output Linear) and replaces ``e(tau)`` by one
learned per-channel input scale ``in_scale``: one evaluation instead of n_quantiles.

The keys are those of IQNHead minus ``tau_proj`` plus ``in_scale``, so an IQN state converts by
copying ``value_head`` and setting ``in_scale`` from the tau embedding (``state_from_iqn``):

* one tau ``t``: ``in_scale = e(t)`` reproduces the IQN head at that tau exactly;
* the mean over n fixed taus (what ``inference.fixed_tau`` and training read) is a mean of n
  *nonlinear* evaluations (RMSNorm and SwiGLU sit between the scale and the output), which no
  single scale reproduces exactly -- an exact fold would need an n-times-wider SwiGLU, i.e. the
  IQN cost again.  ``state_from_iqn`` uses the mean embedding ``mean_j e(tau_j)`` (on D6's eval
  rows: argmax agreement 99.5%), and ``train/distill/head_fit.py`` fits the head on cached trunk
  features to get closer.

Angle head (P11, ``angle_bins`` = K > 0): beside the K bin logits, one offset per bin read off the
same SwiGLU features (``angle_out``), ``tanh``-squashed to +-``angle_scale`` = (``angle_arc_bins`` +
1/2) bin widths -- half a bin by default (``pi / K``), wider so that a bin next to the teacher's
can still point at its exact angle (an arc label makes the arg max any bin of the arc): bin k's
heading is ``2 pi k / K + offset_k``.  ``angle_out`` starts at zero weights and bias, so a head
given one keeps its logits and executes exactly its bin centres until it is trained.  Its keys
(``angle_out.``, and the ``angle_scale`` buffer) are the only new ones.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .ffn_block import SwiGLUBlock

IQN_ONLY_PREFIX = "tau_proj."  # IQNHead keys that a LogitsHead does not have
ANGLE_PREFIX = "angle_"  # the angle head's keys (P11): angle_out.weight/bias, angle_scale


def angle_scale_of(angle_bins: int, arc_bins: int = 0) -> float:
    """The offsets' half range in radians: (arc_bins + 1/2) bin widths."""
    return (arc_bins + 0.5) * 2 * math.pi / angle_bins


def arc_bins_of_scale(scale: float, angle_bins: int) -> int:
    return int(round(float(scale) * angle_bins / (2 * math.pi) - 0.5))


class LogitsHead(nn.Module):
    def __init__(self, hidden_size: int, n_actions: int, angle_bins: int = 0, angle_arc_bins: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_scale = nn.Parameter(torch.ones(hidden_size))
        self.value_head = nn.Sequential(
            SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, n_actions),
        )
        self.angle_bins = int(angle_bins)
        # created last (parameter order of a head without it unchanged), zero: offsets 0 at init
        self.angle_out = nn.Linear(hidden_size, angle_bins) if angle_bins else None
        if self.angle_out is not None:
            nn.init.zeros_(self.angle_out.weight)
            nn.init.zeros_(self.angle_out.bias)
            self.register_buffer("angle_scale", torch.tensor(angle_scale_of(angle_bins, angle_arc_bins)))

    def forward(self, state: torch.Tensor, angle: bool = False):
        """state [B, N, hidden] -> logits [B, N, n_actions]; ``angle`` (an angle head only): also
        the per-bin offsets [B, N, K] in radians, ``(logits, offsets)``."""
        if not angle:
            return self.value_head(state * self.in_scale)
        if self.angle_out is None:
            raise ValueError("this LogitsHead has no angle head (angle_bins=0)")
        features = self.value_head[0](state * self.in_scale)
        offsets = torch.tanh(self.angle_out(features)) * self.angle_scale
        return self.value_head[1](features), offsets


def iqn_tau_embedding(state: dict, tau: torch.Tensor, prefix: str = "") -> torch.Tensor:
    """IQNHead's ``e(tau)`` from its state dict: tau [n] -> [n, hidden]."""
    weight, bias = state[f"{prefix}tau_proj.0.weight"], state[f"{prefix}tau_proj.0.bias"]
    cos_ids = torch.arange(weight.shape[1], dtype=torch.float32, device=weight.device)
    angles = math.pi * cos_ids * tau.to(weight.device, torch.float32).reshape(-1, 1)
    return torch.relu(torch.cos(angles) @ weight.float().T + bias.float()).to(weight.dtype)


def state_from_iqn(state: dict, tau: torch.Tensor, prefix: str = "") -> dict:
    """An IQNHead's (sub-)state with keys under ``prefix`` -> the LogitsHead state under the same
    prefix: ``value_head`` copied, ``in_scale`` = the mean of ``e(tau)`` over ``tau`` [n] (exact
    for a single tau; see the module docstring).  Every other key is passed through."""
    out = {k: v for k, v in state.items() if not k.startswith(prefix + IQN_ONLY_PREFIX)}
    out[f"{prefix}in_scale"] = iqn_tau_embedding(state, tau, prefix).mean(dim=0)
    return out
