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
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .ffn_block import SwiGLUBlock

IQN_ONLY_PREFIX = "tau_proj."  # IQNHead keys that a LogitsHead does not have


class LogitsHead(nn.Module):
    def __init__(self, hidden_size: int, n_actions: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_scale = nn.Parameter(torch.ones(hidden_size))
        self.value_head = nn.Sequential(
            SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state [B, N, hidden] -> logits [B, N, n_actions]."""
        return self.value_head(state * self.in_scale)


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
