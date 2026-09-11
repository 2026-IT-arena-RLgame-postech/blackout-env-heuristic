"""
IQN (Implicit Quantile Networks, Dabney et al. 2018, arXiv:1806.06923) value head.

Chosen over C51 for this project specifically because C51 needs a fixed [Vmin, Vmax] support
decided up front, and this game's reward balance (team score + kill/death + item pickups —
see Unity's RewardConfig/GameBalanceConfig) is still being tuned; IQN represents the return
distribution implicitly via a quantile function instead, so there's no fixed range to keep in
sync with the reward config. It also composes with QMIX for free: since QMixer's mixing
*weights* depend only on the state (not on the value being mixed), the exact same QMixer call
can be reused once per sampled quantile fraction to produce a team-level quantile function too
(DFAC-style, Sun et al. 2021) — see qmix_trainer.py.

Samples n_quantiles fractions tau ~ U(0,1), embeds each via the standard IQN cosine basis, and
multiplies (Hadamard product) that embedding into a shared state embedding before the final
value layers — so the value head genuinely depends on tau, not just on state.
"""

import math

import torch
import torch.nn as nn

from .ffn_block import SwiGLUBlock


class IQNHead(nn.Module):
    def __init__(self, hidden_size: int, n_actions: int, n_cos: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cos = n_cos

        cos_ids = torch.arange(n_cos, dtype=torch.float32).view(1, 1, n_cos)
        self.register_buffer("cos_ids", cos_ids, persistent=False)

        self.tau_proj = nn.Sequential(nn.Linear(n_cos, hidden_size), nn.ReLU())
        self.value_head = nn.Sequential(
            SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
        n_quantiles: int,
        tau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state : [B, N, hidden]   N independent "states" sharing one batch row (e.g. units)
        tau    : [B, n_quantiles] in [0, 1); sampled uniformly if not given

        Returns
        -------
        quantile_values : [B, N, n_quantiles, n_actions]
        tau             : [B, n_quantiles]
        """
        B = state.shape[0]
        device = state.device
        if tau is None:
            tau = torch.rand(B, n_quantiles, device=device)

        angles = math.pi * self.cos_ids * tau.unsqueeze(-1)  # [B, n_quantiles, n_cos]
        tau_emb = self.tau_proj(torch.cos(angles))  # [B, n_quantiles, hidden]

        merged = state.unsqueeze(2) * tau_emb.unsqueeze(1)  # [B, N, n_quantiles, hidden]
        quantile_values = self.value_head(merged)  # [B, N, n_quantiles, n_actions]
        return quantile_values, tau


def quantile_huber_loss(
    pred: torch.Tensor,
    pred_tau: torch.Tensor,
    target: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise quantile Huber loss (IQN eq. 3): every predicted quantile sample is compared
    against every target quantile sample, asymmetrically weighted by how far pred_tau is from
    "this error was negative" — this is what makes the predictions spread out to trace the
    actual return distribution's quantile function instead of collapsing to the mean.

    pred     : [B, Q]   online quantile values (Q samples, tau = pred_tau)
    pred_tau : [B, Q]
    target   : [B, Q']  target quantile values (Q' samples, e.g. from a separately-sampled tau')

    Returns
    -------
    per_sample_loss : [B]
    """
    B, Q = pred.shape
    Qp = target.shape[1]

    delta = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, Q, Q'] = target_j - pred_i
    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta.pow(2),
        kappa * (delta.abs() - 0.5 * kappa),
    )
    weight = (pred_tau.unsqueeze(2) - (delta < 0).float()).abs()  # [B, Q, Q']
    loss = (weight * huber / kappa).mean(dim=2)  # average over target samples -> [B, Q]
    return loss.mean(dim=1)  # average over pred samples -> [B]
