"""
SPR (Self-Predictive Representations, Schwarzer et al. 2021) auxiliary objective — applied
only to MyModel's vision latent: the SPR CLS token's output through spr_head (earlier versions
mean-pooled the vision tokens, hence the "pooled" names below). Project decision: the
unit/team-state vector modality has no natural "future map state" analog the way the semantic
map does, so SPR is scoped to vision only.

Given the current pooled vision embedding and the joint action actually taken (all 10 units,
both teams — the map's future depends on everyone on it, not just "my" team), a small latent
transition model recurrently predicts the next K frames' pooled vision embeddings in an
open loop. Each predicted step is compared, via a BYOL/SimSiam-style stop-gradient cosine
loss, against an EMA target encoder's *real* embedding of that future frame — this regularizes
the shared encoder to capture map dynamics, not just the current instant's Q-values.

Only the online branch gets the extra `predictor` head (SimSiam's asymmetry, needed to avoid
the trivial collapsed solution where the encoder just outputs a constant); the target-side
`target_projector` is EMA-updated from `projector`, never trained by gradient descent
directly — see QMIXTrainer's EMA update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPRPredictor(nn.Module):
    def __init__(self, hidden_size: int, n_actions: int, n_units: int = 10) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, hidden_size)
        self.unit_id_emb = nn.Embedding(n_units, hidden_size)
        self.n_units = n_units

        # Combines (action, unit-identity) PAIRS nonlinearly *before* pooling across units —
        # a plain "sum/mean of action_emb(a_i)" (or even "action_emb(a_i) + unit_id_emb(i)"
        # summed) is permutation-invariant to which unit did what, since it linearly
        # decomposes into independent per-unit terms; the pooled result of "unit 0 goes east,
        # unit 1 goes west" would then be identical to "unit 0 goes west, unit 1 goes east"
        # even though those lead to very different next map states. Passing each unit's
        # (action, id) pair through a shared nonlinear MLP first breaks that decomposition
        # (Deep Sets style, Zaheer et al. 2017), so the sum genuinely depends on the
        # assignment, not just the action histogram.
        self.action_unit_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

        self.transition = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        # EMA-updated mirror of `projector` — never trained by backprop directly. Starts as an
        # exact copy (not an independent random init) so the very first SPR target is a
        # genuine EMA target of the online projector, not an unrelated random function that
        # then takes ~hundreds of updates (at the default ema_tau=0.99) to wash out.
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.target_projector.load_state_dict(self.projector.state_dict())
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    def step(self, pooled_vision: torch.Tensor, joint_action_idx: torch.Tensor) -> torch.Tensor:
        """
        pooled_vision    : [B, hidden]
        joint_action_idx : [B, n_units] int64 — all units' chosen directions that step
        Returns predicted next pooled_vision [B, hidden] (residual update on the current one).
        """
        assert joint_action_idx.shape[1] == self.n_units, (
            f"expected {self.n_units} units' worth of actions, got {joint_action_idx.shape[1]} "
            "— the mean over units below would silently average over the wrong count otherwise"
        )
        act_emb = self.action_emb(joint_action_idx)  # [B, n_units, hidden]
        id_emb = self.unit_id_emb.weight.unsqueeze(0).expand(act_emb.shape[0], -1, -1)  # [B, n_units, hidden]
        per_unit = self.action_unit_mlp(torch.cat([act_emb, id_emb], dim=-1))  # [B, n_units, hidden]
        action_ctx = per_unit.mean(dim=1)  # [B, hidden]

        delta = self.transition(torch.cat([pooled_vision, action_ctx], dim=-1))
        return pooled_vision + delta

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        predicted : [B, hidden]  online transition-model prediction (gradients flow through it)
        target    : [B, hidden]  EMA target encoder's real embedding of that frame (no grad
                    expected from the caller, but detached here too for safety)
        Returns per-sample (1 - cosine similarity), [B].
        """
        online = F.normalize(self.predictor(self.projector(predicted)), dim=-1)
        target_proj = F.normalize(self.target_projector(target.detach()), dim=-1)
        return 1.0 - (online * target_proj).sum(dim=-1)
