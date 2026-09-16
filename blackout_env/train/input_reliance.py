"""
Does the Q-network actually use the map? A cheap probe logged during training.

Every run so far (Run 7, 8, 9) spent its first 7-10k steps not using the graphic input at all,
and the only visible symptom was grad_norm/graphic_encoder sitting near 1e-4 -- which alone cannot
tell "the encoder has nothing left to learn" from "nothing downstream reads it". Probing
checkpoints on real dataset observations told them apart (docs/offline_pretrain_runs.md, Run 9):

    step 5k (Run 8 and Run 9 alike)
      shuffle graphics across the batch -> Q moves 0.002-0.003, greedy action unchanged 100%
      shuffle agent/team vectors instead -> Q moves ~1.0
      SPR latent: cosine similarity between DIFFERENT samples = 1.0000

SPR had collapsed to a constant (its loss ~1e-6, so no gradient), Q was fitted from the vector
tokens alone, and with both paths dead the encoder received nothing. It recovered only once
larger TD errors arrived (Run 9: exactly when on-policy data entered the batch).

Definitions (all on the training batch, own-team rows, IQN at the median tau so sampling noise
does not count as sensitivity):
  graphic_shuffle_dq   mean |Q(graphic of another sample) - Q| / mean across-action Q spread.
                       The map-derived features (derived channels, each unit's wall patch) are
                       recomputed from the swapped graphic, so this measures reliance on the map
                       as a whole.
  vector_shuffle_dq    same, swapping agent_states + team_state and keeping the graphic.
  graphic_shuffle_argmax_kept  fraction of units whose greedy action survives the graphic swap
                       (1.0 = the policy ignores the map).
  spr_latent_cos       mean off-diagonal cosine similarity of the SPR latent (1.0 = collapsed).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def _offdiag_mean_cosine(x: torch.Tensor) -> float:
    x = F.normalize(x.float().reshape(x.shape[0], -1), dim=-1)
    n = x.shape[0]
    if n < 2:
        return float("nan")
    sim = x @ x.T
    return float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def input_reliance_probe(
    net: torch.nn.Module,
    graphic: torch.Tensor,
    team_state: torch.Tensor,
    agent_states: torch.Tensor,
    own_rows: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    seed: int = 0,
) -> dict[str, float]:
    """See the module docstring. `own_rows(q [B, N_UNITS, A], agent_states)` -> [B, N_TEAM, A]."""
    batch = graphic.shape[0]
    tau = torch.full((batch, 1), 0.5, device=graphic.device)
    generator = torch.Generator().manual_seed(seed)
    # A derangement (no sample paired with itself), so a small batch cannot hide reliance.
    perm = (torch.arange(batch) + 1 + torch.randint(0, max(1, batch - 1), (1,), generator=generator)) % batch
    perm = perm.to(graphic.device)

    q, _, _, spr_latent, _ = net(graphic, team_state, agent_states, tau=tau)
    q_graphic, *_ = net(graphic[perm], team_state, agent_states, tau=tau)
    q_vector, *_ = net(graphic, team_state[perm], agent_states[perm], tau=tau)

    own = own_rows(q.float(), agent_states)
    own_graphic = own_rows(q_graphic.float(), agent_states)
    own_vector = own_rows(q_vector.float(), agent_states[perm])
    spread = own.std(dim=-1).mean().clamp_min(1e-8)

    return {
        "graphic_shuffle_dq": float((own_graphic - own).abs().mean() / spread),
        "vector_shuffle_dq": float((own_vector - own).abs().mean() / spread),
        "graphic_shuffle_argmax_kept": float((own_graphic.argmax(-1) == own.argmax(-1)).float().mean()),
        "spr_latent_cos": _offdiag_mean_cosine(spr_latent),
    }
