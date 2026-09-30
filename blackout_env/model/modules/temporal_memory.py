"""Per-slot temporal memory between MyModel's trunk and its Q head (design option "b2",
docs/kda_temporal_design_ko.md section 4).

Input: the trunk's post-attention vector tokens of each frame, [B, T, 11, D] -- the 10 unit
tokens (canonical order: own 0-4, enemy 5-9) and the team_state token.  Each slot is its own
sequence over time through one KDA layer (``modules/kda.py``); the 11 slots of a frame then
exchange what they remember through one small attention layer, and the result is added back
to the tokens:

    m    = KDA(RMSNorm(h + slot_emb))                   per slot, over T (weights shared)
    z    = m + SlotAttn(RMSNorm(m + slot_emb))          across the 11 slots, per frame
    h'   = h + out_proj(RMSNorm(z))                     out_proj zero-initialised

Why share the KDA weights across slots: the 11 sequences are the same *kind* of thing (a unit's
or the game's trajectory), a shared layer sees 11x the data per parameter, costs the same at
rollout (the slots are just batch rows), and slot identity is not lost -- the trunk's tokens
already carry ``vec_slot_emb`` and ``slot_emb`` is added again here, so own/enemy/global slots
can still use the shared memory differently.

Why the cross-slot attention: only the own units' rows reach the Q head, so without it the
enemy slots' and the team_state slot's memory (who the opponent is, who has been camping)
could never affect a Q-value.  It is per frame (no time mixing), 11 tokens, so cheap.

Why zero-init ``out_proj``: h' = h exactly at init, so a temporal net warm-started from a
stateless checkpoint produces the stateless net's Q-values bit for bit, and training starts
from the distilled student rather than from a perturbed one.  The gradient w.r.t. out_proj is
non-zero from the first step, so the branch switches on as soon as it helps.

State (``memory``): a tuple ``(S [B, 11, H, dk, dv], conv [B, 11, K-1, 2*H*dk + H*dv])`` of
fp32 tensors -- the KDA state and short-conv history of every slot; ``None`` = zeros (start of
a match).  Memory is a function of the observation history only (no previous actions).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kda import KDA

__all__ = ["TemporalMemory"]


class TemporalMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_slots: int = 11,
        n_heads: int = 4,
        head_dim: int = 64,
        chunk_size: int = 64,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads (the slot attention's heads)")
        self.hidden_size, self.n_slots, self.n_heads = hidden_size, n_slots, n_heads
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, hidden_size))
        nn.init.normal_(self.slot_emb, std=0.02)
        self.norm_in = nn.RMSNorm(hidden_size)
        self.kda = KDA(hidden_size, n_heads=n_heads, head_dim=head_dim, chunk_size=chunk_size,
                       use_short_conv=True, conv_size=conv_size)
        self.mix_norm = nn.RMSNorm(hidden_size)
        self.mix_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.mix_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_norm = nn.RMSNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def initial_memory(self, batch: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        s, conv = self.kda.initial_state(batch * self.n_slots, device=device)
        return s.reshape(batch, self.n_slots, *s.shape[1:]), conv.reshape(batch, self.n_slots, *conv.shape[1:])

    def _slot_attention(self, m: torch.Tensor) -> torch.Tensor:
        """Attention across the slots of each frame: m [..., S, D] -> [..., S, D]."""
        *lead, S, D = m.shape
        H = self.n_heads
        x = self.mix_norm(m + self.slot_emb)
        q, k, v = self.mix_qkv(x).reshape(-1, S, 3, H, D // H).permute(2, 0, 3, 1, 4)  # [3, N, H, S, d]
        o = F.scaled_dot_product_attention(q, k, v)  # [N, H, S, d]
        return self.mix_out(o.transpose(1, 2).reshape(*lead, S, D))

    def forward(self, tokens: torch.Tensor, memory=None, reset: torch.Tensor | None = None):
        """tokens [B, T, S, D]; memory as in the module docstring or None; reset [B, T] bool
        (zero the memory before those frames).  Returns (tokens' [B, T, S, D], memory)."""
        B, T, S, D = tokens.shape
        x = self.norm_in(tokens + self.slot_emb).transpose(1, 2).reshape(B * S, T, D)
        state = None
        if memory is not None:
            s, conv = memory
            state = (s.reshape(B * S, *s.shape[2:]), conv.reshape(B * S, *conv.shape[2:]))
        if reset is not None:
            reset = reset.to(torch.bool).repeat_interleave(S, dim=0)  # [B*S, T], slot-major within b
        m, (s, conv) = self.kda(x, state, reset=reset)
        m = m.reshape(B, S, T, D).transpose(1, 2)  # [B, T, S, D]
        z = m + self._slot_attention(m)
        out = tokens + self.out_proj(self.out_norm(z))
        return out, (s.reshape(B, S, *s.shape[1:]), conv.reshape(B, S, *conv.shape[1:]))
