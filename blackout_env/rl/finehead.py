"""Hierarchical fine direction head (``GRPOConfig.fine_bins``): a discrete refinement of the angle offset.

A unit's action is (coarse bin k of K = 32, or STAND) and, for a non-STAND unit, a *fine* index j of ``nf`` (32)
sub-bins that tile the angle head's offset range [-S, +S] (S = the ``angle_scale`` buffer): heading = 2 pi k / K + c_j,
c_j = -S + (j + 1/2) w, w = 2 S / nf (S = pi / K by default -> w ~ 0.35 deg, 32 x 32 = 1024 directions).

``FineHead`` is owned by the trainer, not by ``MyModel`` (base checkpoints stay loadable).  Input: the SwiGLU
features the angle head reads (``FeatTap`` captures them with a forward pre-hook on ``angle_out``), a learned embedding
of the chosen coarse bin; output: nf logits = prior + residual.  prior_j = -(c_j - mu_k)^2 / (2 (tau w)^2), mu_k = the
angle head's (deterministic) offset of the chosen bin; the residual MLP's last layer is zero-initialised, so at init the
greedy fine index is the sub-bin nearest the distilled offset (error <= w / 2) and the policy stays ~today's.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeatTap:
    """Forward pre-hook on the (single) ``angle_out`` Linear of a MyModel: ``.feat`` = its last input [B, N, H]
    (the very tensor, graph and dtype included).  ``of(net)`` is idempotent (one tap per net)."""

    def __init__(self, net: nn.Module):
        heads = [m for m in net.modules() if isinstance(getattr(m, "angle_out", None), nn.Module)]
        assert len(heads) == 1, "FeatTap needs a net with exactly one angle head (MyModel with angle_head)"
        self.head, self.module = heads[0], heads[0].angle_out
        self.feat = None
        self.module.register_forward_pre_hook(self._hook)

    def _hook(self, m, inp):
        self.feat = inp[0]

    @classmethod
    def of(cls, net: nn.Module) -> "FeatTap":
        for m in net.modules():
            tap = m.__dict__.get("_fine_tap")
            if tap is not None:
                return tap
        tap = cls(net)
        tap.head.__dict__["_fine_tap"] = tap
        return tap

    @property
    def n_coarse(self) -> int:
        return self.module.out_features

    @property
    def hidden(self) -> int:
        return self.module.in_features

    @property
    def scale(self) -> float:
        return float(self.head.angle_scale)


class FineHead(nn.Module):
    def __init__(self, hidden: int, n_coarse: int = 32, nf: int = 32, scale: float = math.pi / 32, tau: float = 0.5,
                 emb: int = 32, mlp: int = 128):
        super().__init__()
        self.nf, self.n_coarse = int(nf), int(n_coarse)
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.ln = nn.LayerNorm(hidden)
        self.emb = nn.Embedding(n_coarse, emb)
        self.fc1 = nn.Linear(hidden + emb, mlp)
        self.fc2 = nn.Linear(mlp, nf)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @classmethod
    def from_state(cls, sd: dict) -> "FineHead":
        """Rebuild from a saved state dict (shapes give the sizes; ``tau`` / ``scale`` are buffers)."""
        h = cls(sd["ln.weight"].shape[0], sd["emb.weight"].shape[0], sd["fc2.weight"].shape[0], float(sd["scale"]),
                float(sd["tau"]), sd["emb.weight"].shape[1], sd["fc1.weight"].shape[0])
        h.load_state_dict(sd)
        return h

    @property
    def width(self) -> float:
        return 2.0 * float(self.scale) / self.nf

    @property
    def centres(self) -> torch.Tensor:
        w = 2.0 * self.scale / self.nf
        return -self.scale + (torch.arange(self.nf, device=self.scale.device, dtype=self.scale.dtype) + 0.5) * w

    def prior(self, mu: torch.Tensor) -> torch.Tensor:
        w = 2.0 * self.scale / self.nf
        return -((self.centres - mu.unsqueeze(-1)) ** 2) / (2.0 * (self.tau * w) ** 2)

    def forward(self, feat: torch.Tensor, k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """feat [..., H] (float32), chosen coarse bin k [...] (long, < n_coarse), its offset mu [...] (detached) -> logits [..., nf]."""
        x = torch.cat([self.ln(feat), self.emb(k)], -1)
        return self.prior(mu) + self.fc2(torch.nn.functional.silu(self.fc1(x)))

    @torch.no_grad()
    def pick(self, off: torch.Tensor, feat: torch.Tensor, chosen: torch.Tensor, rows=None, temperature: float = 1.0,
             generator=None):
        """Apply the head to coarse choices ``chosen`` [n, 5] (K = STAND).  ``off`` [n, 5, K] canonical offsets, ``feat`` [n, 5, H].
        Fine index = argmax everywhere, except ``rows`` (index tensor over n) which sample softmax(logits / temperature).
        Returns (``off`` with the chosen bin's entry of non-STAND units replaced by c_j -- a new tensor --, j [n, 5] long, 0 for STAND)."""
        k = off.shape[-1]
        ns = chosen < k
        kc = chosen.clamp(max=k - 1)
        mu = off.gather(-1, kc.unsqueeze(-1))[..., 0]
        lg = self(feat.float(), kc, mu)
        j = lg.argmax(-1)
        if rows is not None and len(rows):
            p = torch.softmax(lg[rows] / temperature, -1)
            j[rows] = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).view(p.shape[:-1])
        c = self.centres[j]
        off2 = off.scatter(-1, kc.unsqueeze(-1), torch.where(ns, c, mu).unsqueeze(-1))
        return off2, torch.where(ns, j, torch.zeros_like(j))

    @torch.no_grad()
    def greedy_offsets(self, q: torch.Tensor, off: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Greedy decode: ``off`` with the argmax coarse bin's offset replaced by the argmax fine centre."""
        return self.pick(off, feat, q.argmax(-1))[0]
