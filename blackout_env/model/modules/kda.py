"""Kimi Delta Attention (KDA) in pure PyTorch: gated delta rule with per-channel decay.

Recurrence (one head; S is the d_k x d_v fast-weight memory, read as o = S^T q):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

alpha_t in (0, 1)^{d_k} is a data-dependent decay *per key channel* (Kimi Linear, arXiv
2510.26692; Gated DeltaNet, arXiv 2412.06464, has one scalar per head), beta_t in (0, 1) the
delta-rule write strength.  With ||k|| = 1 the transition is a contraction, so the state stays
bounded whatever the sequence length.  Written as "decay, then delta update":

    S'_{t-1} = Diag(alpha_t) S_{t-1};   S_t = S'_{t-1} + beta_t k_t (v_t - S'^T_{t-1} k_t)^T

Three equivalent forms, all on [B, H, T, .] tensors (``tests/test_kda.py`` checks they agree):

* ``kda_recurrent_reference`` -- the loop over t above, the definition;
* ``kda_step`` -- one t; what batch-1 rollout calls every tick;
* ``kda_chunk`` -- chunk-parallel training form.  Inside a chunk of C steps, with the cumulative
  log-decay g_r = sum_{i<=r} log alpha_i (per channel) and Gamma_r = Diag(exp g_r), the state is
  S_r = Gamma_r (S_0 + sum_{i<=r} Gamma_i^{-1} k_i u_i^T) for pseudo-values u (WY form).  The u
  solve a unit-lower-triangular system (I + A) U = beta * (V - (K Gamma) S_0), with
  A_ri = beta_r sum_c k_rc k_ic exp(g_rc - g_ic) for i < r, so every chunk's intra-chunk work
  (A, one triangular solve, the causal q.k matrix) is batched across chunks and only the
  d_k x d_v state is carried by a Python loop over chunks -- not over time steps.

Numerics.  exp(g_r - g_i) <= 1 for i <= r, but it is computed as a product of two factors, the
q/k side multiplied by exp(g - m) and the k side by exp(m - g), m = g at the chunk midpoint, so
each factor stays within exp(C * gate_bound / 2).  The decay is bounded per step,
log alpha in (-gate_bound, 0) (the sigmoid-bounded gate of fla's KDA ``lower_bound`` option
rather than Kimi's unbounded -exp(A) * softplus), which is what makes that factor finite:
fp32 needs C * gate_bound / 2 < ~80 (asserted); **fp16 overflows at ~11** (C = 64 needs
gate_bound < 0.35, or use C = 16), so under autocast the core runs in fp32 -- ``KDA.forward``
casts to fp32 around it; bf16 has fp32's range but 8 mantissa bits, too few for the triangular
solve.  The upper-triangle products that could exceed range are masked with ``torch.where``,
which passes no gradient into them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

MAX_HALF_RANGE_FP32 = 80.0  # exp(80) ~ 5.5e34 < fp32 max 3.4e38, with headroom for sums

__all__ = ["KDA", "kda_chunk", "kda_recurrent_reference", "kda_step", "unit_lower_inverse_apply"]


# ------------------------------------------------------------------------------ functional core
#
# Shapes: q, k [B, H, T, dk]; v [B, H, T, dv]; log_alpha [B, H, T, dk] (<= 0); beta [B, H, T];
# state S [B, H, dk, dv].  q and k are used as given (normalise / scale them before).
#
# ``reset`` [B, T] bool (optional): the state is zeroed *before* step t wherever reset[:, t] --
# a sequence start (e.g. a new match) in the middle of a batch row.  Every form handles it
# exactly: the loop by zeroing, the chunk form by masking the pairs that straddle a reset (the
# decays themselves stay bounded, so the fp32 range argument above is unchanged).


def kda_step(q, k, v, log_alpha, beta, state):
    """One step: q, k, log_alpha [B, H, dk]; v [B, H, dv]; beta [B, H]; state [B, H, dk, dv].
    Returns (o [B, H, dv], new state)."""
    s = state * log_alpha.exp().unsqueeze(-1)  # Diag(alpha) S
    residual = v - torch.einsum("bhk,bhkv->bhv", k, s)  # v - S'^T k
    s = s + beta[..., None, None] * k.unsqueeze(-1) * residual.unsqueeze(-2)
    return torch.einsum("bhk,bhkv->bhv", q, s), s


def kda_recurrent_reference(q, k, v, log_alpha, beta, state=None, reset=None):
    """The recurrence as a loop over t -- the definition the other forms are tested against."""
    B, H, T, dk = q.shape
    s = q.new_zeros(B, H, dk, v.shape[-1]) if state is None else state
    outs = []
    for t in range(T):
        if reset is not None:
            s = s * (~reset[:, t]).to(s.dtype)[:, None, None, None]
        o, s = kda_step(q[:, :, t], k[:, :, t], v[:, :, t], log_alpha[:, :, t], beta[:, :, t], s)
        outs.append(o)
    return torch.stack(outs, dim=2), s


def unit_lower_inverse_apply(A: torch.Tensor, rhs: torch.Tensor, method: str = "auto") -> torch.Tensor:
    """(I + A)^{-1} rhs for strictly lower-triangular A [..., C, C].

    ``solve``: ``torch.linalg.solve_triangular``.  ``doubling``: the Neumann series of the
    nilpotent -A summed by repeated squaring, (I - A)(I + A^2)(I + A^4)... -- exact after
    ceil(log2 C) matmul pairs, and ~40x faster than MPS's solve_triangular at our shapes (which
    does not run as a batched GPU kernel there).  ``auto``: doubling on MPS, solve elsewhere."""
    if method == "auto":
        method = "doubling" if A.device.type == "mps" else "solve"
    C = A.shape[-1]
    eye = torch.eye(C, dtype=A.dtype, device=A.device)
    if method == "solve":
        return torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
    if method != "doubling":
        raise ValueError(f"unknown method {method!r}")
    power = -A
    inverse = eye + power  # sum of (-A)^j, j < 2
    for _ in range(max(0, (C - 1).bit_length() - 1)):  # doubles the number of terms each pass
        power = power @ power
        inverse = inverse + inverse @ power
    return inverse @ rhs


def kda_chunk(q, k, v, log_alpha, beta, state=None, chunk_size: int = 64, gate_bound: float | None = None,
              solve: str = "auto", reset=None):
    """Chunk-parallel form; same result as ``kda_recurrent_reference``.  T need not be a multiple
    of ``chunk_size`` (the tail is padded with k = v = 0, beta = 0, no decay: a no-op).
    ``gate_bound`` (the per-step bound on -log_alpha), when given, is checked against fp32 range.
    ``solve`` picks the triangular solver (``unit_lower_inverse_apply``).

    ``reset`` [B, T] bool: zero the state before those steps.  Inside a chunk, a reset splits it
    into segments (seg_r = number of resets at positions <= r); only pairs (r, i) in the same
    segment interact (A, P), the start state reaches only segment 0, and the carried state keeps
    only the last segment's writes (plus the decayed start state if the chunk had no reset)."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    C = int(chunk_size)
    if gate_bound is not None and C * gate_bound / 2 > MAX_HALF_RANGE_FP32:
        raise ValueError(f"chunk_size * gate_bound / 2 = {C * gate_bound / 2:.0f} overflows fp32; use a smaller chunk")
    s = q.new_zeros(B, H, dk, dv) if state is None else state

    pad = (-T) % C
    if pad:
        q, k, v, log_alpha = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, log_alpha))
        beta = F.pad(beta, (0, pad))
        if reset is not None:
            reset = F.pad(reset, (0, pad), value=False)
    N = (T + pad) // C
    q, k, log_alpha = (x.reshape(B, H, N, C, dk) for x in (q, k, log_alpha))
    v = v.reshape(B, H, N, C, dv)
    beta = beta.reshape(B, H, N, C, 1)

    g = log_alpha.cumsum(dim=3)  # [B, H, N, C, dk], inclusive, <= 0, decreasing
    m = g[:, :, :, C // 2 : C // 2 + 1]  # per-chunk, per-channel reference point
    k_in = k * (m - g).exp()  # k_i Gamma_i^{-1}, relative to m
    k_out = k * (g - m).exp()  # k_r Gamma_r, relative to m
    q_out = q * (g - m).exp()

    idx = torch.arange(C, device=q.device)
    strict = idx[:, None] > idx[None, :]
    causal = idx[:, None] >= idx[None, :]
    zero = q.new_zeros(())
    k_start = k * g.exp()  # k_r Gamma_r: the start state's correction
    q_dec = q * g.exp()  # q_r Gamma_r: reads the decayed start state
    g_last = g[:, :, :, -1:]  # [B, H, N, 1, dk]
    k_to_end = k * (g_last - g).exp()  # k_i Gamma_C Gamma_i^{-1} (<= |k|)
    decay_end = g_last.squeeze(3).exp().unsqueeze(-1)  # [B, H, N, dk, 1]
    if reset is not None:
        seg = reset.reshape(B, 1, N, C).cumsum(dim=-1)  # segment id within the chunk
        same = seg.unsqueeze(-1) == seg.unsqueeze(-2)  # [B, 1, N, C, C]
        strict, causal = strict & same, causal & same
        first = (seg == 0).unsqueeze(-1).to(q.dtype)  # [B, 1, N, C, 1]: sees the start state
        k_start, q_dec = k_start * first, q_dec * first
        k_to_end = k_to_end * (seg == seg[..., -1:]).unsqueeze(-1).to(q.dtype)
        decay_end = decay_end * (seg[..., -1] == 0).to(q.dtype)[..., None, None]
    A = torch.where(strict, beta * (k_out @ k_in.transpose(-1, -2)), zero)  # [B, H, N, C, C]
    P = torch.where(causal, q_out @ k_in.transpose(-1, -2), zero)  # causal intra-chunk q.k

    # (I + A) [W | U~] = beta * [K Gamma | V]: W maps the chunk's start state to its correction,
    # U~ is the correction from the chunk's own values.  One batched solve for every chunk.
    rhs = beta * torch.cat((k_start, v), dim=-1)
    sol = unit_lower_inverse_apply(A, rhs, solve)
    W, U_own = sol[..., :dk], sol[..., dk:]

    outs = []
    for n in range(N):  # only the state crosses chunks
        u = U_own[:, :, n] - W[:, :, n] @ s  # [B, H, C, dv]
        outs.append(q_dec[:, :, n] @ s + P[:, :, n] @ u)
        s = decay_end[:, :, n] * s + k_to_end[:, :, n].transpose(-1, -2) @ u
    o = torch.stack(outs, dim=2).reshape(B, H, N * C, dv)[:, :, :T]
    return o, s


# ------------------------------------------------------------------------------ layer


class KDA(nn.Module):
    """Multi-head KDA layer, [B, T, d_model] -> [B, T, d_model], carrying a recurrent state.

    Projections as in Kimi Linear / Gated DeltaNet: q, k, v linear -> optional causal depthwise
    short conv -> SiLU; q, k L2-normalised per head (q also scaled by d_k^-1/2); beta = sigmoid
    of a linear; per-channel decay from a low-rank projection; output RMSNorm per head times a
    low-rank sigmoid gate, then a linear back to d_model.  No residual/norm around it: the
    caller's block owns those.

    State is a tuple ``(S [B, H, dk, dv], conv [B, conv_size - 1, 2*H*dk + H*dv] or None)``;
    ``None`` means zeros (sequence start).  ``forward`` uses the chunk form for T > 1 and the
    step form for T == 1 (``mode`` forces one); both return the state after the last step, so a
    sequence can be processed in pieces.

    Decay init: per-channel rates exp-spaced over ``decay_rate_range`` (fraction forgotten per
    step), so some channels remember ~1/rate_min steps from the start.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_dim: int = 64,
        v_head_dim: int | None = None,
        chunk_size: int = 64,
        use_short_conv: bool = True,
        conv_size: int = 4,
        gate_bound: float = 1.0,
        decay_rate_range: tuple[float, float] = (1e-3, 0.5),
    ) -> None:
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.dk, self.dv = head_dim, v_head_dim or head_dim
        self.chunk_size, self.gate_bound = chunk_size, float(gate_bound)
        if chunk_size * gate_bound / 2 > MAX_HALF_RANGE_FP32:
            raise ValueError("chunk_size * gate_bound / 2 must stay under ~80 for fp32")
        key_dim, value_dim = n_heads * self.dk, n_heads * self.dv

        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads)
        # low-rank decay projection (rank dv, as in Kimi Linear's f_proj); last layer starts at
        # zero so the initial decay is exactly the per-channel prior in decay_bias
        self.f_proj = nn.Sequential(nn.Linear(d_model, self.dv, bias=False), nn.Linear(self.dv, key_dim, bias=False))
        nn.init.zeros_(self.f_proj[1].weight)
        lo, hi = decay_rate_range
        rates = torch.exp(torch.linspace(math.log(lo), math.log(hi), key_dim))
        rates = rates.reshape(self.dk, n_heads).T.reshape(-1)  # every head gets the full spread
        frac = (-torch.log1p(-rates) / self.gate_bound).clamp(1e-6, 1 - 1e-6)  # -log alpha / bound
        self.decay_bias = nn.Parameter(torch.logit(frac))
        self.g_proj = nn.Sequential(nn.Linear(d_model, self.dv, bias=False), nn.Linear(self.dv, value_dim))
        self.o_norm = nn.RMSNorm(self.dv)
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)

        self.conv_size = conv_size if use_short_conv else 0
        if self.conv_size:
            # depthwise causal conv weights [channels, K], nn.Conv1d's default init; applied as K
            # shifted multiply-adds -- CPU Conv1d's depthwise backward was 77% of a training step
            channels = 2 * key_dim + value_dim
            bound = 1.0 / math.sqrt(conv_size)
            self.conv_weight = nn.Parameter(torch.empty(channels, conv_size).uniform_(-bound, bound))

    def initial_state(self, batch: int, device=None, dtype=torch.float32):
        s = torch.zeros(batch, self.n_heads, self.dk, self.dv, device=device, dtype=dtype)
        conv = None
        if self.conv_size:
            conv = torch.zeros(batch, self.conv_size - 1, self.conv_weight.shape[0], device=device, dtype=dtype)
        return s, conv

    def _short_conv(self, qkv: torch.Tensor, cache: torch.Tensor | None, reset: torch.Tensor | None = None):
        """Causal depthwise conv over time on [B, T, C] with a (K-1)-step cache [B, K-1, C].
        ``reset`` [B, T]: inputs from before a reset do not reach the steps from it on."""
        B, T, _ = qkv.shape
        K = self.conv_size
        if cache is None:
            cache = qkv.new_zeros(B, K - 1, qkv.shape[-1])
        x = torch.cat((cache.to(qkv.dtype), qkv), dim=1)  # [B, K-1+T, C]
        w = self.conv_weight.to(qkv.dtype)
        if reset is None:
            y = sum(x[:, j : j + T] * w[:, j] for j in range(K))  # tap K-1 = the current step
            return y, x[:, -(K - 1):]
        t = torch.arange(T, device=qkv.device)
        # index of the latest reset at or before t (-K: none, so the whole cache is visible)
        last = torch.where(reset, t, torch.full_like(t, -K)).cummax(dim=1).values  # [B, T]
        y = sum(x[:, j : j + T] * w[:, j] * (t + j - (K - 1) >= last).unsqueeze(-1).to(qkv.dtype) for j in range(K))
        src = torch.arange(T - (K - 1), T, device=qkv.device)  # the new cache's step indices
        keep = (src[None, :] >= last[:, -1:]).unsqueeze(-1).to(qkv.dtype)
        return y, x[:, -(K - 1):] * keep

    def forward(self, x: torch.Tensor, state=None, mode: str | None = None, reset: torch.Tensor | None = None):
        """x [B, T, d_model]; returns (y [B, T, d_model], state).  ``reset`` [B, T] bool zeroes
        the state (and the short-conv history) before those steps."""
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.dk, self.dv
        s, conv_cache = (None, None) if state is None else state

        qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        if self.conv_size:
            qkv, conv_cache = self._short_conv(qkv, conv_cache, reset)
        qkv = F.silu(qkv)
        q, k, v = qkv.split((H * dk, H * dk, H * dv), dim=-1)

        core = torch.promote_types(x.dtype, torch.float32)  # fp32 at least (module docstring)

        def heads(t, d):  # [B, T, H*d] -> [B, H, T, d]
            return t.reshape(B, T, H, d).transpose(1, 2).to(core)

        q = F.normalize(heads(q, dk), dim=-1) * dk ** -0.5
        k = F.normalize(heads(k, dk), dim=-1)
        v = heads(v, dv)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2).to(core)  # [B, H, T]
        log_alpha = -self.gate_bound * torch.sigmoid(self.f_proj(x) + self.decay_bias)
        log_alpha = heads(log_alpha, dk)
        s = s.to(core) if s is not None else None

        mode = mode or ("step" if T == 1 else "chunk")
        if mode == "chunk":
            o, s = kda_chunk(q, k, v, log_alpha, beta, s, self.chunk_size, reset=reset)
        elif mode == "step":
            if s is None:
                s = q.new_zeros(B, H, dk, dv)
            outs = []
            for t in range(T):
                if reset is not None:
                    s = s * (~reset[:, t]).to(s.dtype)[:, None, None, None]
                o_t, s = kda_step(q[:, :, t], k[:, :, t], v[:, :, t], log_alpha[:, :, t], beta[:, :, t], s)
                outs.append(o_t)
            o = torch.stack(outs, dim=2)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        o = self.o_norm(o.transpose(1, 2))  # [B, T, H, dv]
        gate = torch.sigmoid(self.g_proj(x)).reshape(B, T, H, dv)
        y = self.o_proj((o * gate).to(x.dtype).reshape(B, T, H * dv))
        return y, (s, conv_cache)
