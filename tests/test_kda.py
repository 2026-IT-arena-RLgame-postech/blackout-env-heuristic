"""KDA layer (model/modules/kda.py): the chunk-parallel, step and loop forms agree, gradients are
right, state carries across calls, and a tiny model actually remembers across a delay."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from blackout_env.model.modules.kda import (
    KDA,
    kda_chunk,
    kda_recurrent_reference,
    kda_step,
    unit_lower_inverse_apply,
)


def _inputs(B=2, H=3, T=37, dk=8, dv=5, dtype=torch.float64, seed=0, strong_decay=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, dk, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, dv, generator=g, dtype=dtype)
    scale = 1.0 if strong_decay else 0.2
    log_alpha = -scale * torch.rand(B, H, T, dk, generator=g, dtype=dtype)
    beta = torch.rand(B, H, T, generator=g, dtype=dtype)
    state = torch.randn(B, H, dk, dv, generator=g, dtype=dtype)
    return q, k, v, log_alpha, beta, state


@pytest.mark.parametrize("chunk", [1, 4, 8, 16, 64])
@pytest.mark.parametrize("strong_decay", [False, True])
@pytest.mark.parametrize("solve", ["solve", "doubling", "tree", "blocked"])
def test_chunk_matches_loop_reference(chunk, strong_decay, solve):
    q, k, v, la, beta, s0 = _inputs(strong_decay=strong_decay)  # T = 37: never a chunk multiple
    o_ref, s_ref = kda_recurrent_reference(q, k, v, la, beta, s0)
    o, s = kda_chunk(q, k, v, la, beta, s0, chunk_size=chunk, solve=solve)
    torch.testing.assert_close(o, o_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(s, s_ref, rtol=1e-10, atol=1e-10)


def test_chunk_matches_loop_reference_fp32_at_the_decay_bound():
    """fp32, chunk 64, every step at the full gate bound (log alpha = -1): the midpoint-referenced
    exp factors reach e^32 and must still cancel."""
    q, k, v, _, beta, s0 = _inputs(B=2, H=2, T=130, dk=16, dv=16, dtype=torch.float32)
    la = torch.full_like(q, -1.0)
    o_ref, s_ref = kda_recurrent_reference(q, k, v, la, beta, s0)
    o, s = kda_chunk(q, k, v, la, beta, s0, chunk_size=64, gate_bound=1.0)
    assert torch.isfinite(o).all()
    torch.testing.assert_close(o, o_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s, s_ref, rtol=1e-4, atol=1e-4)


def test_step_by_step_matches_loop_reference():
    q, k, v, la, beta, s = _inputs()
    o_ref, s_ref = kda_recurrent_reference(q, k, v, la, beta, s)
    outs = []
    for t in range(q.shape[2]):
        o_t, s = kda_step(q[:, :, t], k[:, :, t], v[:, :, t], la[:, :, t], beta[:, :, t], s)
        outs.append(o_t)
    torch.testing.assert_close(torch.stack(outs, dim=2), o_ref)
    torch.testing.assert_close(s, s_ref)


@pytest.mark.parametrize("solve", ["solve", "doubling", "tree", "blocked"])
def test_gradcheck_chunk_form(solve):
    q, k, v, la, beta, s0 = _inputs(B=1, H=1, T=7, dk=3, dv=2)
    args = [x.clone().requires_grad_(True) for x in (q, k, v, la, beta, s0)]
    assert torch.autograd.gradcheck(lambda *a: kda_chunk(*a, chunk_size=4, solve=solve), args)


def test_doubling_inverse_matches_triangular_solve():
    g = torch.Generator().manual_seed(0)
    for C in (1, 2, 3, 5, 16, 17, 64):
        A = torch.randn(4, C, C, generator=g, dtype=torch.float64).tril(-1) * 0.3
        rhs = torch.randn(4, C, 3, generator=g, dtype=torch.float64)
        torch.testing.assert_close(
            unit_lower_inverse_apply(A, rhs, "doubling"), unit_lower_inverse_apply(A, rhs, "solve")
        )


def test_gradcheck_layer():
    torch.manual_seed(0)
    layer = KDA(6, n_heads=2, head_dim=3, chunk_size=2, conv_size=3).double()
    x = torch.randn(1, 5, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: layer(x)[0], (x,))


@pytest.mark.parametrize("use_short_conv", [True, False])
def test_layer_chunk_step_and_split_agree(use_short_conv):
    """Whole sequence (chunk form) == one step at a time (step form) == two halves with the
    state (conv cache included) handed over."""
    torch.manual_seed(0)
    layer = KDA(16, n_heads=2, head_dim=8, chunk_size=8, use_short_conv=use_short_conv).double()
    x = torch.randn(3, 29, 16, dtype=torch.float64)

    y, (s, conv) = layer(x)
    y_loop, (s_loop, _) = layer(x, mode="step")
    torch.testing.assert_close(y_loop, y)
    torch.testing.assert_close(s_loop, s)

    state, ys = None, []
    for t in range(x.shape[1]):  # what a rollout does: T = 1 per call
        y_t, state = layer(x[:, t : t + 1], state)
        ys.append(y_t)
    torch.testing.assert_close(torch.cat(ys, dim=1), y)

    y1, state = layer(x[:, :13])
    y2, (s2, conv2) = layer(x[:, 13:], state)
    torch.testing.assert_close(torch.cat((y1, y2), dim=1), y)
    torch.testing.assert_close(s2, s)
    if use_short_conv:
        torch.testing.assert_close(conv2, conv)


def test_zero_state_equals_no_state():
    torch.manual_seed(0)
    layer = KDA(16, n_heads=2, head_dim=8, chunk_size=4)
    x = torch.randn(2, 10, 16)
    y, _ = layer(x)
    y0, _ = layer(x, layer.initial_state(2))
    torch.testing.assert_close(y0, y)


def test_rejects_chunk_that_overflows_fp32():
    with pytest.raises(ValueError):
        KDA(8, n_heads=1, head_dim=8, chunk_size=256, gate_bound=1.0)


# ------------------------------------------------------------------------------ memory task

N_VALUES, N_DISTRACTORS, DELAY = 8, 8, 30
QUERY = N_VALUES + N_DISTRACTORS
STEPS = 300  # ~10 s on CPU; accuracy is already 1.0 by ~100 (the short conv spans 4 steps, not 30)


def _recall_batch(n: int, g: torch.Generator):
    """A value token at t = 0, DELAY random distractors, then a query: answer the value."""
    value = torch.randint(0, N_VALUES, (n,), generator=g)
    x = torch.randint(N_VALUES, N_VALUES + N_DISTRACTORS, (n, DELAY + 2), generator=g)
    x[:, 0] = value
    x[:, -1] = QUERY
    return x, value


class _Recall(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.emb = nn.Embedding(QUERY + 1, d)
        self.norm = nn.RMSNorm(d)
        self.kda = KDA(d, n_heads=2, head_dim=16, chunk_size=16)
        self.out = nn.Linear(d, N_VALUES)

    def forward(self, x):
        h = self.emb(x)
        y, _ = self.kda(self.norm(h))
        return self.out(h[:, -1] + y[:, -1])  # only the query step is read out


def test_learns_delayed_recall():
    """Chance is 1/8, and nothing but the recurrent state links t = 0 to the query step."""
    torch.manual_seed(0)
    net = _Recall()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        x, y = _recall_batch(64, g)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = _recall_batch(1024, g)
        accuracy = (net(x).argmax(-1) == y).float().mean().item()
    assert accuracy > 0.9, accuracy



@pytest.mark.parametrize("solve", ["tree", "blocked", "solve"])
def test_chunk_is_stable_on_repeated_keys(solve):
    """A unit standing still feeds (nearly) the same key every tick; with beta near 1 the plain
    doubling solve blew up in fp32 (error ~1e9 at chunk 64) -- the stable solvers must not."""
    torch.manual_seed(0)
    B, H, T, dk, dv = 2, 2, 64, 16, 8
    k = F.normalize(torch.ones(B, H, T, dk) + 1e-3 * torch.randn(B, H, T, dk), dim=-1)
    q, v = torch.randn(B, H, T, dk), torch.randn(B, H, T, dv)
    log_alpha = -0.01 * torch.rand(B, H, T, dk)
    beta = torch.full((B, H, T), 0.95)
    ref, _ = kda_recurrent_reference(*(x.double() for x in (q, k, v, log_alpha, beta)))
    out, _ = kda_chunk(q, k, v, log_alpha, beta, chunk_size=64, solve=solve)
    assert (out.double() - ref).abs().max() < 1e-3


@pytest.mark.parametrize("C", [1, 5, 8, 13, 16, 37, 64])
def test_tree_inverse_matches_triangular_solve_at_any_size(C):
    g = torch.Generator().manual_seed(C)
    A = torch.tril(torch.randn(3, C, C, generator=g, dtype=torch.float64) * 0.3, -1)
    rhs = torch.randn(3, C, 4, generator=g, dtype=torch.float64)
    assert torch.allclose(unit_lower_inverse_apply(A, rhs, "tree"), unit_lower_inverse_apply(A, rhs, "solve"), atol=1e-10)
