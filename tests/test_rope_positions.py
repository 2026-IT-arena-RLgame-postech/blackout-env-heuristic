"""2D RoPE: partial rotation, continuous positions, and where each token type sits."""

from __future__ import annotations

import math

import torch

from blackout_env.model.modules import RotaryEmbedding2D, build_grid_position_ids
from blackout_env.model.modules.rotary import apply_rope
from blackout_env.model.my_model import (
    GRID_H,
    GRID_W,
    N_ATTENTION_HEADS,
    N_UNITS,
    N_VISION_TOKENS,
    ROPE_BASE,
    MyModel,
)

HEAD_DIM = 256 // N_ATTENTION_HEADS  # 32


def test_every_frequency_pair_still_turns_across_the_grid():
    """base 10000 left 5 of 8 pairs under 9 degrees corner to corner -- i.e. carrying nothing."""
    rope = RotaryEmbedding2D(HEAD_DIM, base=ROPE_BASE)
    span = rope.freqs * (GRID_H - 1)
    assert float(span.max()) < 2 * math.pi  # fastest pair must not alias across the grid
    assert float(span.min()) > 0.5  # slowest pair still meaningfully rotates


def test_nope_band_is_left_untouched():
    rope = RotaryEmbedding2D(HEAD_DIM)
    cos, sin = rope(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0]))
    assert cos.shape == (2, rope.rope_dim) and rope.rope_dim == HEAD_DIM // 2

    x = torch.randn(1, N_ATTENTION_HEADS, 2, HEAD_DIM)
    out = apply_rope(x, cos, sin)
    assert torch.equal(out[..., rope.rope_dim :], x[..., rope.rope_dim :])
    assert not torch.allclose(out[..., : rope.rope_dim], x[..., : rope.rope_dim])


def test_rotation_preserves_norm_per_axis_half():
    rope = RotaryEmbedding2D(HEAD_DIM)
    cos, sin = rope(torch.tensor([3.5]), torch.tensor([0.25]))
    x = torch.randn(1, 1, 1, HEAD_DIM)
    out = apply_rope(x, cos, sin)
    half = rope.rope_dim // 2
    assert torch.allclose(out[..., :half].norm(), x[..., :half].norm(), atol=1e-5)
    assert torch.allclose(out[..., half : rope.rope_dim].norm(), x[..., half : rope.rope_dim].norm(), atol=1e-5)


def test_dot_product_depends_on_relative_position_only():
    """Holds for fractional offsets too -- that is what makes sub-cell positions meaningful."""
    rope = RotaryEmbedding2D(HEAD_DIM)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def score(pos_q, pos_k):
        cos_q, sin_q = rope(torch.tensor([pos_q]), torch.tensor([0.0]))
        cos_k, sin_k = rope(torch.tensor([pos_k]), torch.tensor([0.0]))
        return float((apply_rope(q, cos_q, sin_q) * apply_rope(k, cos_k, sin_k)).sum())

    assert math.isclose(score(1.0, 0.0), score(3.25, 2.25), rel_tol=1e-4)
    assert math.isclose(score(0.75, 0.0), score(4.75, 4.0), rel_tol=1e-4)
    assert not math.isclose(score(1.0, 0.0), score(1.5, 0.0), rel_tol=1e-3)


def test_sub_cell_offsets_produce_distinct_angles():
    rope = RotaryEmbedding2D(HEAD_DIM)
    cos_a, _ = rope(torch.tensor([2.0]), torch.tensor([0.0]))
    cos_b, _ = rope(torch.tensor([2.0625]), torch.tensor([0.0]))  # one 96-grid sub-cell over
    assert not torch.allclose(cos_a, cos_b, atol=1e-4)


def test_extra_tokens_sit_at_the_grid_centre():
    rows, cols = build_grid_position_ids(GRID_H, GRID_W, 3)
    assert rows[:N_VISION_TOKENS].max() == GRID_H - 1
    assert torch.allclose(rows[N_VISION_TOKENS:], torch.full((3,), (GRID_H - 1) / 2))
    assert torch.allclose(cols[N_VISION_TOKENS:], torch.full((3,), (GRID_W - 1) / 2))


def test_unit_tokens_take_their_real_position():
    net = MyModel()
    agent_states = torch.zeros(2, N_UNITS, 12)
    agent_states[0, 0, 0], agent_states[0, 0, 1] = -1.0, 1.0  # top-left corner of the map
    agent_states[0, 1, 0], agent_states[0, 1, 1] = 1.0, -1.0  # bottom-right
    agent_states[1, 0, 0], agent_states[1, 0, 1] = 0.0, 0.0   # centre

    rows, cols = net._token_grid_positions(agent_states)
    assert rows.shape == (2, N_VISION_TOKENS + N_UNITS + 2)

    unit0 = N_VISION_TOKENS
    assert (rows[0, unit0], cols[0, unit0]) == (0.0, 0.0)
    assert (rows[0, unit0 + 1], cols[0, unit0 + 1]) == (GRID_H - 1, GRID_W - 1)
    assert torch.allclose(rows[1, unit0], torch.tensor((GRID_H - 1) / 2), atol=1e-5)
    # vision tokens keep the fixed grid, and are identical across samples
    assert torch.equal(rows[0, :N_VISION_TOKENS], rows[1, :N_VISION_TOKENS])


def test_forward_runs_with_per_sample_positions():
    net = MyModel()
    agent_states = torch.rand(3, N_UNITS, 12) * 2 - 1
    q_values, *_ = net(torch.zeros(3, 13, 24, 24), torch.zeros(3, 4), agent_states, n_quantiles=4)
    assert q_values.shape == (3, N_UNITS, 8)

    moved = agent_states.clone()
    moved[:, 0, :2] += 0.3  # only unit 0's position changes
    assert not torch.allclose(q_values, net(torch.zeros(3, 13, 24, 24), torch.zeros(3, 4), moved, n_quantiles=4)[0])
