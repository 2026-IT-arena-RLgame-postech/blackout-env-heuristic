import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard (single-axis) RoPE: x, cos, sin all share their last dim, which must be even."""
    return x * cos + _rotate_half(x) * sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    2D-aware partial RoPE matching RotaryEmbedding2D's cos/sin layout: the leading
    cos.shape[-1] dims of each head are [row half | col half], each half independently
    duplicated-paired for its own rotation (see RotaryEmbedding2D._axis_angles), and every dim
    beyond that is left alone -- the NoPE band.

    Row and col must be rotated as two SEPARATE 1D RoPEs on their own half — rotate_half-ing
    the rotated block at once (a single global split) would pair a row dim with a col dim under
    mismatched angles, which is not a rotation (breaks norm preservation).

    The NoPE band exists because every dot product in this trunk is otherwise position-modulated,
    including the unit tokens now that they carry real positions. Attention also has to match on
    content alone ("the token holding a battery, wherever it is"), and dims that never rotate are
    where that can happen without fighting the rotation. Partial rotation is the usual setting for
    this (GPT-NeoX's rotary_pct, and the halves several later models use); with a 6x6 grid, a few
    frequencies are plenty for position, so the rest of head_dim is better spent on content.

    x        : [B, heads, T, head_dim]
    cos, sin : [T, rope_dim] or [B, T, rope_dim], rope_dim <= head_dim (broadcasts over heads)
    """
    rope_dim = cos.shape[-1]
    if cos.dim() == 3:  # per-sample positions: [B, T, rope_dim] -> broadcast over heads
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    half = rope_dim // 2
    x_row, x_col = x_rot[..., :half], x_rot[..., half:]
    cos_row, cos_col = cos[..., :half], cos[..., half:]
    sin_row, sin_col = sin[..., :half], sin[..., half:]
    return torch.cat(
        [_apply_rope_1d(x_row, cos_row, sin_row), _apply_rope_1d(x_col, cos_col, sin_col), x_pass],
        dim=-1,
    )


def build_grid_position_ids(
    grid_h: int, grid_w: int, n_extra_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Row/col grid coordinates for a [grid_h * grid_w vision tokens] + [n_extra_tokens] sequence,
    in the same row-major order GraphicEncoder's `x.flatten(2)` produces (col varies fastest).
    Float, not int: positions are evaluated continuously (see RotaryEmbedding2D), so a caller
    can hand a unit token its real sub-cell position.

    Extra tokens get the grid CENTRE rather than (0, 0). They used to sit at position 0, which
    is not "no position" once other tokens rotate -- it is the top-left corner, giving a global
    summary token a spatial bias toward one corner of the map. The centre makes its relative
    offset to the grid symmetric. Callers that have a real position for an extra token (unit
    tokens do) should overwrite that slot instead.
    """
    rows = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h)
    centre_row = torch.full((n_extra_tokens,), (grid_h - 1) / 2.0)
    centre_col = torch.full((n_extra_tokens,), (grid_w - 1) / 2.0)
    return torch.cat([rows, centre_row]), torch.cat([cols, centre_col])


class RotaryEmbedding2D(nn.Module):
    """
    2D rotary position embedding for a sequence mixing grid tokens (graphic) with tokens that
    have a position of their own (units) and ones that do not (team_state, SPR CLS). `rope_dim`
    of each head is rotated -- its first half by row position, its second by col position, so a
    query/key dot product depends on their *relative* (row, col) offset the way ordinary 1D RoPE
    depends on relative sequence offset -- and the remaining dims are left unrotated (see
    apply_rope for why that NoPE band is wanted).

    Positions are continuous, not grid indices: the angle is position * frequency, so a token
    sitting between two cells simply gets the in-between angle, and RoPE's relative-offset
    property still holds (rotations compose by adding angles). That is what lets a unit token
    say "I am three quarters of the way across this cell" rather than being rounded to it.

    `base` must suit the grid, not a text sequence. The usual 10000 spreads frequencies over
    thousands of positions; across this 6x6 grid its 5 slowest pairs (of 8) rotate by under 9
    degrees corner to corner -- numerically indistinguishable from no rotation, i.e. most of the
    rotated dims carried no position at all. Small bases keep every pair meaningful; the fastest
    pair must still stay under a full turn across the grid to avoid aliasing (at base 10, 5.0 rad
    over 5 cells).

    (The 8-pair figure above dates from a wider rotated band. With MyModel's defaults --
    rope_dim = head_dim / 2 = 8 at hidden 128 -- each axis has 2 pairs, at frequencies 1 and
    base**-0.5 per cell.)
    """

    def __init__(self, head_dim: int, base: float = 10.0, rope_dim: int | None = None) -> None:
        super().__init__()
        rope_dim = head_dim // 2 if rope_dim is None else rope_dim
        assert rope_dim % 4 == 0, (
            "rope_dim must be divisible by 4 (split into row/col halves, each needing an even "
            "number of dims to form rotation pairs)"
        )
        assert 0 < rope_dim <= head_dim, "rope_dim must fit inside head_dim"
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.axis_dim = rope_dim // 2  # dims given to each axis (row, col)
        freqs = 1.0 / (base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("freqs", freqs, persistent=False)

    def _axis_angles(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions[..., None].float() * self.freqs  # [..., T, axis_dim/2]
        return torch.cat([angles, angles], dim=-1)  # [..., T, axis_dim], paired for _rotate_half

    def forward(self, row_ids: torch.Tensor, col_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        row_ids, col_ids: [T] or [B, T] continuous grid coordinates.
        Returns cos, sin, each [..., T, rope_dim].
        """
        angles = torch.cat([self._axis_angles(row_ids), self._axis_angles(col_ids)], dim=-1)
        return angles.cos(), angles.sin()
