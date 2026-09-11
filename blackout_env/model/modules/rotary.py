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
    2D-aware RoPE matching RotaryEmbedding2D's cos/sin layout: the last dim is [row half |
    col half], each half independently duplicated-paired for its own rotation (see
    RotaryEmbedding2D._axis_angles). Row and col must be rotated as two SEPARATE 1D RoPEs on
    their own half — rotate_half-ing the full head_dim at once (a single global split) would
    pair a row dim with a col dim under mismatched angles, which is not a rotation (breaks
    norm preservation). So this splits x/cos/sin at head_dim//2 first and rotates each half
    on its own.

    x        : [B, heads, T, head_dim]
    cos, sin : [T, head_dim] (broadcasts over batch and heads)
    """
    half = x.shape[-1] // 2
    x_row, x_col = x[..., :half], x[..., half:]
    cos_row, cos_col = cos[..., :half], cos[..., half:]
    sin_row, sin_col = sin[..., :half], sin[..., half:]
    return torch.cat(
        [_apply_rope_1d(x_row, cos_row, sin_row), _apply_rope_1d(x_col, cos_col, sin_col)],
        dim=-1,
    )


def build_grid_position_ids(
    grid_h: int, grid_w: int, n_extra_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Row/col grid coordinates for a [grid_h * grid_w vision tokens] + [n_extra_tokens] sequence,
    in the same row-major order GraphicEncoder's `x.flatten(2)` produces (col varies fastest).
    Extra (non-spatial) tokens get (row=0, col=0) — since RoPE's rotation angle is proportional
    to position, angle 0 means cos=1/sin=0, i.e. RotaryEmbedding2D leaves them unrotated.
    """
    rows = torch.arange(grid_h).repeat_interleave(grid_w)
    cols = torch.arange(grid_w).repeat(grid_h)
    pad = torch.zeros(n_extra_tokens, dtype=torch.long)
    return torch.cat([rows, pad]), torch.cat([cols, pad])


class RotaryEmbedding2D(nn.Module):
    """
    2D rotary position embedding for a sequence mixing spatial (graphic) and non-spatial
    (unit/global) tokens in one attention trunk. head_dim is split into two halves: the first
    half's rotation angle is driven by row position, the second half's by col position — so a
    query/key pair's dot product ends up depending on their *relative* (row, col) offset, the
    way ordinary 1D RoPE makes it depend on relative sequence offset.

    Tokens outside the grid (unit/global) should be passed row=col=0 (see
    build_grid_position_ids), which zeroes their rotation angle and leaves them untouched —
    RoPE is therefore a purely spatial signal for the graphic tokens; which-token-type
    information is carried separately by MyModel's token-type embedding.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, (
            "head_dim must be divisible by 4 for 2D RoPE (split into row/col halves, each "
            "needing an even number of dims to form rotation pairs)"
        )
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # dims given to each axis (row, col)
        freqs = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("freqs", freqs, persistent=False)

    def _axis_angles(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions[:, None].float() * self.freqs[None, :]  # [T, half_dim/2]
        return torch.cat([angles, angles], dim=-1)  # [T, half_dim], paired for _rotate_half

    def forward(self, row_ids: torch.Tensor, col_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """row_ids, col_ids: [T] int64 grid coordinates. Returns cos, sin, each [T, head_dim]."""
        angles = torch.cat([self._axis_angles(row_ids), self._axis_angles(col_ids)], dim=-1)
        return angles.cos(), angles.sin()
