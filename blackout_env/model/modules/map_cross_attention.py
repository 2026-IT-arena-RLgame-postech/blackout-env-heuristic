"""Map-token cross-attention for MyModel (``MyModel(map_xattn=...)``, docs/policy_v2_design_ko.md §25).

The 11 vector tokens (10 units + team_state) attend to a fixed set of map tokens -- keys/values
only, never updated -- taken from the GraphicEncoder's own feature maps:

* ``"f3"``: its final 6x6 x 128 map (the features the trunk's 36 vision tokens are made from);
* ``"f2"``: its 12x12 x 64 map after the stride-2 stage (a finer grid the trunk never sees).

Each map token is ``Linear(C -> d)`` of its cell + a learned 2-D position embedding (row + col);
a unit's query adds the same embedding at the cell it stands in (team_state: none), so a query can
find "the cells around me" by position.  One cross block (pre-norm multi-head attention, no FFN)
sits after each of ``n_layers`` trunk layers, spread over the depth.

Every block's output projection is zero-initialised: a net with the path freshly added computes
exactly what it did without it (``MyModel.from_state_dict(old_state, map_xattn="f3")`` = old net).
All keys live under ``map_xattn.``; the source and depth are read back from their shapes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MAP_SOURCES = {"f3": (6, 128), "f2": (12, 64)}  # name -> (grid side, channels)


def source_of_channels(channels: int, grid: int) -> str:
    for name, spec in MAP_SOURCES.items():
        if spec == (grid, channels):
            return name
    raise ValueError(f"no map-token source with grid {grid} and {channels} channels")


class MapCrossBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h = heads
        self.q_norm = nn.RMSNorm(d)
        self.kv_norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_proj = nn.Linear(d, 2 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x: torch.Tensor, q_pos: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """x [B, N, d] (residual stream of the queries), q_pos [B, N, d], mem [B, S, d] -> the update [B, N, d]."""
        b, n, d = x.shape
        s = mem.shape[1]
        q = self.q_proj(self.q_norm(x) + q_pos).view(b, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv_proj(self.kv_norm(mem)).view(b, s, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(a.transpose(1, 2).reshape(b, n, d))


class MapCrossAttention(nn.Module):
    def __init__(self, d: int, heads: int, n_layers: int, source: str = "f3") -> None:
        super().__init__()
        if source not in MAP_SOURCES:
            raise ValueError(f"map_xattn must be one of {tuple(MAP_SOURCES)}, got {source!r}")
        self.source = source
        self.grid, channels = MAP_SOURCES[source]
        self.in_proj = nn.Linear(channels, d)
        self.pos_row = nn.Parameter(torch.randn(self.grid, d) * 0.02)
        self.pos_col = nn.Parameter(torch.randn(self.grid, d) * 0.02)
        self.blocks = nn.ModuleList(MapCrossBlock(d, heads) for _ in range(n_layers))

    def memory(self, feat: torch.Tensor, agent_states: torch.Tensor, n_queries: int):
        """feat [B, C, g, g] -> (map tokens [B, g*g, d], query positions [B, n_queries, d]): units
        (the first agent_states.shape[1] queries) get their cell's embedding, the rest zero."""
        g = self.grid
        pos = (self.pos_row[:, None, :] + self.pos_col[None, :, :]).reshape(g * g, -1)
        mem = self.in_proj(feat.flatten(2).transpose(1, 2)) + pos
        x, y = agent_states[..., 0], agent_states[..., 1]  # normalized [-1, 1], +y = up (row 0)
        row = ((1.0 - y) * 0.5 * g).floor().clamp(0, g - 1).long()
        col = ((x + 1.0) * 0.5 * g).floor().clamp(0, g - 1).long()
        unit_pos = pos[row * g + col]  # [B, U, d]
        pad = n_queries - unit_pos.shape[1]
        q_pos = F.pad(unit_pos, (0, 0, 0, pad)) if pad else unit_pos
        return mem, q_pos

    @staticmethod
    def after_layers(depth: int, n_layers: int) -> tuple[int, ...]:
        """Trunk layer indices a cross block follows: spread evenly, the last one always last."""
        return tuple((i + 1) * depth // n_layers - 1 for i in range(n_layers))
