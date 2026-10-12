"""Everything stateless between the tracker and the official action, as one exportable graph.

``ForkPolicyGraph`` takes the fork broadcast the tracker renders (``raw_state`` float32[44] +
packed ``raw_map`` [24, 24], the MapObsAgent wire format) and the team it plays, and returns that
team's five moves -- the whole of what ``MyPolicy`` does around a student net:

* ``MyObsPreprocessor``: unpack the map bits into the 13 graphic channels, the 10x12 agent table
  and the 4 team scalars, in the team's own perspective (ally / enemy channels and signs);
* ``team_frame.canonical_obs``: a Team B view mirrored across the map diagonal;
* the student (``MyModel``, logits + angle head, own rows only);
* ``MyPolicy.act`` at ``iqn`` (= the plain arg max for a logits head): the best bin's heading plus
  its offset, STAND -> the zero move, mirrored back to the world frame for Team B.

Exported to ONNX / TorchScript by ``examples/build_official_submission.py``; ``reference_moves``
is the numpy path it must match.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

N_UNITS, N_TEAM = 10, 5
N_BASE, BATTERY_MAX, N_SPECIAL = 8, 15.0, 4
SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY = 4, 5, 6, 7


class ForkPolicyGraph(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net.eval()
        self.bins = int(net.q_head.angle_bins)       # K heading bins; output K is STAND
        k = torch.arange(self.bins, dtype=torch.float32)
        self.register_buffer("centres", 2 * math.pi * k / self.bins, persistent=False)
        swap = list(range(N_BASE + 1 + N_SPECIAL))
        swap[SPAWN_ALLY], swap[SPAWN_ENEMY], swap[STORAGE_ALLY], swap[STORAGE_ENEMY] = (
            SPAWN_ENEMY, SPAWN_ALLY, STORAGE_ENEMY, STORAGE_ALLY)
        self.register_buffer("swap", torch.tensor(swap), persistent=False)
        self.register_buffer("roll", torch.tensor([(u + N_TEAM) % N_UNITS for u in range(N_UNITS)]), persistent=False)

    def forward(self, raw_state: torch.Tensor, raw_map: torch.Tensor, team: torch.Tensor):
        """raw_state float32[44], raw_map int64[24, 24] (packed ushort), team int64[] (0 A / 1 B)
        -> (moves float32[5, 2] for units 5*team .. 5*team+4 in the world frame, logits [5, K+1])."""
        b = team.reshape(()) > 0
        # ---- MyObsPreprocessor.preprocess_graphic (+ flip_team_perspective, canonical mirror)
        packed = raw_map.to(torch.int64)
        base = torch.remainder(packed, 8)
        battery = torch.remainder(torch.div(packed, 8, rounding_mode="floor"), 16)
        item = torch.remainder(torch.div(packed, 128, rounding_mode="floor"), 8)
        g = torch.cat([
            (base.unsqueeze(-1) == torch.arange(N_BASE)).to(torch.float32),
            (battery.to(torch.float32) / BATTERY_MAX).unsqueeze(-1),
            (item.unsqueeze(-1) == torch.arange(1, N_SPECIAL + 1)).to(torch.float32),
        ], dim=-1)                                                  # [H, W, 13] team A
        g_b = g.index_select(-1, self.swap).transpose(0, 1).flip(0).flip(1)
        g = torch.where(b, g_b, g)
        # ---- preprocess_agent_states (+ the Team B sign flip, row roll and x/y swap)
        blocks = raw_state[:4 * N_UNITS].reshape(N_UNITS, 4)
        pos, held, cls = blocks[:, :2], blocks[:, 2], torch.round(blocks[:, 3]).to(torch.int64)
        is_bat = held < 0
        slot = torch.where(is_bat, torch.ones_like(cls), torch.round(held).to(torch.int64))
        items = (slot.unsqueeze(-1) == torch.arange(6)).to(torch.float32)
        items = torch.where(is_bat.unsqueeze(-1) & (torch.arange(6) == 1), (-held / BATTERY_MAX).unsqueeze(-1), items)
        sign = torch.cat([torch.ones(N_TEAM), -torch.ones(N_TEAM)])
        classes = (cls.unsqueeze(-1) == torch.arange(3)).to(torch.float32)
        a = torch.cat([pos, sign.unsqueeze(-1), items, classes], dim=-1)          # [10, 12]
        a_b = torch.cat([pos.flip(-1), -sign.unsqueeze(-1), items, classes], dim=-1).index_select(0, self.roll)
        a = torch.where(b, a_b, a)
        sa, sb, times = raw_state[40:41], raw_state[41:42], raw_state[42:44]
        t = torch.where(b, torch.cat([sb, sa, times]), torch.cat([sa, sb, times]))
        # ---- the student, own rows only (canonical rows 0-4)
        q, *_, offsets = self.net(g.permute(2, 0, 1).contiguous().unsqueeze(0), t.unsqueeze(0), a.unsqueeze(0), q_units=N_TEAM, angle=True)
        q, offsets = q[0], offsets[0]                                              # [5, K+1], [5, K]
        best = torch.argmax(q, dim=-1)
        stand = best >= self.bins
        k = torch.clamp(best, max=self.bins - 1)
        angle = self.centres.index_select(0, k) + torch.gather(offsets, 1, k.unsqueeze(-1)).squeeze(-1)
        move = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        move = torch.where(stand.unsqueeze(-1), torch.zeros_like(move), move)
        move = torch.where(b, move.flip(-1), move)                                  # reflect back: (x, y) -> (y, x)
        return move, q


def reference_moves(policy, raw_state, raw_map, team: int) -> np.ndarray:
    """The numpy path (``obs_from_raw`` + ``MyPolicy.act``) for the five units of ``team``."""
    from ..sim.state import obs_from_raw

    names = [f"unit_{5 * team + k}" for k in range(N_TEAM)]
    moves = policy.act(obs_from_raw(raw_state, raw_map, names))
    return np.stack([moves[n] for n in names])
