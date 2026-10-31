"""The competition's observation (ONNX, stateless, team mode) expressed in the fork's canonical inputs.

docs/competition (official blackout-env, 2026-10-31): a submission is one ONNX model, called once per step
with ``vector (5, 96)`` + ``graphic (5, 11, 96, 96)`` and returning ``(5, 2)``; nothing persists between
calls.  Against the fork's inputs (MapObsAgent: ``graphic [13, 24, 24]``, ``team_state [4]``,
``agent_states [10, 12]``) a single official frame lacks exactly:

* the classes of the enemy units (the map draws every unit as one ally / enemy pixel, the vector carries
  only the observer's own class -- team mode gives all five own rows, so our classes are known);
* battery amounts (on the floor and carried: the map and the held one-hot only say "a battery").

Everything else is recoverable: the 96x96 id map pooled 4x4 is the fork's tile map (walls, storages,
items; sanctuaries / spawns are fixed), positions are the same float32 ``pos / 24``, the absorption clock
follows from the episode clock (every 20 s from the start).

``mask_canonical`` removes those two pieces from fork inputs (enemy class one-hots -> 0, any battery
amount -> ``BATTERY_CONST``), so a student trained on masked inputs (``MyModel(official_obs=True)``) sees
only what the competition gives.  ``OfficialTeamGraph`` is the exported submission: official inputs ->
the same masked canonical tensors -> the student -> the five world-frame moves.
"""

from __future__ import annotations

import math

import torch
from torch import nn

N_UNITS, N_TEAM = 10, 5
BATTERY_CONST = 1.0 / 3.0  # the amount channel's value wherever a battery is (amount unknown)
EPISODE_S, ABSORB_S = 420.0, 20.0
UNIT_PRE = 9          # official vector block per unit: x, y, team_sign, held one-hot (6)
CLASS_AT = 90         # own class one-hot (3), then own score, opponent score, time left


def mask_canonical(graphic: torch.Tensor, team_state: torch.Tensor, agent_states: torch.Tensor):
    """Fork canonical inputs [B,13,H,W], [B,4], [B,10,12] -> the same with only official information.
    Idempotent (the export adapter produces already-masked tensors)."""
    c = torch.as_tensor(BATTERY_CONST, dtype=graphic.dtype, device=graphic.device)
    bat = graphic[:, 8:9]
    graphic = torch.cat([graphic[:, :8], torch.where(bat > 0, c, torch.zeros_like(bat)), graphic[:, 9:]], dim=1)
    held_bat = agent_states[..., 4:5]
    enemy = torch.zeros_like(agent_states[:, :, 9:12])
    own_rows = (torch.arange(N_UNITS, device=agent_states.device) < N_TEAM).view(1, N_UNITS, 1)
    classes = torch.where(own_rows, agent_states[..., 9:12], enemy)
    tl = team_state[:, 2:3]
    absorb = 1.0 - torch.remainder((1.0 - tl) * EPISODE_S, ABSORB_S) / ABSORB_S
    team_state = torch.cat([team_state[:, :3], absorb, team_state[:, 4:]], dim=1)
    agent_states = torch.cat([agent_states[..., :4], torch.where(held_bat > 0, c.to(agent_states.dtype), torch.zeros_like(held_bat)),
                              agent_states[..., 5:9], classes, agent_states[..., 12:]], dim=-1)
    return graphic, team_state, agent_states


class OfficialTeamGraph(nn.Module):
    """Competition team-mode model: ``(vector [5, 96], graphic [5, 11, 96, 96]) -> action [5, 2]``.

    ``static``: the fork's team-A tile channels that never change (hunter sanctuary, carrier sanctuaries,
    spawns: base channels 2-5), [4, 24, 24]; walls and storages come from the official map."""

    def __init__(self, net: nn.Module, static: torch.Tensor):
        super().__init__()
        self.net = net.eval()
        self.bins = int(net.q_head.angle_bins)
        k = torch.arange(self.bins, dtype=torch.float32)
        self.register_buffer("centres", 2 * math.pi * k / self.bins, persistent=False)
        self.register_buffer("static", static.to(torch.float32), persistent=False)
        self.register_buffer("roll", torch.tensor([(u + N_TEAM) % N_UNITS for u in range(N_UNITS)]), persistent=False)

    def forward(self, vector: torch.Tensor, graphic: torch.Tensor) -> torch.Tensor:
        fork_g, t, a, b = self.canonical(vector, graphic)
        q, *_, offsets = self.net(fork_g.unsqueeze(0), t.unsqueeze(0), a.unsqueeze(0), q_units=N_TEAM, angle=True)
        q, offsets = q[0], offsets[0]
        best = torch.argmax(q, dim=-1)
        stand = best >= self.bins
        k = torch.clamp(best, max=self.bins - 1)
        angle = self.centres.index_select(0, k) + torch.gather(offsets, 1, k.unsqueeze(-1)).squeeze(-1)
        move = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        move = torch.where(stand.unsqueeze(-1), torch.zeros_like(move), move)
        return torch.where(b, move.flip(-1), move)

    def canonical(self, vector: torch.Tensor, graphic: torch.Tensor):
        """Official team inputs -> masked fork canonical (graphic [13,24,24], team_state [4], agents [10,12], is_b)."""
        # the runner zero-pads the rows of absent units: read the shared table / map from a present row
        # (unit_0's team sign is +-1 in every real row)
        r = torch.argmax(vector[:, 2].abs())
        v0 = vector.index_select(0, r.reshape(1))[0]
        b = v0[2] < 0                                                  # unit_0 is an enemy: we are team B
        # ---- map: 4x4 pooling to tiles (rows top-down, like the fork), team B anti-transposed
        g = graphic.index_select(0, r.reshape(1))[0].reshape(11, 24, 4, 24, 4).amax(dim=(2, 4))  # [11, 24, 24]
        g_b = g.transpose(1, 2).flip(1).flip(2)
        g = torch.where(b, g_b, g)
        wall, ally_st, enemy_st = g[1:2], g[2:3], g[3:4]
        static = self.static                                           # [4, 24, 24] canonical (team A) = canonical B
        base_rest = torch.cat([wall, static, ally_st, enemy_st], dim=0)  # channels 1..7
        empty = 1.0 - base_rest.amax(dim=0, keepdim=True)
        bat = torch.where(g[6:7] > 0, torch.full_like(g[6:7], BATTERY_CONST), torch.zeros_like(g[6:7]))
        fork_g = torch.cat([empty, base_rest, bat, g[7:11]], dim=0)    # [13, 24, 24]
        # ---- unit table
        blocks = v0[:N_UNITS * UNIT_PRE].reshape(N_UNITS, UNIT_PRE)
        pos = blocks[:, :2] * 2.0 - 1.0
        sign = blocks[:, 2:3]
        held = blocks[:, 3:9]
        held = torch.cat([held[:, :1], held[:, 1:2] * BATTERY_CONST, held[:, 2:]], dim=1)
        own_cls = vector[:, CLASS_AT:CLASS_AT + 3]                     # [5, 3] slots 0-4
        zero_cls = torch.zeros_like(own_cls)
        cls_a = torch.cat([own_cls, zero_cls], dim=0)                  # absolute rows: A's units first
        cls_b = torch.cat([zero_cls, own_cls], dim=0)
        cls_abs = torch.where(b, cls_b, cls_a)
        a = torch.cat([pos, sign, held, cls_abs], dim=-1)              # [10, 12] absolute rows
        a_b = torch.cat([pos.flip(-1), sign, held, cls_abs], dim=-1).index_select(0, self.roll)
        a = torch.where(b, a_b, a)
        # ---- team scalars: own, opponent, time left, absorption clock (every 20 s from the start)
        tl = v0[CLASS_AT + 5:CLASS_AT + 6]
        elapsed = (1.0 - tl) * EPISODE_S
        absorb = 1.0 - torch.remainder(elapsed, ABSORB_S) / ABSORB_S
        t = torch.cat([v0[CLASS_AT + 3:CLASS_AT + 5], tl, absorb])
        return fork_g, t, a, b


def static_channels() -> torch.Tensor:
    """Fork base channels 2-5 (sanctuaries, spawns) of the team-A view, from a rendered sim state."""
    import numpy as np

    from ..sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from ..sim.mapdata import H, W
    from ..sim.state import SimState

    bs = BatchSim([SimState.from_seed(1)])
    g = np.zeros((1, 2, N_CH, H, W), np.float32)
    bs.render(np.array([0]), g, np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 10, N_AGENT_COLS), np.float32))
    return torch.from_numpy(g[0, 0, 2:6].copy())
