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


MASK_ALL, MASK_BATTERY = 1, 2  # official_obs modes: hide enemy classes + battery amounts / battery amounts only


def mask_canonical(graphic: torch.Tensor, team_state: torch.Tensor, agent_states: torch.Tensor, mode: int = MASK_ALL):
    """Fork canonical inputs [B,13,H,W], [B,4], [B,10,12] -> the same with only official information.
    Idempotent (the export adapter produces already-masked tensors)."""
    c = torch.as_tensor(BATTERY_CONST, dtype=graphic.dtype, device=graphic.device)
    bat = graphic[:, 8:9]
    graphic = torch.cat([graphic[:, :8], torch.where(bat > 0, c, torch.zeros_like(bat)), graphic[:, 9:]], dim=1)
    held_bat = agent_states[..., 4:5]
    enemy = torch.zeros_like(agent_states[:, :, 9:12])
    own_rows = (torch.arange(N_UNITS, device=agent_states.device) < N_TEAM).view(1, N_UNITS, 1)
    classes = torch.where(own_rows, agent_states[..., 9:12], enemy) if mode == MASK_ALL else agent_states[..., 9:12]
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
        self.register_buffer("regions", region_grid(), persistent=False)
        self.tracked = getattr(net, "official_mode", 1) == MASK_BATTERY

    def forward(self, vector: torch.Tensor, graphic: torch.Tensor, state_in: torch.Tensor | None = None):
        """mode 1: (vector, graphic) -> action; mode 2 (enemy classes tracked): (vector, graphic, state_in [5, S])
        -> (action, state_out [5, S])."""
        state_out = None
        if self.tracked:
            fork_g, t, a, b, state_out = self.canonical_tracked(vector, graphic, state_in)
        else:
            fork_g, t, a, b = self.canonical(vector, graphic)
        q, *_, offsets = self.net(fork_g.unsqueeze(0), t.unsqueeze(0), a.unsqueeze(0), q_units=N_TEAM, angle=True)
        q, offsets = q[0], offsets[0]
        best = torch.argmax(q, dim=-1)
        stand = best >= self.bins
        k = torch.clamp(best, max=self.bins - 1)
        angle = self.centres.index_select(0, k) + torch.gather(offsets, 1, k.unsqueeze(-1)).squeeze(-1)
        move = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        move = torch.where(stand.unsqueeze(-1), torch.zeros_like(move), move)
        move = torch.where(b, move.flip(-1), move)
        return move if state_out is None else (move, state_out)

    def canonical_tracked(self, vector, graphic, state_in):
        """``canonical`` with the enemy classes from the tracker (state: the freshest row, i.e. the largest step)."""
        fork_g, t, a, b = self.canonical(vector, graphic)
        r = torch.argmax(vector[:, 2].abs())
        v0 = vector.index_select(0, r.reshape(1))[0]
        pos = v0[:N_UNITS * UNIT_PRE].reshape(N_UNITS, UNIT_PRE)[:, :2] * 24.0
        fresh = state_in.index_select(0, torch.argmax(state_in[:, _STEP]).reshape(1))[0]
        cls_abs, new_state = track_classes(fresh, pos, vector[:, CLASS_AT:CLASS_AT + 3], b, self.regions)
        cls_can = torch.where(b, cls_abs.index_select(0, self.roll), cls_abs)
        a = torch.cat([a[:, :9], cls_can, a[:, 12:]], dim=-1)
        return fork_g, t, a, b, new_state.unsqueeze(0).expand(vector.shape[0], STATE_SIZE).contiguous()

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


# ------------------------------------------------------------------------------------------- class tracker

STATE_SIZE = 64            # per-row recurrent state (state_in / state_out): prev positions 20, classes 30, step 1
_POS, _CLS, _STEP = slice(0, 20), slice(20, 50), 50
JUMP_TILES = 0.6           # a move longer than this in one decision is a respawn teleport (max ~0.36 with a speed buff)
COLLECTOR, HUNTER, CARRIER = 0, 1, 2
AGENT_ORDER = (3, 6, 2, 7, 5, 8, 4, 9, 1, 0)  # sim/state.py DEFAULT_ORDER


def region_grid() -> torch.Tensor:
    """[24 * 24] region id by flat world cell x * 24 + y: 1 Hunter sanctuary, 2 / 3 team A / B Carrier sanctuary."""
    g = torch.zeros(24, 24, dtype=torch.int64)
    g[11:13, 11:13] = 1
    g[1, 20] = 2
    g[20, 1] = 3
    return g.reshape(-1)


def track_classes(state: torch.Tensor, pos: torch.Tensor, own_cls: torch.Tensor, b: torch.Tensor,
                  regions: torch.Tensor):
    """One decision of the enemy-class tracker (the sim's rules: a Collector turns into a Hunter when its tile
    enters the Hunter sanctuary and its team has < 2 Hunters, into a Carrier on its own locked Carrier sanctuary
    while its team has none; a teleport = death = Collector).

    state [STATE_SIZE] (freshest row), pos [10, 2] world tiles, own_cls [5, 3] (rows = own slots), b: team B.
    Returns (classes one-hot [10, 3] absolute unit order, new state [STATE_SIZE])."""
    first = state[_STEP] < 0.5
    prev = torch.where(first, pos.reshape(-1), state[_POS]).reshape(10, 2)
    cls = state[_CLS].reshape(10, 3)
    collector = torch.tensor([1.0, 0.0, 0.0])
    cls = torch.where(first, collector.expand(10, 3), cls)
    def region_of(p):
        c = torch.clamp(torch.floor(p), 0, 23).to(torch.int64)
        return regions.index_select(0, c[:, 0] * 24 + c[:, 1]), c

    reg, cell = region_of(pos)
    preg, pcell = region_of(prev)
    mreg, mcell = region_of(0.5 * (prev + pos))  # a decision is two physics ticks: the tile in between
    # the sanctuary a unit's tile entered this decision (0 = none), at the middle tick or at the end
    via_mid = (mreg != preg) & (mreg > 0) & (mcell != pcell).any(dim=1)
    via_end = (reg != torch.where(via_mid, mreg, preg)) & (reg > 0) & (cell != torch.where(via_mid.unsqueeze(1), mcell, pcell)).any(dim=1)
    jump = (pos - prev).norm(dim=1) > JUMP_TILES
    # a respawn teleport enters nothing on its way (its midpoint can fall on a sanctuary)
    entered_reg = torch.where(via_mid, mreg, torch.where(via_end, reg, torch.zeros_like(reg)))
    entered_reg = torch.where(jump, torch.zeros_like(entered_reg), entered_reg)
    # own team: known exactly from the rows
    cls = torch.where(b, torch.cat([cls[:5], own_cls]), torch.cat([own_cls, cls[5:]]))
    hunter_oh = torch.tensor([0.0, 1.0, 0.0])
    carrier_oh = torch.tensor([0.0, 0.0, 1.0])
    new = [cls[u] for u in range(10)]
    for u in AGENT_ORDER:  # the order the sim (ML-Agents AgentAct) moves units in
        team = 0 if u < 5 else 1
        c = torch.where(jump[u], collector, new[u])
        is_col = c[COLLECTOR] > 0.5
        cur = torch.stack(new[5 * team:5 * team + 5])
        n_h = cur[:, HUNTER].sum()
        n_c = cur[:, CARRIER].sum()
        to_h = (entered_reg[u] == 1) & is_col & (n_h < 2)
        to_c = (entered_reg[u] == 2 + team) & is_col & (n_c < 0.5)
        new[u] = torch.where(to_h, hunter_oh, torch.where(to_c, carrier_oh, c))
    rows = [n.unsqueeze(0) for n in new]
    tracked = torch.cat(rows)
    # own rows stay as observed (the tracker only fills the enemy's)
    out = torch.where(b, torch.cat([tracked[:5], own_cls]), torch.cat([own_cls, tracked[5:]]))
    new_state = torch.cat([pos.reshape(-1), out.reshape(-1), (state[_STEP:_STEP + 1] + 1.0),
                           torch.zeros(STATE_SIZE - _STEP - 1)])
    return out, new_state
