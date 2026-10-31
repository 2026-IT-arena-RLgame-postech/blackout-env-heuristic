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
    hidden_bat = torch.where(bat > 0, c, torch.zeros_like(bat))
    if mode == 3:  # storage amounts are tracked exactly (track_storage): only the floor's are unknown
        store = (graphic[:, 6:7] > 0) | (graphic[:, 7:8] > 0)
        hidden_bat = torch.where(store, bat, hidden_bat)
    graphic = torch.cat([graphic[:, :8], hidden_bat, graphic[:, 9:]], dim=1)
    held_bat = agent_states[..., 4:5]
    enemy = torch.zeros_like(agent_states[:, :, 9:12])
    own_rows = (torch.arange(N_UNITS, device=agent_states.device) < N_TEAM).view(1, N_UNITS, 1)
    classes = torch.where(own_rows, agent_states[..., 9:12], enemy) if mode == MASK_ALL else agent_states[..., 9:12]  # 2, 3: tracked
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
        mode = getattr(net, "official_mode", 1)
        self.tracked = mode in (MASK_BATTERY, 3)
        self.store_tracked = mode == 3
        self.state_size = STATE_SIZE + (STORE_SIZE if self.store_tracked else 0)
        sreg, sorder = storage_tables()
        self.register_buffer("store_region", sreg, persistent=False)
        self.register_buffer("store_order", sorder, persistent=False)

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
        cls_abs, new_state = track_classes(fresh[:STATE_SIZE], pos, vector[:, CLASS_AT:CLASS_AT + 3], b, self.regions)
        cls_can = torch.where(b, cls_abs.index_select(0, self.roll), cls_abs)
        a = torch.cat([a[:, :9], cls_can, a[:, 12:]], dim=-1)
        if self.store_tracked:
            first = fresh[_STEP] < 0.5
            g_abs = graphic.index_select(0, r.reshape(1))[0].reshape(11, 24, 4, 24, 4).amax(dim=(2, 4))
            amounts, new_store = track_storage(fresh[STATE_SIZE:], v0, g_abs, b, first, self.store_region, self.store_order)
            # storage tiles' amounts into the battery channel (team A rows top-down; canonical flip for team B)
            amap = amounts.reshape(24, 24).transpose(0, 1).flip(0) / 15.0           # [row = 23 - y, col = x]
            amap_b = amap.transpose(0, 1).flip(0).flip(1)
            amap = torch.where(b, amap_b, amap)
            store = (fork_g[6] > 0) | (fork_g[7] > 0)
            fork_g = torch.cat([fork_g[:8], torch.where(store, amap, fork_g[8]).unsqueeze(0), fork_g[9:]], dim=0)
            new_state = torch.cat([new_state, new_store])
        return fork_g, t, a, b, new_state.unsqueeze(0).expand(vector.shape[0], self.state_size).contiguous()

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


# ------------------------------------------------------------------------------------------- storage tracker
# Score rises the moment a battery enters a storage and falls when one is stolen; an absorption (every 20 s)
# empties the storages without touching the score.  Storage.TryDistributeItem fills the region's tiles in a fixed
# order (farthest from the team's spawn first, then x, y; the bases' A1/B1 in rect order), each up to 10, skipping
# tiles holding another item type.  So with the scores, the battery pixels and who just put down a battery, the
# amount on every storage tile is known exactly (official/storage tracker below; validated in sim at 50 Hz).

STORE_SIZE = 576 * 2 + 2 + 10 + 1 + 2   # amounts, last presence, last scores [2], last held [10], last absorb clock, locked scores [2]
MASK_STORE = 3                      # official_obs mode: floor / carried battery amounts hidden, storage amounts tracked


def storage_tables():
    """(region id by flat world cell x * 24 + y [576] (-1 none; team A 0-5, team B 6-11: base storage, then candidates
    0-4), fill order per region [12, 8] (flat cells, -1 padding))."""
    from ..sim.mapdata import FIXED_REGIONS, SPAWN_CELL, STORAGE_CANDIDATES, _mirror
    region = torch.full((576,), -1, dtype=torch.int64)
    order = torch.full((12, 8), -1, dtype=torch.int64)

    def cells_of(rects):
        out = []
        for (rx, ry, rw, rh) in rects:
            for y in range(ry, ry + rh):
                for x in range(rx, rx + rw):
                    if (x, y) not in out:
                        out.append((x, y))
        return out

    base = {0: next(r for n, t, r, *_ in FIXED_REGIONS if n == "Storage A1"),
            1: next(r for n, t, r, *_ in FIXED_REGIONS if n == "Storage B1")}
    for team in (0, 1):
        regs = [cells_of(base[team])]
        ax, ay = SPAWN_CELL[team]
        for _, rects in STORAGE_CANDIDATES:
            cells = cells_of(rects if team == 0 else _mirror(rects))
            cells.sort(key=lambda c: (-(abs(c[0] - ax) + abs(c[1] - ay)), c[0], c[1]))
            regs.append(cells)
        for k, cells in enumerate(regs):
            rid = 6 * team + k
            for j, (x, y) in enumerate(cells):
                region[x * 24 + y] = rid
                order[rid, j] = x * 24 + y
    return region, order


def track_storage(store: torch.Tensor, v0: torch.Tensor, g_abs: torch.Tensor, b: torch.Tensor, first: torch.Tensor,
                  region: torch.Tensor, order: torch.Tensor):
    """One decision of the storage tracker, absolute frame.  store [STORE_SIZE] (previous), v0 [96] a present row,
    g_abs [11, 24, 24] the pooled id map (rows top-down), b team B, first: first decision of the episode.
    Returns (amount per flat cell x * 24 + y [576], new store)."""
    # (presence / held flags as 0/1 floats: onnxruntime has no Where over bool tensors)
    amounts, prev_pres_f = store[:576], store[576:1152]
    prev_scores, prev_held_f, prev_abs = store[1152:1154], store[1154:1164], store[1164]
    locked = store[1165:1167]  # each team's score at the last absorption: storage total = score - locked
    # flat cell index x * 24 + y of the pooled map (row = 23 - y, col = x)
    flat = g_abs.flip(1).transpose(1, 2).reshape(11, 576)
    presence_f = (flat[6] > 0.5).to(store.dtype)
    special = flat[7:11].amax(dim=0) > 0.5
    blocks = v0[:N_UNITS * UNIT_PRE].reshape(N_UNITS, UNIT_PRE)
    pos = blocks[:, :2] * 24.0
    held_f = (blocks[:, 4] > 0.5).to(store.dtype)      # held one-hot index 1 = a battery
    tile = torch.clamp(torch.floor(pos), 0, 23).to(torch.int64)
    tile_flat = tile[:, 0] * 24 + tile[:, 1]
    # a unit pixel on a tile's centre pixel hides its item: presence unknown there, keep the last one
    px = torch.floor(pos * 4.0)
    on_centre = (px == tile.to(px.dtype) * 4.0 + 2.0).all(dim=1)
    hidden = (torch.nn.functional.one_hot(tile_flat, 576) * on_centre.unsqueeze(1)).amax(dim=0) > 0
    presence_f = torch.where(hidden, prev_pres_f, presence_f)
    own, opp = v0[CLASS_AT + 3] * 100.0, v0[CLASS_AT + 4] * 100.0
    scores = torch.round(torch.where(b, torch.stack([opp, own]), torch.stack([own, opp])))
    tl = v0[CLASS_AT + 5]
    absorb = 1.0 - torch.remainder((1.0 - tl) * EPISODE_S, ABSORB_S) / ABSORB_S
    amounts = torch.where(first, torch.zeros_like(amounts), amounts)
    prev_pres_f = torch.where(first, presence_f, prev_pres_f)
    prev_scores = torch.where(first, scores, prev_scores)
    prev_held_f = torch.where(first, held_f, prev_held_f)
    prev_abs = torch.where(first, absorb, prev_abs)
    locked = torch.where(first, torch.zeros_like(locked), locked)
    presence, prev_pres = presence_f > 0.5, prev_pres_f > 0.5
    held, prev_held = held_f > 0.5, prev_held_f > 0.5
    reg_cell = region                                   # [576]
    unit_reg = region.index_select(0, tile_flat)       # [10]
    for team in (0, 1):
        tiles = (reg_cell >= 6 * team) & (reg_cell < 6 * team + 6)
        removed = tiles & prev_pres & ~presence
        stolen = (amounts * removed).sum()
        amounts = torch.where(removed, torch.zeros_like(amounts), amounts)
        dep = scores[team] - prev_scores[team] + stolen
        put_down = prev_held[5 * team:5 * team + 5] & ~held[5 * team:5 * team + 5]
        ureg = unit_reg[5 * team:5 * team + 5]
        mine = put_down & (ureg >= 6 * team) & (ureg < 6 * team + 6)
        new_tiles = tiles & presence & ~prev_pres
        new_reg = torch.where(new_tiles, reg_cell, torch.full_like(reg_cell, -1)).amax()
        r = torch.where(mine.any(), ureg[torch.argmax(mine.to(torch.int64))], new_reg)
        slots = order.index_select(0, torch.clamp(r, 0, 11).reshape(1))[0]       # [8]
        valid = (slots >= 0) & (r >= 0) & (dep > 0)
        s = torch.clamp(slots, 0, 575)
        cap = torch.where(valid & ~special.index_select(0, s), 10.0 - amounts.index_select(0, s), torch.zeros(8))
        cap = torch.clamp(cap, min=0.0)
        before = torch.cumsum(cap, 0) - cap
        fill = torch.minimum(torch.clamp(dep - before, min=0.0), cap)
        amounts = amounts + (torch.nn.functional.one_hot(s, 576).to(fill.dtype) * fill.unsqueeze(1)).sum(dim=0)
    absorbed = absorb > prev_abs + 0.5
    amounts = torch.where(absorbed, torch.zeros_like(amounts), amounts)
    locked = torch.where(absorbed, scores, locked)
    # safety net (never needed in sim tests): each team's tiles must sum to score - locked; put a surplus into
    # its present tiles in order up to 10, take a deficit from its fullest tiles first
    for team in (0, 1):
        tiles = (reg_cell >= 6 * team) & (reg_cell < 6 * team + 6)
        diff = (scores[team] - locked[team]) - (amounts * tiles).sum()
        room = torch.where(tiles & presence & ~special, torch.clamp(10.0 - amounts, min=0.0), torch.zeros_like(amounts))
        before = torch.cumsum(room, 0) - room
        add = torch.minimum(torch.clamp(diff - before, min=0.0), room)
        have = torch.where(tiles, amounts, torch.zeros_like(amounts))
        before_t = torch.cumsum(have, 0) - have
        take = torch.minimum(torch.clamp(-diff - before_t, min=0.0), have)
        amounts = amounts + torch.where(diff > 0.5, add, torch.zeros_like(add)) - torch.where(diff < -0.5, take, torch.zeros_like(take))
    new = torch.cat([amounts, presence_f, scores, held_f, absorb.reshape(1), locked])
    return amounts, new
