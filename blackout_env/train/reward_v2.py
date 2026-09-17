"""
Reward v2: per-unit potentials computed from observations (docs/reward_v2_design.md).

Every quantity is in battery points and is a pure function of one observation (graphic,
agent_states, team_state) from one team's perspective, batched, so it can be evaluated on stored
data, on hypothetical one-step moves, and inside n-step returns with the learner's own gamma.

Team value      V_T(s) = C_T(s) + sum_{u in T} U_u(s)
Zero-sum reward r_own  = d(score_own - score_enemy) + (g*V_own' - V_own) - (g*V_enemy' - V_enemy)
Per-unit credit        own unit i gets its own dU_i, a fifth of each team's dC, and minus each
                       enemy unit's dU_j when i is the Hunter assigned to that enemy (a fifth of it
                       when no Hunter is) -- see shaped_rewards()

The displayed score counts batteries sitting in a team's storages too (ScoreItemEffect adds on
entering storage, subtracts when one leaves, and only an absorption locks it in). Confirmed score =
displayed score - batteries in own storages, and that is NOT part of V: it is the real reward, paid
when an absorption locks stored batteries in (at which point their stored value, undiscounted as
tau -> 0, leaves V). Keeping
it inside V would add -(1-g)*score to every tick -- a constant pull proportional to the lead that
measured 0.07 points/tick against near-zero informative shaping at g = 0.997.

C_T  batteries in T's storages, each discounted by the chance an enemy
     Collector/Carrier (the classes that can pick it up) steals it before the next absorption;
     a storage inside T's own base cannot be reached by the enemy
U1   carrying: amount * survival * lambda(distance to a storage that can take the whole amount)
U2   empty Collector/Carrier with a team-assigned battery: the value it will have once picked up,
     so the pickup itself is worth nothing and only the approach is rewarded. A battery in the
     enemy's storage counts only its (1 - keep) share -- the part the enemy's C already writes off
     to thieves -- so a steal that can no longer happen before the absorption is worth nothing and
     the two teams never claim the same battery twice
U3   Hunter with a team-assigned prey: beta * prey value * closeness; beta < 1 so a kill nets
     positive against the prey's value leaving the enemy's side
U4   the one Hunter per team closest to the enemy base exit: e(t) * exit_value * closeness
     (a Hunter's U3/U4 slot takes the larger of the two)
U5   every unit: e(t) * (class value + seconds already travelled from spawn) -- what a death throws
     away (death respawns a Collector at spawn), worth nothing once the field is empty
e(t) batteries still to be won (on the floor or carried, not already stored) / initial total. Stored
     batteries are left out so an absorption, which empties every storage at once, does not make
     every unit's readiness drop in the same tick.

Survival = exp(-hazard * seconds_exposed / distance_to_nearest_threat). Threats are the classes
that can actually destroy that value: enemy Hunters for a carrying Collector, any enemy
Hunter/Collector for a carrying Carrier (Collector beats Carrier), enemy Collectors/Carriers for
a stored battery (heuristic_findings_for_reward_20260916.md §4.1).

Geometry: the wall layout is identical in every match, each base is the 4x4 corner square around
its spawn, and a base floor blocks the enemy team (MapManager.IsWalkable, BlockEnemy). Path
distances are all-pairs over 8-connected moves with no corner cutting, per team, read at a unit's
continuous position by bilinear interpolation over the four surrounding tiles, so a sub-tile step
changes the potential.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

H = W = 24
N_CELLS = H * W
N_UNITS = 10
N_TEAM = 5

WALL, SPAWN_OWN, SPAWN_ENEMY, STORAGE_OWN, STORAGE_ENEMY, BATTERY = 1, 4, 5, 6, 7, 8
FIRST_SPECIAL, N_SPECIALS = 9, 4
TEAM_COL, CARGO_COL = 2, 4
SPECIAL_COLS, CLASS_COLS = slice(5, 9), slice(9, 12)
COLLECTOR, HUNTER, CARRIER = 0, 1, 2

BATTERY_SCALE = 15.0  # channel 8 and agent_states column 4 store count / 15
MAX_ITEM_AMOUNT = 10
TARGET_SCORE = 100.0  # team_state scores are score / TargetScore
ABSORPTION_SECONDS = 20.0
SPEED = np.array([4.0, 6.0, 6.0], dtype=np.float32)  # tiles per second by class
INF = np.float32(1e6)


@dataclass(frozen=True)
class RewardV2Config:
    initial_battery_total: float = 200.0  # e(t) denominator (heuristic findings: ~200 per match)
    steal_hazard: float = 0.14            # 1/s times tiles: a thief 3 tiles away over 20 s keeps 40%
    carry_hazard: float = 0.14
    lambda_rho: float = 0.5               # a battery infinitely far from delivery is worth half
    lambda_length: float = 12.0           # tiles
    hunt_beta: float = 0.5
    hunt_length: float = 4.0              # tiles
    exit_value: float = 5.0               # points at e(t) = 1
    exit_length: float = 3.0              # tiles
    class_value: tuple[float, float, float] = (0.0, 6.0, 3.0)  # Collector, Hunter, Carrier
    travel_value: float = 0.5             # points per second travelled from spawn
    travel_cap_seconds: float = 10.0


# Fitted 2026-09-17 by examples/fit_reward_v2.py (coordinate search, 2 sweeps) on 120 recorded
# heuristic matches: maximises how well confirmed diff + V_A - V_B at 5-30 s correlates with the
# 64 s score difference (mean r 0.49 -> 0.76; at 10 s r 0.71 / AUC 0.88, where the displayed score
# has r 0.05 / AUC 0.57 and the old Unity Psi r 0.08). Held-out V17 variants: rank correlation of
# win rate with value at 10 s +0.71 (displayed score -0.31). exit_value sits at its grid edge and
# Hunter vs exit value trade off along a flat ridge -- small sample, refit on more matches.
FITTED_20260917 = RewardV2Config(
    steal_hazard=0.3, carry_hazard=0.3, lambda_rho=0.8, hunt_beta=0.8,
    exit_value=30.0, class_value=(0.0, 15.0, 3.0), travel_value=1.5,
)


# ---------------------------------------------------------------------------------------- geometry


def base_mask(spawn: tuple[int, int]) -> np.ndarray:
    """The 4x4 corner square holding a spawn (same rule as heuristics/v17_planner._base_mask)."""
    mask = np.zeros((H, W), dtype=bool)
    sy, sx = spawn
    rows = range(sy - 3, sy + 1) if sy > H // 2 else range(sy, sy + 4)
    cols = range(sx - 3, sx + 1) if sx > W // 2 else range(sx, sx + 4)
    for y in rows:
        for x in cols:
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = True
    return mask


def all_pairs_distance(walkable: np.ndarray) -> np.ndarray:
    """[N_CELLS, N_CELLS] shortest 8-connected path length (diagonal sqrt 2, no corner cutting)
    between walkable tiles, INF otherwise (0 on the diagonal regardless)."""
    dist = np.full((N_CELLS, N_CELLS), INF, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for r in range(H):
        for c in range(W):
            if not walkable[r, c]:
                continue
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or not walkable[nr, nc]:
                    continue
                if dr and dc and not (walkable[r, nc] and walkable[nr, c]):
                    continue
                dist[r * W + c, nr * W + nc] = math.sqrt(2.0) if dr and dc else 1.0
    for k in np.flatnonzero(walkable.reshape(-1)):
        np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :], out=dist)
    return dist


def _touching(mask: np.ndarray) -> np.ndarray:
    grown = np.zeros_like(mask)
    padded = np.pad(mask, 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            grown |= padded[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
    return grown & ~mask


@dataclass
class Geometry:
    """Static map facts for one perspective. Index 0 = the observing team, 1 = the enemy."""

    walkable: tuple[np.ndarray, np.ndarray]
    base: tuple[np.ndarray, np.ndarray]
    dist: tuple[np.ndarray, np.ndarray]   # paths walked by that team's units
    spawn: tuple[int, int]
    exits: tuple[np.ndarray, np.ndarray]  # cells team t can stand on that touch the OTHER base


_GEOMETRY_CACHE: dict[bytes, Geometry] = {}


def geometry_for(graphic: np.ndarray) -> Geometry:
    """Geometry for one [H, W, C] observation, cached by walls + spawn channels."""
    wall = graphic[..., WALL] > 0.5
    spawn_own = np.argwhere(graphic[..., SPAWN_OWN] > 0.5)
    spawn_enemy = np.argwhere(graphic[..., SPAWN_ENEMY] > 0.5)
    key = wall.tobytes() + spawn_own.tobytes() + spawn_enemy.tobytes()
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    spawns = (tuple(int(v) for v in spawn_own[0]), tuple(int(v) for v in spawn_enemy[0]))
    bases = (base_mask(spawns[0]), base_mask(spawns[1]))
    walkable = (~wall & ~bases[1], ~wall & ~bases[0])
    geometry = Geometry(
        walkable=walkable,
        base=bases,
        dist=(all_pairs_distance(walkable[0]), all_pairs_distance(walkable[1])),
        spawn=(spawns[0][0] * W + spawns[0][1], spawns[1][0] * W + spawns[1][1]),
        exits=(
            np.flatnonzero((_touching(bases[1]) & walkable[0]).reshape(-1)),
            np.flatnonzero((_touching(bases[0]) & walkable[1]).reshape(-1)),
        ),
    )
    _GEOMETRY_CACHE[key] = geometry
    return geometry


# ---------------------------------------------------------------------------------------- helpers


def unit_positions(agent_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (row, col) in tile units, tile centres at integers. [B, N] each."""
    row = (1.0 - agent_states[..., 1]) * 0.5 * H - 0.5
    col = (agent_states[..., 0] + 1.0) * 0.5 * W - 0.5
    return np.clip(row, 0, H - 1).astype(np.float32), np.clip(col, 0, W - 1).astype(np.float32)


def interpolated_rows(dist: np.ndarray, walkable: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """[B, N, N_CELLS] path distance from each continuous position to every tile.

    Bilinear over the four tiles around the position. A corner tile the team cannot stand on
    borrows the unit's own tile distance plus the offset to it, so the result stays finite and
    continuous next to walls."""
    near_r = np.rint(row).astype(np.int64)
    near_c = np.rint(col).astype(np.int64)
    near_rows = dist[near_r * W + near_c]
    r0 = np.floor(row).astype(np.int64).clip(0, H - 2)
    c0 = np.floor(col).astype(np.int64).clip(0, W - 2)
    fr, fc = row - r0, col - c0
    flat_walk = walkable.reshape(-1)
    out = np.zeros(near_rows.shape, dtype=np.float32)
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        weight = ((fr if dr else 1 - fr) * (fc if dc else 1 - fc))[..., None]
        cell = (r0 + dr) * W + (c0 + dc)
        rows = dist[cell]
        offset = np.hypot(r0 + dr - near_r, c0 + dc - near_c).astype(np.float32)[..., None]
        usable = flat_walk[cell][..., None] & (rows < INF)
        out += weight * np.where(usable, rows, near_rows + offset)
    return np.minimum(out, INF)


def component_capacity(storage: np.ndarray, battery: np.ndarray, special: np.ndarray) -> np.ndarray:
    """[B, H, W] free capacity of the 4-connected storage component each tile belongs to (0 off
    storage). Unity spreads a deposit over the whole region and refuses it whole if it does not
    fit; a tile holding a special item takes no battery."""
    batch = storage.shape[0]
    labels = np.where(storage, np.arange(1, N_CELLS + 1, dtype=np.int64).reshape(1, H, W), 0)
    for _ in range(12):  # > largest storage diameter
        p = np.pad(labels, ((0, 0), (1, 1), (1, 1)))
        spread = np.maximum.reduce([p[:, 1:-1, 1:-1], p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]])
        labels = np.where(storage, spread, 0)
    free = np.where(storage & ~special, MAX_ITEM_AMOUNT - battery, 0).clip(0).astype(np.float64)
    offsets = (np.arange(batch) * (N_CELLS + 1))[:, None]
    flat = (labels.reshape(batch, -1) + offsets).reshape(-1)
    totals = np.bincount(flat, weights=free.reshape(-1), minlength=batch * (N_CELLS + 1)).reshape(batch, -1)
    totals[:, 0] = 0
    return np.take_along_axis(totals, labels.reshape(batch, -1), 1).reshape(batch, H, W).astype(np.float32)


def greedy_assign(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[B, R, K] -> [B, R] chosen column per row (-1 none): highest value first, each row and each
    column used at most once, ties to the lower row then the lower column."""
    batch, rows, cols = values.shape
    work = np.where(valid, values, -np.inf).astype(np.float64)
    chosen = np.full((batch, rows), -1, dtype=np.int64)
    arange = np.arange(batch)
    for _ in range(min(rows, cols)):
        flat = work.reshape(batch, -1).argmax(1)
        ok = np.isfinite(work.reshape(batch, -1)[arange, flat])
        if not ok.any():
            break
        b, r, c = arange[ok], flat[ok] // cols, flat[ok] % cols
        chosen[b, r] = c
        work[b, r, :] = -np.inf
        work[b, :, c] = -np.inf
    return chosen


# ---------------------------------------------------------------------------------------- potentials


@dataclass
class Potentials:
    """Per-unit potentials for one batch of observations (own perspective)."""

    unit: np.ndarray       # [B, 10] U_u in points
    common: np.ndarray     # [B, 2]  C_own, C_enemy
    confirmed: np.ndarray  # [B, 2]  locked-in score own, enemy (points)
    hunter_of: np.ndarray  # [B, 10] index of the Hunter assigned to hunt this unit, -1 none
    own: np.ndarray        # [B, 10] unit is on the observing team
    terms: dict[str, np.ndarray] = field(default_factory=dict)  # [B, 10] breakdown, for evaluation

    def team_value(self) -> np.ndarray:
        """[B, 2] V_own, V_enemy."""
        own = np.where(self.own, self.unit, 0).sum(1) + self.common[:, 0]
        enemy = np.where(~self.own, self.unit, 0).sum(1) + self.common[:, 1]
        return np.stack([own, enemy], 1)


def compute_potentials(
    graphic: np.ndarray, agent_states: np.ndarray, team_state: np.ndarray, cfg: RewardV2Config = RewardV2Config()
) -> Potentials:
    """graphic [B, H, W, C], agent_states [B, 10, 12], team_state [B, 4], all one perspective and
    one map (geometry is read from graphic[0])."""
    geo = geometry_for(graphic[0])
    batch = graphic.shape[0]
    arange = np.arange(batch)

    own = agent_states[..., TEAM_COL] > 0
    side = np.where(own, 0, 1)  # 0 = observing team
    cls = agent_states[..., CLASS_COLS].argmax(-1)
    cargo = np.rint(np.maximum(agent_states[..., CARGO_COL], 0) * BATTERY_SCALE).astype(np.float32)
    holds_special = (agent_states[..., SPECIAL_COLS] > 0.5).any(-1)
    row, col = unit_positions(agent_states)
    cell = np.rint(row).astype(np.int64) * W + np.rint(col).astype(np.int64)
    speed = SPEED[cls]

    battery = np.rint(graphic[..., BATTERY] * BATTERY_SCALE).astype(np.float32)
    special = (graphic[..., FIRST_SPECIAL : FIRST_SPECIAL + N_SPECIALS] > 0.5).any(-1)
    storage = (graphic[..., STORAGE_OWN] > 0.5, graphic[..., STORAGE_ENEMY] > 0.5)
    capacity = (component_capacity(storage[0], battery, special), component_capacity(storage[1], battery, special))
    tau = ABSORPTION_SECONDS * team_state[:, 3].astype(np.float32)

    # distance rows from each unit's continuous position, walked by that unit's own team
    rows = np.where(
        own[..., None],
        interpolated_rows(geo.dist[0], geo.walkable[0], row, col),
        interpolated_rows(geo.dist[1], geo.walkable[1], row, col),
    )
    unit_to_cell = np.take_along_axis(rows, np.broadcast_to(cell[:, None, :], (batch, N_UNITS, N_UNITS)), 2)  # [B, from, to]

    lam = lambda d: 1.0 - cfg.lambda_rho * np.tanh(np.minimum(d, 1e3) / cfg.lambda_length)
    storage_flat = [s.reshape(batch, -1) for s in storage]
    capacity_flat = [c.reshape(batch, -1) for c in capacity]
    battery_flat = battery.reshape(batch, -1)
    loose = np.where(storage_flat[0] | storage_flat[1], 0, battery_flat).sum(1)
    econ = np.clip((loose + cargo.sum(1)) / cfg.initial_battery_total, 0.0, 1.0)

    # --- threat distance to each unit, from the enemy classes that can kill it while carrying
    enemy_of = side[:, :, None] != side[:, None, :]  # [B, from, to]
    hunter_from = (cls == HUNTER)[:, :, None] & enemy_of
    killer_from = hunter_from | ((cls == COLLECTOR)[:, :, None] & enemy_of)
    d_hunter = np.where(hunter_from, unit_to_cell, INF).min(1)
    d_killer = np.where(killer_from, unit_to_cell, INF).min(1)
    d_threat = np.where(cls == CARRIER, d_killer, d_hunter)  # [B, 10]

    def survival(d_deliver: np.ndarray, threat: np.ndarray, spd: np.ndarray) -> np.ndarray:
        return np.exp(-cfg.carry_hazard * (np.minimum(d_deliver, 60.0) / spd) / np.maximum(threat, 0.5))

    # --- U1 carry
    u1 = np.zeros((batch, N_UNITS), dtype=np.float32)
    for t in (0, 1):
        accepts = storage_flat[t][:, None, :] & (capacity_flat[t][:, None, :] >= cargo[..., None])
        d_deliver = np.where(accepts, rows, INF).min(-1)
        carrying = (side == t) & (cargo > 0) & (cls != HUNTER)
        u1 = np.where(carrying, cargo * survival(d_deliver, d_threat, speed) * lam(d_deliver), u1)

    # --- C: score + stored batteries discounted by thieves of the other team
    thief_from = ((cls == COLLECTOR) | (cls == CARRIER))
    common = np.zeros((batch, 2), dtype=np.float32)
    confirmed = np.zeros((batch, 2), dtype=np.float32)
    keep_by_team = []
    for t in (0, 1):
        thieves = thief_from & (side != t)                                   # [B, 10]
        # a thief walks its own team's distance table from its tile to the storage tile
        d_thief = np.where(thieves[..., None], rows, INF).min(1)             # [B, N_CELLS]
        protected = geo.base[t].reshape(1, -1)
        keep = np.where(protected, 1.0, np.exp(-cfg.steal_hazard * tau[:, None] / np.maximum(d_thief, 0.5)))
        keep_by_team.append(keep)
        stored = np.where(storage_flat[t], battery_flat * keep, 0).sum(1)
        common[:, t] = stored
        confirmed[:, t] = team_state[:, t] * TARGET_SCORE - np.where(storage_flat[t], battery_flat, 0).sum(1)

    # --- U2 fetch: empty Collector/Carrier x batteries off that team's own storages
    u2 = np.zeros((batch, N_UNITS), dtype=np.float32)
    for t in (0, 1):
        fetchers = (side == t) & (cargo <= 0) & ~holds_special & (cls != HUNTER)
        candidate = (battery_flat > 0) & ~storage_flat[t]
        k = int(candidate.sum(1).max()) if candidate.any() else 0
        if not k or not fetchers.any():
            continue
        order = np.argsort(~candidate, axis=1, kind="stable")[:, :k]
        valid = np.take_along_axis(candidate, order, 1)
        amount = np.take_along_axis(battery_flat, order, 1)
        on_enemy_storage = np.take_along_axis(storage_flat[1 - t], order, 1)
        stealable = np.where(on_enemy_storage, 1.0 - np.take_along_axis(keep_by_team[1 - t], order, 1), 1.0)
        # the battery's own delivery distance from its tile to an accepting storage of team t,
        # over only the tiles that are storage anywhere in this batch (a few dozen, not 576)
        tiles = np.flatnonzero(storage_flat[t].any(0))
        if len(tiles) == 0:
            continue
        accepts = storage_flat[t][:, None, tiles] & (capacity_flat[t][:, None, tiles] >= amount[..., None])  # [B, K, T]
        d_battery = np.where(accepts, geo.dist[t][order[..., None], tiles], INF).min(-1)                     # [B, K]
        d_fetch = np.take_along_axis(rows, np.broadcast_to(order[:, None, :], (batch, N_UNITS, k)), 2)  # [B, 10, K]
        total = d_fetch + d_battery[:, None, :]
        value = (amount * stealable)[:, None, :] * survival(total, d_threat[..., None], speed[..., None]) * lam(total)
        ok = fetchers[..., None] & valid[:, None, :] & (d_fetch < INF) & (d_battery < INF)[:, None, :]
        pick = greedy_assign(value, ok)
        got = pick >= 0
        chosen = np.take_along_axis(value, np.maximum(pick, 0)[..., None], 2)[..., 0]
        u2 = np.where(got, chosen, u2)

    # --- U5 readiness
    travelled = np.take_along_axis(rows, np.array(geo.spawn)[side][..., None], 2)[..., 0] / speed
    u5 = econ[:, None] * (np.asarray(cfg.class_value, dtype=np.float32)[cls] + cfg.travel_value * np.minimum(travelled, cfg.travel_cap_seconds))

    # --- U3 hunt and U4 exit
    u34 = np.zeros((batch, N_UNITS), dtype=np.float32)
    hunter_of = np.full((batch, N_UNITS), -1, dtype=np.int64)
    prey_value = u1 + u5
    for t in (0, 1):
        hunters = (side == t) & (cls == HUNTER)
        if not hunters.any():
            continue
        prey = (side != t) & (cls != HUNTER)
        closeness = 1.0 - np.tanh(np.minimum(unit_to_cell, 1e3) / cfg.hunt_length)
        value = cfg.hunt_beta * prey_value[:, None, :] * closeness
        ok = hunters[:, :, None] & prey[:, None, :] & (unit_to_cell < INF)
        pick = greedy_assign(value, ok)
        got = pick >= 0
        hunt = np.where(got, np.take_along_axis(value, np.maximum(pick, 0)[..., None], 2)[..., 0], 0.0)
        b, h = np.nonzero(got)
        hunter_of[b, pick[b, h]] = h

        exits = geo.exits[t]
        d_exit = rows[..., exits].min(-1) if len(exits) else np.full((batch, N_UNITS), INF, dtype=np.float32)
        exit_close = np.where(hunters, 1.0 - np.tanh(np.minimum(d_exit, 1e3) / cfg.exit_length), -1.0)
        camper = exit_close.argmax(1)
        has = exit_close[arange, camper] >= 0
        camp = np.zeros((batch, N_UNITS), dtype=np.float32)
        camp[arange[has], camper[has]] = econ[has] * cfg.exit_value * exit_close[arange[has], camper[has]]
        u34 = np.where(hunters, np.maximum(hunt, camp), u34)

    unit = (u1 + u2 + u34 + u5).astype(np.float32)
    return Potentials(
        unit=unit, common=common, confirmed=confirmed, hunter_of=hunter_of, own=own,
        terms={"carry": u1, "fetch": u2, "hunt_exit": u34, "ready": u5, "econ": np.broadcast_to(econ[:, None], (batch, N_UNITS))},
    )


# ---------------------------------------------------------------------------------------- rewards


def score_reward(before: Potentials, after: Potentials) -> np.ndarray:
    """[B] change in (own - enemy) confirmed score, in points. Zeroing it across an episode reset
    (scores drop back to 0 there) is the caller's job."""
    diff = lambda p: p.confirmed[:, 0] - p.confirmed[:, 1]
    return (diff(after) - diff(before)).astype(np.float32)


def shaped_rewards(before: Potentials, after: Potentials, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-sum potential shaping between consecutive observations of the same perspective.

    Returns (per_unit [B, 10], team [B]) in points for the OBSERVING team: per_unit holds the credit
    of own units (enemy rows are 0), and team == per_unit.sum(1) == dV_own - dV_enemy. The score
    reward (score_reward) is team-level and not included."""
    d_unit = gamma * after.unit - before.unit
    d_common = gamma * after.common - before.common
    own = before.own
    per_unit = np.where(own, d_unit + (d_common[:, 0:1] - d_common[:, 1:2]) / N_TEAM, 0.0)
    enemy_delta = np.where(~own, d_unit, 0.0)
    assigned = before.hunter_of >= 0
    # an enemy unit's change goes to the Hunter assigned to it, otherwise a fifth to every own unit
    batch = own.shape[0]
    b, j = np.nonzero(assigned & ~own)
    np.subtract.at(per_unit, (b, before.hunter_of[b, j]), enemy_delta[b, j])
    unassigned = np.where(~assigned & ~own, enemy_delta, 0.0).sum(1, keepdims=True)
    per_unit = np.where(own, per_unit - unassigned / N_TEAM, 0.0)
    return per_unit.astype(np.float32), per_unit.sum(1).astype(np.float32)


# ---------------------------------------------------------------------------------------- training data

POINTS_PER_REWARD = 20.0  # 20 battery points = reward 1.0; a win/loss stays +-5 (five agents x +-1)
TERMINAL_REWARD = 5.0


def match_ends(team_state: np.ndarray) -> np.ndarray:
    """[T] True on the last transition of each match in a sequential block: the next row's episode
    time goes back up (it only ever falls within a match; a match that ends early at the target
    score resets from wherever it was, so no large-jump threshold), and the block's final row."""
    return np.r_[team_state[1:, 2] > team_state[:-1, 2] + 1e-4, True]


def annotate_sequence(
    graphic: np.ndarray,
    agent_states: np.ndarray,
    team_state: np.ndarray,
    stored_reward: np.ndarray,
    done: np.ndarray,
    cfg: RewardV2Config,
    points_per_reward: float = POINTS_PER_REWARD,
    chunk: int = 2048,
) -> dict[str, np.ndarray]:
    """
    Rewrites one team stream's sequential transitions for reward v2. Returns, per row:

      reward     terminal outcome (+-5, recovered from the sign of the stored reward on the last
                 row of a finished match, where Unity's +-1 per agent dominates its shaping) plus
                 the change in confirmed score difference to the next row, in reward units
      potential  V_own - V_enemy of the row's state, in reward units -- the shaping itself is
                 gamma * potential[next] - potential[this], applied inside the n-step return
                 (compute_n_step_return) with the learner's own gamma
      terminal   the row ends a match: the next row belongs to another match, so its potential
                 must not be bootstrapped (an absorption `done` still bootstraps it)
      done       the stored `done`, also forced on every match end -- shards of a parallel
                 collection are concatenated, and a shard can end mid-match
    """
    parts = [compute_potentials(graphic[i : i + chunk], agent_states[i : i + chunk], team_state[i : i + chunk], cfg)
             for i in range(0, len(graphic), chunk)]
    value = np.concatenate([p.team_value() for p in parts])
    confirmed = np.concatenate([p.confirmed for p in parts])
    terminal = match_ends(team_state)
    diff = confirmed[:, 0] - confirmed[:, 1]
    score = np.r_[diff[1:] - diff[:-1], 0.0]
    score[terminal] = 0.0
    finished = terminal & done
    outcome = np.where(finished, np.clip(np.rint(stored_reward / TERMINAL_REWARD), -1, 1), 0.0)
    return {
        "outcome": (outcome * TERMINAL_REWARD).astype(np.float32),
        "reward": (outcome * TERMINAL_REWARD + score / points_per_reward).astype(np.float32),
        "potential": ((value[:, 0] - value[:, 1]) / points_per_reward).astype(np.float32),
        "terminal": terminal,
        "done": done | terminal,
    }


def _annotate_npz_range(args) -> tuple[int, dict[str, np.ndarray]]:
    from blackout_env.train.offline_dataset import npz_member_memmap

    path, start, stop, cfg, points_per_reward, overlap = args
    lo, hi = start, min(stop + overlap, len(npz_member_memmap(path, "done")))
    arrays = {f: np.asarray(npz_member_memmap(path, f)[lo:hi]) for f in ("graphic", "agent_states", "team_state", "reward", "done")}
    arrays["graphic"] = arrays["graphic"].astype(np.float32)
    out = annotate_sequence(arrays["graphic"], arrays["agent_states"], arrays["team_state"], arrays["reward"], arrays["done"], cfg, points_per_reward)
    keep = stop - start
    return start, {k: v[:keep] for k, v in out.items()}


def annotate_dataset(path, cfg: RewardV2Config, points_per_reward: float = POINTS_PER_REWARD, workers: int = 8, block: int = 50_000) -> dict[str, np.ndarray]:
    """annotate_sequence over a whole stored stream (.npz), in parallel blocks, cached next to it
    as reward_v2_<hash>_<stem>.npz. Blocks overlap by one row so each block's last score change and
    match-end test see the following row."""
    import hashlib
    import json
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict
    from pathlib import Path

    from blackout_env.train.offline_dataset import npz_member_memmap

    path = Path(path)
    key = hashlib.sha1(json.dumps({**asdict(cfg), "ppr": points_per_reward, "v": 2}, sort_keys=True).encode()).hexdigest()[:10]
    cache = path.parent / f"reward_v2_{key}_{path.stem}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    n = len(npz_member_memmap(path, "done"))
    tasks = [(str(path), s, min(s + block, n), cfg, points_per_reward, 1) for s in range(0, n, block)]
    results: dict[int, dict[str, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start, part in pool.map(_annotate_npz_range, tasks):
            results[start] = part
    merged = {k: np.concatenate([results[s][k] for s in sorted(results)]) for k in results[0]}
    np.savez(cache, **merged)
    return merged
