"""OfficialTracker: fork-format observations reconstructed from the official ones.

The simulator is *driven* instead of stepped: every tick each unit is put at its observed
position (``Unit.ChangePos`` -> region entry: deposits, sanctuary transforms) in ML-Agents'
AgentAct order and its interactions run (pick-ups, steals, combat), then the world update runs
(absorption, timers).  Positions are exact in both formats, so every rule fires as in Unity --
including on the official build, whose movement clamp differs from the fork's: movement itself
is never simulated.  What the official observation hides is then carried by the simulator:

* classes of the other units (transforms / deaths replayed; the observer's own class checks it),
* held and stored battery amounts: a deposit's true amount is read off the score jump (the tick
  is replayed with the corrected amount), a steal's from the stack it took,
* the absorption timer (from the elapsed tick count),
* field battery amounts: exact from the reset seed (``seed_search``), else a prior that keeps
  the generated total (~203 over the observed stacks) until each stack is deposited.

Anything the replay gets wrong (a float32 overlap decided the other way, a missed item pixel)
is resynced against the observation and counted in ``self.stats``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..sim import core
from ..sim.mapdata import CARRIER, H, R_SANCTUARY, SPAWN_CELL, STORAGE_CANDIDATES, STORAGE_COUNT, W
from ..sim.state import SimState, _timer_after, _timer_table, exact_position
from .render import HO, ID_ALLY_UNIT, ID_ENEMY_UNIT, ID_ITEM, SCALE, parse_vector

F32 = np.float32
PRIOR_TOTAL = 203          # E[sum of initial amounts]: the draw stops at the first total >= 200
PRIOR_AMOUNT = 4           # an unknown stack's amount (initial amounts are uniform 1..8)


@njit(cache=True)
def drive_tick(S, M, pos, dead, order, moved):
    """One tick with observed end positions: units in AgentAct order, then the world update.

    ``dead[u]``: u teleported (was killed this tick).  Its death is applied first: Unity killed
    it at a position the observation never shows, and keeping it anywhere on the field would let
    living units fight a ghost.  Respawned on its own base floor (enemies cannot enter, no items)
    it interacts with nothing, and its turn then moves it on to the observed position.  What
    the dying unit did earlier in the tick (a pick-up, rarely) is left to the resync."""
    g = S[5]
    for u in range(10):
        if dead[u] and g[core.G_OVER] == 0:
            core._die(S, M, u, -1)
    for j in range(order.shape[0]):
        u = order[j]
        if g[core.G_OVER]:
            break
        if not moved[u]:
            continue
        x, y = pos[u, 0], pos[u, 1]
        core._change_pos(S, M, u, x, y)
        if g[core.G_OVER] == 0:
            core._process_interactions(S, M, u, x, y)
    core.world_update(S, M)
    g[core.G_TICK] += 1


@njit(cache=True)
def world_ticks(S, M, n):
    for _ in range(n):
        core.world_update(S, M)
        S[5][core.G_TICK] += 1


def elapsed_from_time_left(time_left) -> int:
    _, _, obs_ep, _ = _timer_table()
    hits = np.nonzero(obs_ep == F32(time_left))[0]
    return int(hits[0]) if len(hits) else int(np.argmin(np.abs(obs_ep - F32(time_left))))


def chosen_from_official(gmap) -> tuple[int, ...]:
    """Storage candidates shown by the official map (sorted; the shuffle order is unobservable
    and only orders region ids)."""
    g = np.asarray(gmap)
    out = []
    for i, (_, rects) in enumerate(STORAGE_CANDIDATES):
        x, y = rects[0][0], rects[0][1]
        if g[HO - 1 - (SCALE * y + 1), SCALE * x + 1] == 2:   # ally storage pixel, off the item / unit centres
            out.append(i)
    if len(out) != STORAGE_COUNT:
        raise ValueError(f"official map shows storage candidates {out}")
    return tuple(out)


def tile_ids(gmap) -> np.ndarray:
    """The id at every tile's item pixel (tile centre), int [W, H]; 4/5 = hidden by a unit."""
    g = np.asarray(gmap)
    xs = SCALE * np.arange(W) + SCALE // 2
    ys = SCALE * np.arange(H) + SCALE // 2
    return g[HO - 1 - ys[None, :], xs[:, None]].astype(np.int64)


def prior_amounts(n: int, total: int = PRIOR_TOTAL) -> np.ndarray:
    base, extra = divmod(total, max(n, 1))
    a = np.full(n, base, np.int64)
    a[:extra] += 1
    return np.clip(a, 1, 8)


class OfficialTracker:
    """Feed one official observation per tick (any unit of the team, plus the map); read the
    fork-format ``raw_state`` / ``raw_map`` (or ``obs()``) the fork policies take.

    ``rng_finder``: callable(chosen, battery_tiles, prev_state) -> (generation RNG state, how) or
    None (``seed_search.RngFinder``).  A match whose state is found starts from
    ``SimState.from_rng_state``: field amounts exact, and the special spawns are predicted by the
    RNG (the map still corrects them); the next match is searched from where this one ends."""

    def __init__(self, rng_finder=None, order=None):
        self.rng_finder = rng_finder
        self.order = order
        self.sim: SimState | None = None
        self.elapsed = -1
        self.seed = None
        self.stats = {}

    # ------------------------------------------------------------------ reset
    def _start(self, p, gmap, elapsed):
        ids = tile_ids(gmap)
        chosen = chosen_from_official(gmap)
        tiles = [(x, y) for y in range(H) for x in range(W) if ids[x, y] == ID_ITEM]
        prev = self.sim.rng.copy() if self.sim is not None and self.seed is not None else None
        found = self.rng_finder(chosen, tiles, prev) if self.rng_finder else None
        self.seed = None if found is None else found[1]
        if found is not None:
            sim = SimState.from_rng_state(found[0], order=self.order, elapsed_ticks=elapsed)
        else:
            sim = SimState.empty(chosen, order=self.order)
            for (x, y), a in zip(tiles, prior_amounts(len(tiles))):
                sim.add_item(core.BATTERY, int(a), x, y)
            sim.g[core.G_SCORE_N] = int(sum(prior_amounts(len(tiles))))
            sim.gf[core.GF_EP] = _timer_after(elapsed, None)
            sim.gf[core.GF_ABS] = _timer_after(elapsed, 20.0)
            sim.g[core.G_TICK] = elapsed
        self.sim = sim
        self.elapsed = elapsed
        self.stats = {"ticks": 0, "amount_fix": 0, "score_override": 0, "held_fix": 0, "item_fix": 0,
                      "class_fix": 0, "pos_fix": 0, "spawns": 0, "seeded": self.seed}
        self._sync_positions(p, np.ones(10, bool))

    # ------------------------------------------------------------------ per tick
    def observe(self, vector, gmap):
        """Official float32[45] of any agent + uint8 [96, 96] team A id map (both as sent)."""
        p = parse_vector(vector)
        el = elapsed_from_time_left(p["time_left"])
        if self.sim is None or el <= min(self.elapsed, 2) or el < self.elapsed:
            self._start(p, gmap, el)
            return self
        if el == self.elapsed:
            return self
        self._tick(p, gmap, el - self.elapsed)
        self.elapsed = el
        return self

    def _targets(self, p):
        pos = np.zeros((10, 2), F32)
        for u in range(10):
            pos[u] = exact_position(p["pos_obs"][u, 0]), exact_position(p["pos_obs"][u, 1])
        return pos

    def _tick(self, p, gmap, n_ticks):
        sim = self.sim
        pos = self._targets(p)
        # a teleport: farther than any n-tick move (max speed 6 x 1.5 buff x 0.02 s) and ending
        # within one tick's move of the own spawn (the respawned unit may move on that tick)
        step = 0.2 * n_ticks
        dead = np.zeros(10, bool)
        for u in range(10):
            sx, sy = SPAWN_CELL[0 if u < 5 else 1]
            jump = np.abs(sim.uf[u].astype(np.float64) - pos[u]).max()
            near_spawn = np.abs(pos[u].astype(np.float64) - (sx + 0.5, sy + 0.5)).max() <= step
            dead[u] = jump > step and near_spawn
        moved = (sim.uf != pos).any(axis=1) | dead
        snap = sim.copy()
        obs_scores = p["scores"]
        for attempt in range(3):
            sim.g[core.G_NEV] = 0
            drive_tick(sim.S, sim.M, pos, dead, sim.order, moved)
            if n_ticks > 1:
                world_ticks(sim.S, sim.M, n_ticks - 1)
            if self._fix_amounts(snap, obs_scores):
                continue
            break
        self.stats["ticks"] += 1
        self._sync_items(gmap)
        self._sync_held(p)
        self._sync_class(p)
        self._sync_scores(obs_scores)
        self._sync_positions(p, np.zeros(10, bool))

    # ------------------------------------------------------------------ reconciliation
    def _fix_amounts(self, snap, obs_scores) -> bool:
        """Score jump != replayed deposit: restore the tick with the depositor's battery amount
        corrected (True = replay again)."""
        sim = self.sim
        d = [obs_scores[0] - int(sim.g[core.G_SCORE_A]), obs_scores[1] - int(sim.g[core.G_SCORE_B])]
        if d == [0, 0]:
            return False
        ev = sim.ev[: sim.g[core.G_NEV]]
        for e in ev:
            if e[1] == core.EV_DEPOSIT and e[3] == core.BATTERY:
                u = int(e[2])
                team = 0 if u < 5 else 1
                if d[team] == 0:
                    continue
                h = snap.ui[u, core.U_HELD]
                new = int(snap.it[h, core.I_AMT]) + d[team]
                if h < 0 or not 1 <= new <= 10:
                    continue
                snap.it[h, core.I_AMT] = new
                sim.load(snap)
                self.stats["amount_fix"] += 1
                return True
        return False

    def _sync_scores(self, obs_scores):
        sim = self.sim
        for t in (0, 1):
            if int(sim.g[core.G_SCORE_A + t]) != obs_scores[t]:
                sim.g[core.G_SCORE_A + t] = obs_scores[t]
                self.stats["score_override"] += 1

    def _sync_items(self, gmap):
        """Ground items against the map's item pixels (tiles hidden by a unit are skipped)."""
        sim = self.sim
        ids = tile_ids(gmap)
        have = np.full((W, H), -1, np.int64)
        slot = np.full((W, H), -1, np.int64)
        it = sim.it
        for i in np.nonzero((it[:, core.I_STATE] == core.GROUND) & (it[:, core.I_TX] >= 0))[0]:
            x, y = it[i, core.I_TX], it[i, core.I_TY]
            if have[x, y] < 0:
                have[x, y], slot[x, y] = it[i, core.I_TYPE], i
        visible = (ids != ID_ALLY_UNIT) & (ids != ID_ENEMY_UNIT)
        obs_t = np.where(ids >= ID_ITEM, ids - ID_ITEM, -1)
        for x, y in zip(*np.nonzero(visible & (obs_t != have))):
            t = int(obs_t[x, y])
            if have[x, y] >= 0:
                core._destroy(sim.S, int(slot[x, y]), False)
            if t >= 0:
                i = sim.add_item(t, PRIOR_AMOUNT if t == core.BATTERY else 1, int(x), int(y))
                if t != core.BATTERY:
                    sim.g[core.G_SPECIAL] = i
                    self.stats["spawns"] += 1
                    continue
            self.stats["item_fix"] += 1

    def _sync_held(self, p):
        sim = self.sim
        for u in range(10):
            h = sim.ui[u, core.U_HELD]
            have = 0 if h < 0 else int(sim.it[h, core.I_TYPE]) + 1
            want = int(p["held"][u])
            if have == want:
                continue
            if h >= 0:
                core._destroy(sim.S, int(h), False)
                sim.ui[u, core.U_HELD] = -1
            if want:
                sim.give_item(u, want - 1, PRIOR_AMOUNT if want == 1 else 1)
            self.stats["held_fix"] += 1

    def _sync_class(self, p):
        sim, me = self.sim, p["me"]
        if int(sim.ui[me, core.U_CLS]) != p["my_class"]:
            sim.ui[me, core.U_CLS] = p["my_class"]
            self.stats["class_fix"] += 1
            lay = sim.layout
            for r in range(lay.n_regions):
                if lay.reg_kind[r] == R_SANCTUARY and lay.reg_locked[r] and lay.reg_team[r] == p["team"]:
                    if p["my_class"] == CARRIER:
                        sim.reg_lock[r] = me
                    elif sim.reg_lock[r] == me:
                        sim.reg_lock[r] = -1

    def _sync_positions(self, p, force):
        sim = self.sim
        pos = self._targets(p)
        for u in range(10):
            if force[u] or (sim.uf[u] != pos[u]).any():
                if not force[u]:
                    self.stats["pos_fix"] += 1
                sim.place_unit(u, float(pos[u, 0]), float(pos[u, 1]))

    # ------------------------------------------------------------------ outputs
    def raw_state(self) -> np.ndarray:
        return self.sim.raw_state()

    def raw_map(self) -> np.ndarray:
        return self.sim.raw_map()

    def obs(self, agents=None) -> dict:
        return self.sim.obs(agents)
