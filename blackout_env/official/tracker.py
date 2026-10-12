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
from ..sim.state import SimState, _timer_after, _timer_table, exact_position, position_candidates
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
        # Unit.Move runs every tick, also for a zero input: the interactions too (a special item
        # spawned under a standing Collector is picked up on the next tick)
        if moved[u]:
            core._change_pos(S, M, u, pos[u, 0], pos[u, 1])
        if g[core.G_OVER] == 0:
            core._process_interactions(S, M, u, S[0][u, 0], S[0][u, 1])
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


def informative(gmap) -> bool:
    """False for a missing map or the constant frame a headless (``-nographics``) official build
    sends (its RenderTexture readback never sees the semantic texture)."""
    return gmap is not None and int((np.asarray(gmap) == 1).sum()) > 100


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
        prev = self.sim.rng.copy() if self.sim is not None and self.seed is not None else None
        if informative(gmap):
            ids = tile_ids(gmap)
            chosen = chosen_from_official(gmap)
            tiles = [(x, y) for y in range(H) for x in range(W) if ids[x, y] == ID_ITEM]
            found = self.rng_finder(chosen, tiles, prev) if self.rng_finder else None
        else:   # headless official build: the map is a constant -- follow the RNG stream blind
            chosen, tiles = (0, 1, 2, 3), []
            found = self.rng_finder.blind(prev) if self.rng_finder else None
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

        def attempt(pv):
            sim.load(snap)
            for _ in range(3):
                sim.g[core.G_NEV] = 0
                drive_tick(sim.S, sim.M, pv, dead, sim.order, moved | (pv != sim.uf).any(axis=1))
                if n_ticks > 1:
                    world_ticks(sim.S, sim.M, n_ticks - 1)
                if not self._fix_amounts(snap, obs_scores):
                    break
            return self._inconsistent(p, dead)

        bad, suspects = attempt(pos)
        if bad:
            # Auto-correction: a float32 position is only known up to 1-3 values (pos / 24 is
            # many-to-one); an overlap decided on that edge picks up / kills differently.  Replay
            # the tick with the other candidates of the units involved; keep the first variant the
            # observation agrees with (else the best, then resync).
            best = (bad, pos)
            for u in suspects:
                for cx in position_candidates(p["pos_obs"][u, 0]):
                    for cy in position_candidates(p["pos_obs"][u, 1]):
                        if cx == pos[u, 0] and cy == pos[u, 1]:
                            continue
                        pv = pos.copy()
                        pv[u] = cx, cy
                        b_, _ = attempt(pv)
                        if b_ < best[0]:
                            best = (b_, pv)
                        if b_ == 0:
                            break
                    if best[0] == 0:
                        break
                if best[0] == 0:
                    break
            if best[1] is not pos:
                self.stats["pos_retry"] = self.stats.get("pos_retry", 0) + 1
            attempt(best[1])
        self.stats["ticks"] += 1
        if informative(gmap):
            self._sync_items(gmap)
        self._sync_held(p)
        self._sync_class(p)
        self._sync_scores(obs_scores)
        self._sync_positions(p, np.zeros(10, bool))

    def _inconsistent(self, p, dead):
        """(number of contradictions with the observation, units involved): held item types,
        deaths (replayed victims vs observed teleports), the observer's class."""
        sim = self.sim
        bad, sus = 0, []
        for u in range(10):
            h = sim.ui[u, core.U_HELD]
            have = 0 if h < 0 else int(sim.it[h, core.I_TYPE]) + 1
            if have != int(p["held"][u]):
                bad += 1
                sus.append(u)
        ev = sim.ev[: sim.g[core.G_NEV]]
        victims = {int(e[3]) for e in ev if e[1] == core.EV_KILL}
        for u in range(10):
            if (u in victims) != bool(dead[u]):
                bad += 1
                sus.append(u)
                sus += [int(e[2]) for e in ev if e[1] == core.EV_KILL and e[3] == u and e[2] >= 0]
        if int(sim.ui[p["me"], core.U_CLS]) != p["my_class"]:
            bad += 1
        return bad, list(dict.fromkeys(sus))

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
        rendered = (sim.uf / core.BOUNDS * core.F2 - core.F1).astype(F32)
        for u in range(10):
            if force[u] or (rendered[u] != p["pos_obs"][u]).any():
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


class _Fixed:
    """rng_finder that hands out one fixed generation state."""

    def __init__(self, state, how):
        self.state, self.how = state, how

    def __call__(self, chosen, tiles, prev=None):
        return self.state, self.how

    def blind(self, prev=None):
        return self.state, self.how


FIX_KEYS = ("amount_fix", "score_override", "held_fix", "item_fix", "class_fix")


class MultiTracker:
    """Several RNG-stream hypotheses tracked side by side; the ones the observation contradicts
    are dropped.  Needed when the map cannot pin the stream (the headless official build) and
    the number of generations between two matches is not certain: a match ending inside Unity
    restarts it at once (EndEpisode -> OnEpisodeBegin -> a generation) and ``env.reset()`` then
    generates again, so the next match starts 0..``max_gens`` generations after the last state.

    A wrong hypothesis lays batteries on other tiles: the first pick-up, deposit or score jump
    contradicts it (a resync fix).  Until then the most likely one (``order``) is served."""

    def __init__(self, rng_finder, max_gens: int = 3, prefer=(1, 0, 2, 3), order=None):
        self.finder, self.max_gens, self.prefer, self.order = rng_finder, max_gens, prefer, order
        self.children: list[OfficialTracker] = []
        self.best: OfficialTracker | None = None
        self.elapsed = -1
        self.dropped = 0

    def _candidates(self, gmap):
        from ..sim.unity_random import generate_state

        if self.best is not None and self.best.seed is not None:
            base, tag = self.best.sim.rng.copy(), "prev"
        else:
            base, tag = self.finder.blind(None)[0], self.finder.blind(None)[1]
        states, s = [], np.array(base, dtype=np.uint32)
        for k in range(self.max_gens + 1):
            states.append((s.copy(), f"{tag}+{k}gen"))
            s = generate_state(s)[2]
        # a fresh player's first match generates right after the startup one (k = 0); later
        # matches follow an in-Unity restart plus env.reset() (k = 1 first)
        prefer = self.prefer if tag == "prev" else tuple(range(self.max_gens + 1))
        rank = {k: i for i, k in enumerate(prefer)}
        return [states[k] for k in sorted(range(len(states)), key=lambda k: rank.get(k, 99))]

    def observe(self, vector, gmap=None):
        p = parse_vector(vector)
        el = elapsed_from_time_left(p["time_left"])
        new = self.best is None or el <= min(self.elapsed, 2) or el < self.elapsed
        if new and informative(gmap):
            prev = self.best.sim.rng.copy() if self.best is not None and self.best.seed is not None else None
            t = OfficialTracker(rng_finder=_PrevFinder(self.finder, prev), order=self.order)
            self.children = [t]
        elif new:
            self.children = [OfficialTracker(rng_finder=_Fixed(st, how), order=self.order)
                             for st, how in self._candidates(gmap)]
        for c in self.children:
            c.observe(vector, gmap)
        if len(self.children) > 1:
            score = [sum(c.stats.get(k, 0) for k in FIX_KEYS) for c in self.children]
            m = min(score)
            keep = [c for c, s in zip(self.children, score) if s == m]
            self.dropped += len(self.children) - len(keep)
            self.children = keep
        self.best = self.children[0]
        self.elapsed = el
        return self

    @property
    def sim(self):
        return self.best.sim

    @property
    def seed(self):
        return self.best.seed

    @property
    def stats(self):
        return dict(self.best.stats, hypotheses=len(self.children), dropped=self.dropped)

    def raw_state(self):
        return self.best.raw_state()

    def raw_map(self):
        return self.best.raw_map()

    def obs(self, agents=None):
        return self.best.obs(agents)


class _PrevFinder:
    def __init__(self, finder, prev):
        self.finder, self.prev = finder, prev

    def __call__(self, chosen, tiles, prev=None):
        return self.finder(chosen, tiles, self.prev)
