"""SimState: the explicit game state, its constructors and the Python-facing step / render API.

    from blackout_env.sim import SimState
    sim = SimState.from_seed(seed)                       # exact match start (Unity's own RNG)
    sim = SimState.from_raw(raw_state, raw_map)          # from one MapObsAgent broadcast
    sim = SimState.from_obs(obs)                         # from a BlackOutEnv observation dict
    sim.step(moves)                                      # float32 [10, 2], one decision (2 ticks)
    obs = sim.obs()                                      # BlackOutEnv-format observations
    fork = sim.copy()                                    # cheap deep copy for rollouts

Hidden state that one observation does not show (``from_raw`` / ``from_obs``):

* the special-item spawn timer -- 0 while a special item exists (it is reset on spawn and only
  runs while none is active); otherwise unknown in [0, 10) s: pass ``spawn_timer`` (default 0,
  i.e. the next spawn comes a full cooldown later) or a sampled value;
* UnityEngine.Random's state -- exact only from the reset seed (``from_seed``, or
  ``rng_state=`` + the number of spawns so far); otherwise a random state, which samples
  spawns from Unity's own distribution (uniform type, uniform free neutral floor tile);
* the order of objects inside one tile's ``MapObjects`` list (only matters when one unit
  overlaps several objects on the same tile in the same tick): items first, then units by index;
* the insertion order of stat modifiers (only float rounding of their sum can differ).

Everything else is recovered exactly: positions (float32 inverted through the obs
normalisation), classes, held items and amounts, ground items, scores, both timers (from the
elapsed tick count), sanctuary locks (a living Carrier holds its team's lock), applied effects.
"""

from __future__ import annotations

import numpy as np

from . import core
from .mapdata import (CARRIER, H, R_SANCTUARY, R_STORAGE, SPAWN_CELL, TILE_COLLISION, BLOCK_ALL, W, Layout,
                      SITE_CARRIER, SITE_HUNTER, build_layout, chosen_from_map)
from .unity_random import generate_state, init_state

F32 = np.float32
# Agent order of ML-Agents' AgentAct (OnEnable order of the BlackOutAgent components); validated
# against recorded traces in examples/sim_validate.py.
DEFAULT_ORDER = np.array([3, 6, 2, 7, 5, 8, 4, 9, 1, 0], dtype=np.int64)
TICKS_PER_SECOND = 50


def background(lay: Layout) -> np.ndarray:
    """SemanticMapRenderer.RenderBackground for team A, rows top-down (obs layout)."""
    out = np.zeros((H, W), dtype=np.uint16)
    for x in range(W):
        for y in range(H):
            t, r = lay.tile[x, y], lay.region[x, y]
            if TILE_COLLISION[t] == BLOCK_ALL:
                v = 1
            elif lay.reg_kind[r] == R_STORAGE:
                v = 6 if lay.reg_team[r] == 0 else 7
            elif t == SITE_HUNTER:
                v = 2
            elif t == SITE_CARRIER:
                v = 3
            elif (x, y) == SPAWN_CELL[0]:
                v = 4
            elif (x, y) == SPAWN_CELL[1]:
                v = 5
            else:
                v = 0
            out[H - 1 - y, x] = v
    return out


_TIMER_TABLE = None
MAX_TICKS = 21200


def _timer_table():
    """GameTimer.CurrentTime after n ticks (float32 accumulation of dt) for every n: the episode
    timer, the looping absorption timer, and their rendered time-left observations."""
    global _TIMER_TABLE
    if _TIMER_TABLE is None:
        ep = np.zeros(MAX_TICKS + 1, dtype=np.float32)
        ab = np.zeros(MAX_TICKS + 1, dtype=np.float32)
        e = a = F32(0.0)
        dt, wrap = F32(core.DT), F32(20.0)
        for n in range(1, MAX_TICKS + 1):
            e = F32(e + dt)
            a = F32(a + dt)
            if a >= wrap:
                a = F32(a - wrap)
            ep[n], ab[n] = e, a
        one, zero = F32(1), F32(0)
        obs_ep = (one - np.clip(ep / core.EPISODE_SECONDS, zero, one)).astype(np.float32)
        obs_ab = (one - np.clip(ab / core.ABSORPTION_SECONDS, zero, one)).astype(np.float32)
        _TIMER_TABLE = (ep, ab, obs_ep, obs_ab)
    return _TIMER_TABLE


def _timer_after(ticks: int, wrap: float | None) -> np.float32:
    """GameTimer.CurrentTime after ``ticks`` float32 additions of dt (looping at ``wrap``)."""
    ep, ab, _, _ = _timer_table()
    return ab[ticks] if wrap is not None else ep[ticks]


def elapsed_ticks_of(raw_state) -> int:
    """Ticks since the match started: the n whose float32 timers render exactly the observed
    episode and absorption time-left values (the accumulated float32 timer drifts from n * dt
    by several ticks late in a match, so n cannot be read off time_left directly)."""
    raw_state = np.asarray(raw_state, dtype=np.float32)
    _, _, obs_ep, obs_ab = _timer_table()
    hits = np.nonzero((obs_ep == raw_state[42]) & (obs_ab == raw_state[43]))[0]
    if len(hits):
        return int(hits[0])
    return int(np.argmin(np.abs(obs_ep - raw_state[42])))


def exact_position(o: np.float32) -> np.float32:
    """The float32 world coordinate whose observation ``pos / 24 * 2 - 1`` equals ``o``."""
    o = F32(o)
    guess = F32(F32(F32(o + F32(1.0)) / F32(2.0)) * F32(24.0))
    cands = [guess]
    up = dn = guess
    for _ in range(8):
        up, dn = np.nextafter(up, F32(np.inf)), np.nextafter(dn, F32(-np.inf))
        cands += [up, dn]
    for c in cands:
        if F32(F32(c / F32(24.0)) * F32(2.0)) - F32(1.0) == o:
            return c
    return guess


def position_candidates(o) -> list:
    """Every float32 coordinate whose observation ``pos / 24 * 2 - 1`` equals ``o``."""
    o = F32(o)
    guess = exact_position(o)
    out = [guess]
    for direction in (F32(np.inf), F32(-np.inf)):
        c = guess
        for _ in range(4):
            c = np.nextafter(c, direction)
            if F32(F32(c / F32(24.0)) * F32(2.0)) - F32(1.0) == o:
                out.append(c)
            else:
                break
    return out


def refine_positions(raw_states, raw_maps, moves) -> np.ndarray:
    """Exact float32 positions at the last broadcast, from a short history.

    One observation leaves each coordinate ambiguous between 1-3 float32 values.  A unit's own
    movement depends only on its position, class and team speed modifiers (walls and bases,
    not other units), so each candidate is moved through every recorded step (2 ticks, the
    simulator's own movement code) and kept only if it renders the next observation; a death
    or class change restarts that unit's candidates.  Coordinates that stay ambiguous fall back
    to ``exact_position``."""
    raw_states = np.asarray(raw_states, dtype=np.float32)
    n = len(raw_states)
    last = raw_states[-1]
    out = np.zeros((10, 2), dtype=np.float32)
    sims = [SimState.from_raw(raw_states[k], raw_maps[k]) for k in range(n - 1)]
    for u in range(10):
        team = 0 if u < 5 else 1
        cands = None
        for k in range(n):
            obs_x, obs_y = raw_states[k][4 * u], raw_states[k][4 * u + 1]
            fresh = [(x, y) for x in position_candidates(obs_x) for y in position_candidates(obs_y)]
            if cands is None:
                cands = fresh
            else:
                fresh_set = {(float(x), float(y)) for x, y in fresh}
                moved = []
                sim = sims[k - 1]
                spd = F32(core._speed(sim.S, u))
                r = core.CLS_WIDTH[int(raw_states[k - 1][4 * u + 3])] / core.F2
                nx, ny = map(F32, core._normalized(F32(moves[k - 1][u][0]), F32(moves[k - 1][u][1])))
                dt = sim.gf[core.GF_DT]
                dx, dy = F32(F32(nx * dt) * spd), F32(F32(ny * dt) * spd)
                for (x, y) in cands:
                    for _ in range(2):  # (numba returns Python floats: keep the float32 typing)
                        x, y = map(F32, core._clamp_axis(sim.M, F32(x), F32(y), dx, core.F0, r, team))
                        x, y = map(F32, core._clamp_axis(sim.M, x, y, core.F0, dy, r, team))
                    if (float(x), float(y)) in fresh_set:
                        moved.append((x, y))
                same_class = raw_states[k][4 * u + 3] == raw_states[k - 1][4 * u + 3]
                cands = moved if (moved and same_class) else fresh
        if len({(float(x), float(y)) for x, y in cands}) == 1:
            out[u] = cands[0]
        else:
            out[u] = (exact_position(last[4 * u]), exact_position(last[4 * u + 1]))
    return out


class SimState:
    def __init__(self, lay: Layout, order=None):
        self.layout = lay
        self.order = np.asarray(DEFAULT_ORDER if order is None else order, dtype=np.int64)
        self.uf = np.zeros((10, 2), dtype=np.float32)
        self.ui = np.full((10, 4), -1, dtype=np.int32)
        self.it = np.zeros((core.MAXI, 9), dtype=np.int32)
        self.it[:, core.I_TYPE] = -1
        self.it[:, core.I_TX:core.I_TY + 1] = -1
        self.it[:, core.I_CTX] = -1
        self.it[:, core.I_HOLDER] = -1
        self.it[:, core.I_MODTEAM] = -1
        self.tl = np.zeros((W, H, core.MAXK), dtype=np.int32)
        self.tn = np.zeros((W, H), dtype=np.int32)
        self.g = np.zeros(core.NG, dtype=np.int64)
        self.g[core.G_SPECIAL] = -1
        self.g[core.G_WINNER] = -1
        self.gf = np.zeros(4, dtype=np.float32)
        self.gf[core.GF_DT] = core.DT
        self.rng = init_state(np.random.randint(1, 2**31 - 1))
        self.ev = np.zeros((core.MAXEV, 6), dtype=np.int32)
        self.reg_lock = np.full(lay.n_regions, -1, dtype=np.int32)
        self.fs = np.zeros((core.MAXF, 4), dtype=np.int32)
        self.ml = np.zeros((3, core.MAXM), dtype=np.int32)
        self.mn = np.zeros(3, dtype=np.int32)
        self.bg = background(lay)
        self.M = (lay.tile, lay.region, lay.reg_kind, lay.reg_team, lay.reg_in, lay.reg_out,
                  lay.reg_locked, lay.reg_tiles, lay.reg_ntiles, lay.walk, lay.half)

    # ------------------------------------------------------------------ plumbing
    @property
    def S(self):
        return (self.uf, self.ui, self.it, self.tl, self.tn, self.g, self.gf, self.rng, self.ev,
                self.reg_lock, self.fs, self.ml, self.mn)

    _ARRAYS = ("uf", "ui", "it", "tl", "tn", "g", "gf", "rng", "ev", "reg_lock", "fs", "ml", "mn")

    def copy(self) -> "SimState":
        new = object.__new__(SimState)
        new.layout, new.order, new.bg, new.M = self.layout, self.order, self.bg, self.M
        for name in self._ARRAYS:
            setattr(new, name, getattr(self, name).copy())
        return new

    def load(self, other: "SimState") -> None:
        """Overwrite this state with ``other`` in place (same layout), no allocation."""
        for name in self._ARRAYS:
            getattr(self, name)[...] = getattr(other, name)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_seed(cls, seed: int, order=None, elapsed_ticks: int = 2) -> "SimState":
        """The state Unity starts a match in after ``env.reset(seed=seed)`` (first observation),
        with Unity's RNG state after generation: special spawns then follow Unity exactly.

        ``elapsed_ticks``: world-update ticks Unity ran before that first observation -- 2 for
        every match but the first one after the player starts (0; that match's first decision
        then moves once).  Read it off the observation with ``elapsed_ticks_of``."""
        return cls.from_rng_state(init_state(seed), order, elapsed_ticks)

    @classmethod
    def from_rng_state(cls, state, order=None, elapsed_ticks: int = 2) -> "SimState":
        """``from_seed`` for generation starting at an arbitrary UnityEngine.Random state."""
        chosen, batteries, rng = generate_state(state)
        sim = cls(build_layout(chosen), order)
        S = sim.S
        for u in range(10):  # MatchManager.ResetAllUnits: Teleport each unit in order
            sx, sy = SPAWN_CELL[0 if u < 5 else 1]
            sim.ui[u] = (core.COLLECTOR, -1, -1, -1)
            core._change_pos(S, sim.M, u, F32(sx) + core.FHALF, F32(sy) + core.FHALF)
        for x, y, a in batteries:
            core.new_item(S, sim.M, core.BATTERY, a, x, y)
        sim.rng[:] = rng
        sim.g[core.G_SCORE_A] = sim.g[core.G_SCORE_B] = 0
        sim.g[core.G_SCORE_N] = sum(a for _, _, a in batteries)
        for _ in range(elapsed_ticks):  # no unit moves on these ticks
            core.world_update(S, sim.M)
            sim.g[core.G_TICK] += 1
        sim.g[core.G_FIRST] = 1 if elapsed_ticks == 0 else 0
        return sim

    @classmethod
    def from_raw(cls, raw_state, raw_map, *, order=None, spawn_timer: float = 0.0, rng_state=None,
                 elapsed_ticks: int | None = None, positions=None) -> "SimState":
        """``positions``: exact float32 unit positions [10, 2] when known (``refine_positions``);
        otherwise each coordinate is inverted from the observation, which is ambiguous by an
        ulp or two (pos / 24 maps several float32 positions onto one value)."""
        raw_state = np.asarray(raw_state, dtype=np.float32)
        raw_map = np.asarray(raw_map).astype(np.int64)
        sim = cls(build_layout(chosen_from_map(raw_map)), order)
        S, M = sim.S, sim.M
        # ground items first (MapObjects order: items were there before units walked in)
        for y in range(H):
            for x in range(W):
                v = int(raw_map[H - 1 - y, x])
                n, t = (v >> 3) & 15, (v >> 7) & 7
                if n:
                    core.new_item(S, M, core.BATTERY, n, x, y)
                if t:
                    i = core.new_item(S, M, t, 1, x, y)
                    sim.g[core.G_SPECIAL] = i
        for u in range(10):
            block = raw_state[4 * u: 4 * u + 4]
            if positions is not None:
                x, y = F32(positions[u][0]), F32(positions[u][1])
            else:
                x, y = exact_position(block[0]), exact_position(block[1])
            sim.ui[u, core.U_CLS] = int(round(float(block[3])))
            sim.ui[u, core.U_HELD] = -1
            # placed, not moved: the unit entered its region before this observation (a
            # sanctuary or storage entry must not fire again)
            cx, cy = int(np.floor(x)), int(np.floor(y))
            core._tile_add(S, cx, cy, u)
            sim.ui[u, core.U_TX], sim.ui[u, core.U_TY] = cx, cy
            sim.uf[u] = (x, y)
            code = float(block[2])
            if code != 0.0:
                t, a = (core.BATTERY, int(round(-code))) if code < 0 else (int(round(code)) - 1, 1)
                for i in range(core.MAXI):
                    if sim.it[i, core.I_STATE] == core.POOL:
                        sim.it[i] = (t, a, core.CARRIED, -1, -1, -1, u, -1, 0)
                        break
                sim.ui[u, core.U_HELD] = i
                if t != core.BATTERY:
                    sim.g[core.G_SPECIAL] = i
        # sanctuary locks: a living Carrier holds its team's locked sanctuary
        lay = sim.layout
        for r in range(lay.n_regions):
            if lay.reg_kind[r] == R_SANCTUARY and lay.reg_locked[r]:
                team = lay.reg_team[r]
                for u in range(5 * team, 5 * team + 5):
                    if sim.ui[u, core.U_CLS] == CARRIER:
                        sim.reg_lock[r] = u
        sim.g[core.G_SCORE_A] = int(round(float(raw_state[40]) * 100))
        sim.g[core.G_SCORE_B] = int(round(float(raw_state[41]) * 100))
        sim.g[core.G_SCORE_N] = 0
        if elapsed_ticks is None:
            elapsed_ticks = elapsed_ticks_of(raw_state)
        sim.gf[core.GF_EP] = _timer_after(elapsed_ticks, None)
        sim.gf[core.GF_ABS] = _timer_after(elapsed_ticks, 20.0)
        sim.gf[core.GF_SPAWN] = 0.0 if sim.g[core.G_SPECIAL] >= 0 else F32(spawn_timer)
        sim.g[core.G_FIRST] = 1 if elapsed_ticks == 0 else 0
        sim.g[core.G_TICK] = elapsed_ticks
        if rng_state is not None:
            sim.rng[:] = np.asarray(rng_state, dtype=np.uint32)
        return sim

    @classmethod
    def empty(cls, chosen=(0, 1, 2, 3), order=None) -> "SimState":
        """Every unit a Collector on its spawn, no items, timers at 0 (for scenario tests)."""
        sim = cls(build_layout(chosen), order)
        for u in range(10):
            sx, sy = SPAWN_CELL[0 if u < 5 else 1]
            sim.ui[u] = (core.COLLECTOR, -1, -1, -1)
            core._change_pos(sim.S, sim.M, u, F32(sx) + core.FHALF, F32(sy) + core.FHALF)
        sim.no_spawns()
        return sim

    def place_unit(self, u: int, x: float, y: float, cls: int | None = None) -> None:
        """Put unit ``u`` at (x, y) (no region entry fires), optionally setting its class."""
        ox, oy = self.ui[u, core.U_TX], self.ui[u, core.U_TY]
        if ox >= 0:
            core._tile_remove(self.S, ox, oy, u)
        cx, cy = int(np.floor(x)), int(np.floor(y))
        core._tile_add(self.S, cx, cy, u)
        self.ui[u, core.U_TX], self.ui[u, core.U_TY] = cx, cy
        self.uf[u] = (F32(x), F32(y))
        if cls is not None:
            self.ui[u, core.U_CLS] = cls

    def add_item(self, t: int, amount: int, x: int, y: int) -> int:
        """An item lying on cell (x, y); effects of a storage cell apply (score, modifiers)."""
        return core.new_item(self.S, self.M, t, amount, x, y)

    def give_item(self, u: int, t: int, amount: int) -> int:
        for i in range(core.MAXI):
            if self.it[i, core.I_STATE] == core.POOL:
                self.it[i] = (t, amount, core.CARRIED, -1, -1, -1, u, -1, 0)
                self.ui[u, core.U_HELD] = i
                return i
        raise RuntimeError("no free item slot")

    @classmethod
    def from_history(cls, raw_states, raw_maps, moves, **kw) -> "SimState":
        """``from_raw`` on the last of several consecutive broadcasts, with unit positions made
        exact by ``refine_positions`` from the earlier ones and the moves applied in between
        (``moves[k]`` taken after ``raw_states[k]``)."""
        pos = refine_positions(raw_states, raw_maps, moves)
        return cls.from_raw(raw_states[-1], raw_maps[-1], positions=pos, **kw)

    @classmethod
    def from_obs(cls, obs: dict, **kw) -> "SimState":
        """From a BlackOutEnv observation dict (any agent's view; both teams' units are in it)."""
        view = next(iter(obs.values()))
        return cls.from_raw(*raw_from_view(view), **kw)

    # ------------------------------------------------------------------ dynamics
    def step(self, moves, moves_b=None) -> None:
        """One decision step.  ``moves``: float32 [10, 2] for all units (unit index order), or
        team A's [5, 2] with ``moves_b`` team B's [5, 2]."""
        if moves_b is not None:
            moves = np.concatenate([np.asarray(moves, F32).reshape(5, 2), np.asarray(moves_b, F32).reshape(5, 2)])
        core.step_kernel(self.S, self.M, np.ascontiguousarray(moves, dtype=F32), self.order)

    def run(self, moves_seq) -> np.ndarray:
        """Steps through a [T, 10, 2] sequence; returns positions [T', 10, 2] (stops at game end)."""
        moves_seq = np.ascontiguousarray(moves_seq, dtype=F32)
        out = np.zeros_like(moves_seq)
        n = core.run_kernel(self.S, self.M, moves_seq, self.order, out)
        return out[:n]

    def force_spawns(self, spawns) -> None:
        """Special spawns are taken in order from ``spawns`` [(type 1-4, x, y), ...] instead of
        the RNG (validation against a recorded trace); when exhausted no more spawn."""
        spawns = np.asarray(spawns, dtype=np.int32).reshape(-1, 3)
        self.fs[: len(spawns), :3] = spawns
        self.g[core.G_FS_N] = len(spawns)
        self.g[core.G_FS_I] = 0
        self.g[core.G_SPAWN_MODE] = core.SPAWN_FORCED

    def force_timed_spawns(self, spawns) -> None:
        """Special spawns [(type, x, y, tick), ...] placed at world-update number ``tick`` (the
        elapsed-tick count after that update), bypassing the hidden spawn timer."""
        spawns = np.asarray(spawns, dtype=np.int32).reshape(-1, 4)
        self.fs[: len(spawns)] = spawns
        self.g[core.G_FS_N] = len(spawns)
        self.g[core.G_FS_I] = 0
        self.g[core.G_SPAWN_MODE] = core.SPAWN_TIMED

    def no_spawns(self) -> None:
        self.g[core.G_SPAWN_MODE] = core.SPAWN_NONE

    def reseed(self, seed: int) -> None:
        """Fresh RNG state for sampled spawns (Unity's distribution)."""
        self.rng[:] = init_state(seed)
        self.g[core.G_SPAWN_MODE] = core.SPAWN_RNG

    # ------------------------------------------------------------------ views
    @property
    def over(self) -> bool:
        return bool(self.g[core.G_OVER])

    @property
    def winner(self) -> int:
        return int(self.g[core.G_WINNER])

    @property
    def scores(self) -> tuple[int, int]:
        return int(self.g[core.G_SCORE_A]), int(self.g[core.G_SCORE_B])

    @property
    def positions(self) -> np.ndarray:
        return self.uf.copy()

    def events(self) -> np.ndarray:
        """This step's events: int32 [n, 6] (tick, kind, a, b, c, d), see core.EV_*."""
        return self.ev[: self.g[core.G_NEV]].copy()

    def raw_state(self) -> np.ndarray:
        out = np.zeros(44, dtype=np.float32)
        core.render_raw_state(self.S, out)
        return out

    def raw_map(self) -> np.ndarray:
        out = np.zeros((H, W), dtype=np.uint16)
        core.render_raw_map(self.S, self.bg, out)
        return out

    def obs(self, agents=None) -> dict:
        """BlackOutEnv observations ({agent: {graphic, team_state, agent_states}}) rendered from
        this state through the env's own preprocessor."""
        return obs_from_raw(self.raw_state(), self.raw_map(), agents)


# ---------------------------------------------------------------------- env-format conversion
_PRE = None


def _preprocessor():
    global _PRE
    if _PRE is None:
        from blackout_env.env.my_obs_preprocessor import MyObsPreprocessor
        from blackout_env.env.obs_preprocessor import load_semantic_config
        from blackout_env.env.blackout_env import BlackOutEnv

        cfg = load_semantic_config(BlackOutEnv._DEFAULT_CONFIG)
        _PRE = MyObsPreprocessor(cfg, n_items=cfg["n_items"], n_classes=cfg["n_classes"])
    return _PRE


def obs_from_raw(raw_state, raw_map, agents=None) -> dict:
    """Exactly BlackOutEnv._collect_map_obs + _build_obs for a raw broadcast."""
    pre = _preprocessor()
    visual = (np.asarray(raw_map, dtype=np.float32) / F32(pre.PACK_DIVISOR))[:, :, None]
    ga, gb = pre.preprocess_team_graphics(visual)
    sa, sb = pre.preprocess_agent_states(np.asarray(raw_state, dtype=np.float32))
    ta, tb = pre.preprocess_team_states(np.asarray(raw_state, dtype=np.float32))
    names = [f"unit_{i}" for i in range(10)] if agents is None else agents
    out = {}
    for name in names:
        b = int(name.split("_")[1]) >= 5
        out[name] = {"graphic": gb if b else ga, "team_state": tb if b else ta, "agent_states": sb if b else sa}
    return out


def raw_from_view(view: dict) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the preprocessor for one view: the raw 44-float broadcast and packed map."""
    states = np.asarray(view["agent_states"], dtype=np.float32)
    team_b = states[0, 2] < 0
    raw = np.zeros(44, dtype=np.float32)
    for u in range(10):
        row = states[u]
        raw[4 * u: 4 * u + 2] = row[:2]
        items = row[3:9]
        if items[1] > 0:
            raw[4 * u + 2] = -np.round(items[1] * 15.0)
        else:
            k = int(np.argmax(items))
            raw[4 * u + 2] = 0.0 if k == 0 or items[k] <= 0 else float(k)
        raw[4 * u + 3] = float(np.argmax(row[9:12]))
    ts = np.asarray(view["team_state"], dtype=np.float32)
    raw[40], raw[41] = (ts[1], ts[0]) if team_b else (ts[0], ts[1])
    raw[42], raw[43] = ts[2], ts[3]
    gr = np.asarray(view["graphic"])
    base = np.argmax(gr[..., :8], axis=-1)
    if team_b:  # undo flip_team_perspective
        base = np.select([base == 4, base == 5, base == 6, base == 7], [5, 4, 7, 6], base)
    bat = np.rint(gr[..., 8] * 15.0).astype(np.int64)
    item = np.zeros(base.shape, dtype=np.int64)
    for k in range(gr.shape[-1] - 9):
        item[gr[..., 9 + k] > 0.5] = k + 1
    packed = base | (bat << 3) | (item << 7)
    return raw, packed.astype(np.uint16)
