"""Recorded Unity traces (examples/sim_record_traces.py): loading, observation-level events, and
open-loop replay through the simulator for validation (examples/sim_validate.py).

Events are extracted from *observation sequences* by the same function for Unity and for the
simulator (``obs_events``), so the comparison never depends on simulator internals:

  kill       a unit is back on its spawn cell centre as a Collector after being elsewhere
  pickup     held code 0 -> non-zero (no death); ``steal`` when the cell it stood next to lost
             an item from an enemy storage (classified by the map diff)
  deposit    held code non-zero -> 0 without dying, and the unit's team storage gained it
  lost       held code non-zero -> 0 by death or a Hunter transformation
  transform  class change without death
  absorb     the absorption timer wrapped (all storage items gone)
  spawn      a special item appears on a neutral floor cell
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import core
from .mapdata import H, NEUTRAL, SPAWN_CELL, build_layout, chosen_from_map
from .state import SimState, elapsed_ticks_of

F32 = np.float32


@dataclass
class Trace:
    raw_state: np.ndarray   # float32 [T, 44]
    raw_map: np.ndarray     # uint16 [T, 24, 24]
    moves: np.ndarray       # float32 [T, 10, 2]
    meta: dict
    path: str = ""

    @property
    def steps(self) -> int:
        """Usable steps: the last row is the terminal broadcast (may already be the next match)."""
        return len(self.raw_state) - 1

    @classmethod
    def load(cls, path) -> "Trace":
        d = np.load(path)
        rs = d["raw_state"]
        # The env's terminal step can carry the reset broadcast of the next match: drop it.
        n = len(rs)
        if n >= 2 and (rs[-1, 42] > rs[-2, 42] + 1e-6):
            n -= 1
        return cls(rs[:n], d["raw_map"][:n], d["moves"][:n], json.loads(str(d["meta"])), str(path))


def positions(raw_state: np.ndarray) -> np.ndarray:
    """World positions [T, 10, 2] (float64) from raw states (exact float32 inversion)."""
    rs = np.asarray(raw_state, dtype=np.float32)
    flat = rs[..., :40].reshape(*rs.shape[:-1], 10, 4)[..., :2]
    return ((flat.astype(np.float64) + 1.0) * 12.0)


def _specials(raw_map) -> int:
    return int(np.count_nonzero((np.asarray(raw_map).astype(np.int64) >> 7) & 7))


def special_exists(rs, rm) -> bool:
    held = rs[2:40:4]
    return _specials(rm) > 0 or bool(np.any(held >= 2))


def detect_spawns(tr: Trace, lay=None):
    """[(step, type, x, y)]: special items appearing on neutral cells (not via a deposit)."""
    lay = lay or build_layout(chosen_from_map(tr.raw_map[0]))
    out = []
    prev = (tr.raw_map[0].astype(np.int64) >> 7) & 7
    for t in range(1, len(tr.raw_map)):
        cur = (tr.raw_map[t].astype(np.int64) >> 7) & 7
        for r, x in zip(*np.nonzero((cur > 0) & (prev == 0))):
            y = H - 1 - r
            if lay.reg_team[lay.region[x, y]] == NEUTRAL:
                out.append((t, int(cur[r, x]), int(x), int(y)))
        prev = cur
    return out


def obs_events(rs: np.ndarray, rm: np.ndarray) -> list[tuple]:
    """Observation-level events between consecutive raw broadcasts: (step, kind, unit, detail)."""
    rs = np.asarray(rs)
    rm = np.asarray(rm).astype(np.int64)
    ev = []
    pos = positions(rs)
    lay = build_layout(chosen_from_map(rm[0]))
    for t in range(1, len(rs)):
        a, b = rs[t - 1], rs[t]
        for u in range(10):
            team = 0 if u < 5 else 1
            sx, sy = SPAWN_CELL[team]
            at_spawn = abs(pos[t, u, 0] - (sx + 0.5)) < 1e-6 and abs(pos[t, u, 1] - (sy + 0.5)) < 1e-6
            jumped = np.abs(pos[t, u] - pos[t - 1, u]).max() > 1.0
            died = at_spawn and jumped
            h0, h1 = float(a[4 * u + 2]), float(b[4 * u + 2])
            c0, c1 = int(a[4 * u + 3]), int(b[4 * u + 3])
            if died:
                ev.append((t, "kill", u, c0))
            if h0 != h1:
                if h0 == 0:
                    ev.append((t, "pickup", u, h1))
                elif h1 == 0:
                    ev.append((t, "lost" if (died or c1 == 1) else "deposit", u, h0))
                else:
                    ev.append((t, "swap", u, h1))
            if c0 != c1 and not died:
                ev.append((t, "transform", u, c1))
        if b[43] > a[43] + 1e-6:
            ev.append((t, "absorb", -1, round(float(b[40]) * 100) + 1000 * round(float(b[41]) * 100)))
        # steals: an item vanished from an enemy storage cell
        for r, x in zip(*np.nonzero((rm[t] >> 3) != (rm[t - 1] >> 3))):
            y = H - 1 - r
            reg = lay.region[x, y]
            if lay.reg_kind[reg] == 2 and (rm[t - 1][r, x] >> 3) and not (rm[t][r, x] >> 3) and b[43] <= a[43] + 1e-6:
                ev.append((t, "storage_out", int(lay.reg_team[reg]), int(x) * 100 + int(y)))
        sa, sb = round(float(a[40]) * 100), round(float(a[41]) * 100)
        ta, tb = round(float(b[40]) * 100), round(float(b[41]) * 100)
        if (sa, sb) != (ta, tb):
            ev.append((t, "score", -1, ta * 1000 + tb))
    return ev


def spawn_accumulations() -> int:
    """World updates the spawn timer needs to reach the cooldown (float32 sums of dt)."""
    t, n = F32(0.0), 0
    while t < core.SPAWN_COOLDOWN:
        t = F32(t + core.DT)
        n += 1
    return n


def timed_spawns(tr: Trace) -> list[tuple[int, int, int, int]]:
    """Every special spawn of a trace with the world-update tick it happened at.

    A spawn seen first at observation s happened in one of the two world updates of the step
    before it; the one consistent with how the previous special disappeared is chosen (the
    timer runs from 0 for ``spawn_accumulations()`` updates after that): from the match start,
    after an absorption (exact tick), or after a destruction during the moves of one of the
    two ticks of a step."""
    from .state import _timer_table

    _, ab, _, _ = _timer_table()
    n_acc = spawn_accumulations()
    spawns = detect_spawns(tr)
    exists = [special_exists(tr.raw_state[t], tr.raw_map[t]) for t in range(len(tr.raw_state))]
    out, prev_s = [], 0
    for (s, ty, x, y) in spawns:
        es = elapsed_ticks_of(tr.raw_state[s])
        cands = [es - 1, es]
        gone = None
        for g in range(s - 1, prev_s, -1):  # the special before this one vanished at obs g
            if exists[g - 1] and not exists[g]:
                gone = g
                break
        if gone is None and prev_s == 0:
            pred = {n_acc}
        elif gone is None:
            pred = set()
        else:
            eg, eg1 = elapsed_ticks_of(tr.raw_state[gone]), elapsed_ticks_of(tr.raw_state[gone - 1])
            wraps = [t for t in range(eg1 + 1, eg + 1) if ab[t] < ab[t - 1]]
            if wraps:
                pred = {wraps[0] + n_acc}
            else:
                pred = {eg1 + 1 + n_acc - 1, eg + n_acc - 1}
        both = [c for c in cands if c in pred]
        out.append((ty, x, y, both[0] if both else cands[0]))
        prev_s = s
    return out


def spawn_timer_at(tr: Trace, t0: int) -> float:
    """LevelDirector.currentSpawnTimer at observation t0 implied by the trace's next spawn."""
    if special_exists(tr.raw_state[t0], tr.raw_map[t0]):
        return 0.0
    e0 = elapsed_ticks_of(tr.raw_state[t0])
    nxt = [tick for (_, _, _, tick) in timed_spawns(tr) if tick > e0]
    n_acc = spawn_accumulations()
    done = n_acc - (nxt[0] - e0) if nxt else 0
    t = F32(0.0)
    for _ in range(max(0, done)):
        t = F32(t + core.DT)
    return float(t)


def init_sim(tr: Trace, t0: int, order=None, exact_seed: bool = True, spawns: str = "trace",
             history: int = 0) -> SimState:
    """Simulator state at observation t0 of a trace.  ``spawns="trace"`` forces the special
    spawns the trace shows; ``"rng"`` draws them from Unity's own RNG, which needs the exact seed
    initialisation (t0 = 0; not the first match after a player start, see SimState.from_seed)."""
    sim = None
    if t0 == 0 and exact_seed and "seed" in tr.meta:
        cand = SimState.from_seed(tr.meta["seed"], order=order, elapsed_ticks=elapsed_ticks_of(tr.raw_state[0]))
        if np.array_equal(cand.raw_state(), tr.raw_state[0]) and np.array_equal(cand.raw_map(), tr.raw_map[0]):
            sim = cand
    if sim is None:
        k0 = max(0, t0 - history)
        if k0 < t0:  # positions made exact from the preceding broadcasts (SimState.from_history)
            sim = SimState.from_history(tr.raw_state[k0:t0 + 1], tr.raw_map[k0:t0 + 1], tr.moves[k0:t0],
                                        order=order, spawn_timer=spawn_timer_at(tr, t0))
        else:
            sim = SimState.from_raw(tr.raw_state[t0], tr.raw_map[t0], order=order, spawn_timer=spawn_timer_at(tr, t0))
        e0 = elapsed_ticks_of(tr.raw_state[t0])
        sim.force_timed_spawns([sp for sp in timed_spawns(tr) if sp[3] > e0])
    elif spawns == "trace":
        sim.force_spawns([(ty, x, y) for (s, ty, x, y) in detect_spawns(tr, sim.layout) if s > t0])
    return sim


def replay(tr: Trace, t0: int = 0, steps: int | None = None, order=None, exact_seed: bool = True,
           spawns: str = "trace", history: int = 0):
    """Open-loop replay from observation t0 with the recorded moves of both teams.
    Returns (sim raw states [n+1, 44], sim raw maps [n+1, 24, 24]) aligned with trace rows
    t0 .. t0+n (n stops early if the simulated match ends)."""
    sim = init_sim(tr, t0, order, exact_seed, spawns, history)
    if spawns == "rng" and sim.g[core.G_SPAWN_MODE] != core.SPAWN_RNG:
        raise ValueError("spawns='rng' needs a seed-reproducible match start")
    rows = len(tr.raw_state)
    # the last recorded move ends the match (its terminal broadcast is not kept): apply it too
    end = rows if steps is None else min(rows, t0 + steps)
    states = [sim.raw_state()]
    maps = [sim.raw_map()]
    for t in range(t0, end):
        sim.step(tr.moves[t])
        states.append(sim.raw_state())
        maps.append(sim.raw_map())
        if sim.over:
            break
    keep = rows - t0
    return np.stack(states)[:keep], np.stack(maps)[:keep], sim
