"""Unit rules of the BlackOut simulator (blackout_env/sim) against the Unity C# semantics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blackout_env.sim import SimState, core
from blackout_env.sim.mapdata import (CARRIER, COLLECTOR, HUNTER, SPAWN_CELL, chosen_from_map,
                                      storage_cells)
from blackout_env.sim.unity_random import generate

ROOT = Path(__file__).resolve().parents[1]
F32 = np.float32
BATTERY, BUFF_SPEED, DEBUFF_SPEED, BUFF_SIZE, DEBUFF_SIZE = 0, 1, 2, 3, 4


def _unity_physics():
    spec = importlib.util.spec_from_file_location("anyangle_probe", ROOT / "examples" / "anyangle_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UnityPhysics


def _still(sim, steps=1):
    for _ in range(steps):
        sim.step(np.zeros((10, 2), np.float32))


def _park_all(sim):
    """Everyone far apart in their own bases, out of each other's way."""
    for u in range(10):
        sx, sy = SPAWN_CELL[0 if u < 5 else 1]
        sim.place_unit(u, sx + 0.5, sy + 0.5, COLLECTOR)


# --------------------------------------------------------------------------- movement

def test_movement_matches_unity_physics_port():
    """Random walks of a lone Collector agree with examples/anyangle_probe.UnityPhysics (the
    double-precision port of UnitMovementSystem.ClampPosition) to float32 accuracy."""
    UnityPhysics = _unity_physics()
    rng = np.random.default_rng(0)
    sim0 = SimState.empty()
    walk_a = sim0.layout.walk[0]                       # [x, y]
    walkable = np.zeros((24, 24), dtype=bool)          # [row, col], row = 23 - y
    for x in range(24):
        for y in range(24):
            walkable[23 - y, x] = walk_a[x, y]
    radius, speed = 0.275, 4.0
    worst = 0.0
    for trial in range(20):
        sim = sim0.copy()
        _park_all(sim)
        while True:
            start = (2.5 + rng.integers(0, 20), 2.5 + rng.integers(0, 20))
            if walk_a[int(start[0]), int(start[1])]:
                break
        sim.place_unit(0, *start)
        phys = UnityPhysics(walkable, radius)
        px, py = start[0] - 0.5, start[1] - 23.5  # probe frame: cell centres at integers, y = -row
        heading = rng.normal(size=2)
        for t in range(150):
            if rng.random() < 0.1:
                heading = rng.normal(size=2)
            mv = np.zeros((10, 2), np.float32)
            mv[0] = (heading / np.linalg.norm(heading)).astype(np.float32)
            sim.step(mv)
            for _ in range(2):
                px, py = phys.move(px, py, float(mv[0, 0]), float(mv[0, 1]), speed * float(core.DT))
            sx, sy = sim.uf[0]
            worst = max(worst, abs(sx - 0.5 - px), abs(sy - 23.5 - py))
    assert worst < 1e-3


def test_move_is_normalised_and_uses_dt():
    sim = SimState.empty()
    _park_all(sim)
    sim.place_unit(0, 6.5, 10.5)
    mv = np.zeros((10, 2), np.float32)
    mv[0] = (0.3, 0.4)  # |v| = 0.5: still moves at full speed
    sim.step(mv)
    assert np.hypot(*(sim.uf[0] - (6.5, 10.5))) == pytest.approx(2 * 4.0 * float(core.DT), rel=1e-5)


def test_walls_use_unscaled_radius_and_enemy_base_blocks():
    sim = SimState.empty()
    _park_all(sim)
    # a team B Collector walking west into team A's base wall at x = 5 (base is x 1-4, y 19-22)
    sim.place_unit(5, 6.0, 20.5)
    mv = np.zeros((10, 2), np.float32)
    mv[5] = (-1, 0)
    _still_mv = mv.copy()
    for _ in range(40):
        sim.step(_still_mv)
    assert sim.uf[5, 0] == pytest.approx(5.0 + 0.275, abs=1e-5)
    # team A walks in
    sim.place_unit(0, 6.0, 20.5)
    mv = np.zeros((10, 2), np.float32)
    mv[0] = (-1, 0)
    for _ in range(10):
        sim.step(mv)
    assert sim.uf[0, 0] < 5.0


# --------------------------------------------------------------------------- combat

@pytest.mark.parametrize("ca,cb,dead_a,dead_b", [
    (HUNTER, COLLECTOR, False, True),
    (HUNTER, CARRIER, False, True),
    (HUNTER, HUNTER, True, True),
    (COLLECTOR, CARRIER, False, True),
    (CARRIER, CARRIER, True, True),
    (COLLECTOR, COLLECTOR, False, False),
    (CARRIER, COLLECTOR, True, False),
])
def test_combat_matrix(ca, cb, dead_a, dead_b):
    sim = SimState.empty()
    _park_all(sim)
    sim.place_unit(0, 12.0, 6.5, ca)
    sim.place_unit(5, 12.3, 6.5, cb)
    _still(sim)
    spawn_a = np.array(SPAWN_CELL[0], np.float32) + 0.5
    spawn_b = np.array(SPAWN_CELL[1], np.float32) + 0.5
    assert np.array_equal(sim.uf[0], spawn_a) == dead_a
    assert np.array_equal(sim.uf[5], spawn_b) == dead_b
    if dead_a:
        assert sim.ui[0, core.U_CLS] == COLLECTOR
    kills = [e for e in sim.events() if e[1] == core.EV_KILL]
    assert len(kills) == int(dead_a) + int(dead_b)


def test_death_destroys_held_item_and_unlocks_carrier_sanctuary():
    sim = SimState.empty()
    _park_all(sim)
    # unit 1 becomes the Carrier through its locked sanctuary at (1, 20)
    sim.place_unit(1, 1.5, 21.5)
    mv = np.zeros((10, 2), np.float32)
    mv[1] = (0, -1)
    for _ in range(8):
        sim.step(mv)
    assert sim.ui[1, core.U_CLS] == CARRIER
    lock = [r for r in range(sim.layout.n_regions) if sim.layout.names[r] == "Carrier Sanctuary A"][0]
    assert sim.reg_lock[lock] == 1
    # unit 2 cannot follow while the lock is held
    sim.place_unit(2, 1.5, 21.5)
    mv[:] = 0
    mv[2] = (0, -1)
    for _ in range(8):
        sim.step(mv)
    assert sim.ui[2, core.U_CLS] == COLLECTOR
    # the Carrier dies holding a battery: item gone, lock released
    sim.give_item(1, BATTERY, 5)
    sim.place_unit(1, 12.0, 6.5)
    sim.place_unit(6, 12.2, 6.5, HUNTER)
    _still(sim)
    assert sim.ui[1, core.U_HELD] == -1 and sim.ui[1, core.U_CLS] == COLLECTOR
    assert sim.reg_lock[lock] == -1
    assert not np.any((sim.it[:, core.I_STATE] != core.POOL) & (sim.it[:, core.I_TYPE] == BATTERY))


def test_hunter_transformation_drops_item():
    sim = SimState.empty()
    _park_all(sim)
    sim.place_unit(0, 10.5, 11.5)
    sim.give_item(0, BATTERY, 3)
    mv = np.zeros((10, 2), np.float32)
    mv[0] = (1, 0)
    for _ in range(6):
        sim.step(mv)
    assert sim.ui[0, core.U_CLS] == HUNTER and sim.ui[0, core.U_HELD] == -1


# --------------------------------------------------------------------------- items and storage

def test_pickup_rules():
    sim = SimState.empty()
    _park_all(sim)
    sim.add_item(BATTERY, 4, 12, 7)
    sim.place_unit(6, 12.5, 7.5, HUNTER)  # Hunters cannot collect
    _still(sim)
    assert sim.ui[6, core.U_HELD] == -1
    sim.place_unit(6, 22.5, 1.5)
    sim.place_unit(0, 12.5, 7.5)  # a Collector does
    _still(sim)
    assert sim.ui[0, core.U_HELD] >= 0 and sim.it[sim.ui[0, core.U_HELD], core.I_AMT] == 4
    # only one item at a time
    sim.add_item(BATTERY, 2, 13, 7)
    sim.place_unit(0, 13.5, 7.5)
    _still(sim)
    assert sim.it[sim.ui[0, core.U_HELD], core.I_AMT] == 4


def test_deposit_fill_order_and_atomicity():
    sim = SimState.empty((0, 1, 2, 3))
    _park_all(sim)
    lay = sim.layout
    reg = lay.names.index("Storage_A2")
    tiles = [tuple(t) for t in lay.reg_tiles[reg, : lay.reg_ntiles[reg]]]
    assert tiles == [(10, 21), (9, 21), (10, 22), (9, 22)]  # farthest from spawn (1, 22) first, then x, y
    sim.add_item(BATTERY, 7, 10, 21)
    sim.add_item(BATTERY, 10, 9, 21)
    sim.add_item(BUFF_SIZE, 1, 10, 22)
    assert sim.scores == (17, 0)
    # carrying 8: tops (10, 21) up to 10, skips the full and the mismatched tile, rest on (9, 22)
    sim.give_item(0, BATTERY, 8)
    sim.place_unit(0, 9.5, 20.95)
    mv = np.zeros((10, 2), np.float32)
    mv[0] = (0, 1)
    _still(sim, 0)
    sim.step(mv)
    assert sim.ui[0, core.U_HELD] == -1
    amounts = {(int(sim.it[i, core.I_TX]), int(sim.it[i, core.I_TY])): int(sim.it[i, core.I_AMT])
               for i in range(core.MAXI) if sim.it[i, core.I_STATE] == core.GROUND and sim.it[i, core.I_TYPE] == 0}
    assert amounts == {(10, 21): 10, (9, 21): 10, (9, 22): 5}
    assert sim.scores == (25, 0)
    # a full storage refuses atomically: the unit keeps its item
    sim2 = SimState.empty((0, 1, 2, 3))
    _park_all(sim2)
    for (x, y) in tiles:
        sim2.add_item(BATTERY, 9, x, y)
    sim2.give_item(0, BATTERY, 5)  # 4 free units only
    sim2.place_unit(0, 9.5, 20.95)
    sim2.step(mv)
    assert sim2.ui[0, core.U_HELD] >= 0 and sim2.scores == (36, 0)


def test_deposit_only_on_region_entry():
    sim = SimState.empty((0, 1, 2, 3))
    _park_all(sim)
    sim.place_unit(0, 9.9, 21.5)  # already inside storage A2 when it gets the item
    sim.give_item(0, BATTERY, 3)
    mv = np.zeros((10, 2), np.float32)
    mv[0] = (1, 0)
    sim.step(mv)  # moves to the next storage tile: same region, no deposit
    assert sim.ui[0, core.U_HELD] >= 0


def test_steal_reduces_enemy_score_and_own_storage_is_off_limits():
    sim = SimState.empty((0, 1, 2, 3))
    _park_all(sim)
    sim.add_item(BATTERY, 6, 21, 10)  # team B storage B2
    assert sim.scores == (0, 6)
    sim.place_unit(5, 21.5, 10.5)  # B cannot lift from its own storage
    _still(sim)
    assert sim.ui[5, core.U_HELD] == -1
    sim.place_unit(5, 22.5, 1.5)
    sim.place_unit(0, 21.5, 10.5)
    _still(sim)
    assert sim.ui[0, core.U_HELD] >= 0 and sim.scores == (0, 0)
    assert any(e[1] == core.EV_STEAL for e in sim.events())


def test_absorption_locks_score_and_clears_storages():
    sim = SimState.empty((0, 1, 2, 3))
    _park_all(sim)
    sim.add_item(BATTERY, 6, 9, 21)
    sim.add_item(BUFF_SPEED, 1, 10, 21)
    sim.gf[core.GF_ABS] = F32(19.95)
    _still(sim, 2)
    assert sim.scores == (6, 0)
    assert not np.any(sim.it[:, core.I_STATE] == core.GROUND)
    assert sim.mn[0] == 0  # the speed buff ended with the item


def test_buffs_and_debuffs():
    def displacement(setup, u, cls=COLLECTOR):
        sim = SimState.empty((0, 1, 2, 3))
        _park_all(sim)
        setup(sim)
        sim.place_unit(u, 12.5, 7.5, cls)
        mv = np.zeros((10, 2), np.float32)
        mv[u] = (1, 0)
        sim.step(mv)
        return float(sim.uf[u, 0] - 12.5)

    base = displacement(lambda s: None, 0)
    assert displacement(lambda s: s.add_item(BUFF_SPEED, 1, 9, 21), 0) == pytest.approx(1.5 * base, rel=1e-5)
    # DebuffSpeed in A's storage slows B's Collectors only
    assert displacement(lambda s: s.add_item(DEBUFF_SPEED, 1, 9, 21), 5) == pytest.approx(0.1 * base, rel=1e-4)
    hunter = displacement(lambda s: None, 5, HUNTER)
    assert displacement(lambda s: s.add_item(DEBUFF_SPEED, 1, 9, 21), 5, HUNTER) == pytest.approx(hunter, rel=1e-6)
    # size: scales the interaction radius, not the wall radius
    sim = SimState.empty((0, 1, 2, 3))
    sim.add_item(BUFF_SIZE, 1, 9, 21)
    sim.add_item(DEBUFF_SIZE, 1, 21, 9)  # B's storage: -30% on A
    assert core._half_width(sim.S, 0) == pytest.approx(0.55 * 1.2 / 2, rel=1e-6)
    assert core._half_width(sim.S, 5) == pytest.approx(0.55 / 2, rel=1e-6)


def test_special_spawn_rules():
    sim = SimState.empty()
    sim.reseed(3)
    _park_all(sim)
    for _ in range(int(10.0 / (2 * float(core.DT))) + 2):
        _still(sim)
    specials = [i for i in range(core.MAXI) if sim.it[i, core.I_STATE] == core.GROUND and sim.it[i, core.I_TYPE] > 0]
    assert len(specials) == 1
    i = specials[0]
    x, y = sim.it[i, core.I_TX], sim.it[i, core.I_TY]
    lay = sim.layout
    assert lay.reg_team[lay.region[x, y]] == 2 and lay.tile[x, y] == 0  # neutral floor
    # no second spawn while it exists
    for _ in range(300):
        _still(sim)
    assert sum(1 for j in range(core.MAXI) if sim.it[j, core.I_STATE] == core.GROUND and sim.it[j, core.I_TYPE] > 0) == 1


def test_seeded_generation_matches_trace():
    d = np.load(ROOT / "tests" / "data" / "sim_trace_short.npz")
    import json

    meta = json.loads(str(d["meta"]))
    chosen, batteries, _ = generate(meta["seed"])
    assert tuple(sorted(chosen)) == chosen_from_map(d["raw_map"][0])
    m0 = d["raw_map"][0].astype(np.int64)
    seen = {(x, 23 - r): int((m0[r, x] >> 3) & 15) for r in range(24) for x in range(24) if (m0[r, x] >> 3) & 15}
    assert seen == {(x, y): a for x, y, a in batteries}


def test_storage_layout_mirrors_team_a():
    a, b = storage_cells((0, 1, 2, 3))
    assert {(y, x) for (x, y) in a} == b
