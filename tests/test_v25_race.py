import math

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV23, StrategicHeuristicV25
from blackout_env.heuristics.race_path import RacePlanner, emulate_move, seg_box_distance, seg_clear_exact

from .test_v22_teacher import _fixtures


def _same(a, b):
    for obs in _fixtures():
        x, y = a.act(obs), b.act(obs)
        assert set(x) == set(y)
        for name in x:
            assert np.array_equal(x[name], y[name])


def test_v25_is_registered():
    assert isinstance(POLICY_REGISTRY["strategic_v25"](), StrategicHeuristicV25)


def test_with_both_rules_off_it_is_v23():
    _same(StrategicHeuristicV23(), StrategicHeuristicV25(race_walk=False, race_commit=False))


def test_v25_is_a_function_of_the_observation():
    played = StrategicHeuristicV25()
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV25().act(obs)
        for name in x:
            assert np.array_equal(x[name], y[name])


def test_segment_box_distance_matches_sampling():
    rng = np.random.default_rng(0)
    for _ in range(300):
        y0, x0, y1, x1 = rng.uniform(-2, 2, 4)
        cy, cx = rng.uniform(-1, 1, 2)
        h = 0.5
        ts = np.linspace(0, 1, 4001)
        py, px = y0 + (y1 - y0) * ts, x0 + (x1 - x0) * ts
        qy, qx = np.clip(py, cy - h, cy + h), np.clip(px, cx - h, cx + h)
        brute = np.hypot(py - qy, px - qx).min()
        assert abs(seg_box_distance(y0, x0, y1, x1, cy, cx, h) - brute) < 2e-3


def test_planner_open_floor_and_round_a_wall():
    walk = np.ones((9, 9), bool)
    walk[0, :] = walk[-1, :] = walk[:, 0] = walk[:, -1] = False
    pl = RacePlanner(walk, [(4, 7)], 0.275)
    # open floor: straight to the goal square's nearest point (inset 1e-3)
    assert abs(pl.distance((4.0, 2.0)) - (4.5 + 1e-3)) < 1e-6
    # a wall between: longer than straight, path keeps the radius
    walk[2:7, 4] = False
    pl = RacePlanner(walk, [(4, 7)], 0.275)
    d = pl.distance((4.0, 2.0))
    assert d > 4.5 + 0.5
    assert not seg_clear_exact(~walk, 4.0, 2.0, 4.0, 7.0, 0.275)  # straight through the wall
    h = pl.heading((4.0, 2.0))
    assert h is not None and abs(float(np.hypot(*h)) - 1.0) < 1e-5


def test_emulated_move_slides_along_a_wall():
    walk = np.ones((5, 5), bool)
    walk[0, :] = False  # wall row 0: square rows [-0.5, 0.5]
    blocked = ~walk
    # pressed against the wall (row 0.5 + 0.275), heading diagonally into it: only the column moves
    y, x = emulate_move(blocked, 0.775, 1.0, -math.sqrt(0.5), math.sqrt(0.5), 0.08, 2, 0.275)
    assert abs(y - 0.775) < 1e-9 and abs(x - (1.0 + 2 * 0.08 * math.sqrt(0.5))) < 1e-9


def test_exact_walk_reaches_the_sanctuary_at_decision_99():
    from blackout_env.sim import core
    from blackout_env.sim.mapdata import HUNTER
    from blackout_env.sim.state import SimState

    for team in (0, 1):
        sim = SimState.from_seed(4_000_000)
        mask = sim.layout.walk[team].T[::-1].copy()  # [row = 23 - y, col = x]
        pl = RacePlanner(mask, [(23 - y, x) for x in (11, 12) for y in (11, 12)], 0.275)
        u = 5 * team
        assert abs(pl.distance((23.5 - float(sim.uf[u, 1]), float(sim.uf[u, 0]) - 0.5)) - 15.6766) < 1e-3
        first = None
        for t in range(110):
            mv = np.zeros((10, 2), np.float32)
            p = sim.uf[u]
            mv[u] = pl.heading((23.5 - float(p[1]), float(p[0]) - 0.5))
            sim.step(mv)
            if sim.ui[u, core.U_CLS] == HUNTER:
                first = t + 1
                break
        assert first == 99
