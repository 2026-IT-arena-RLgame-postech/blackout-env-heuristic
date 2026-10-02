import math

import numpy as np
import pytest

from blackout_env.heuristics import (
    POLICY_REGISTRY, StrategicHeuristicV21AnyAngle, StrategicHeuristicV21Teach, StrategicHeuristicV22,
    StrategicHeuristicV22Teach,
)
from blackout_env.heuristics.v17_planner import BUFF_SIZE, BUFF_SPEED, DEBUFF_SIZE, DEBUFF_SPEED

from .test_v20_anticamp import _obs, _xy

COLLECTOR, HUNTER, CARRIER = 0, 1, 2
ALL_OFF = dict(race_cargo=False, item_free_walkers=False, size_aware=False, sentry=False, commit=False, break_ties=False, region_approach=False)


def _with(obs, fn):
    view = next(iter(obs.values()))
    fn(view["graphic"], view["agent_states"])
    return obs


def _hold(states, row, amount=4):
    states[row, 3] = 0
    states[row, 4] = amount / 15


def _fixtures():
    yield _obs({0: (2, 2), 1: (5, 8), 2: (8, 5)}, camper=None)
    yield _obs({0: (2, 2), 1: (5, 8), 2: (8, 5)}, camper=(11, 11))
    yield _obs({0: (2, 2)}, camper=(5, 5))
    yield _with(_obs({1: (9, 9), 2: (6, 6)}, camper=None), lambda g, s: _hold(s, 1))
    yield _with(_obs({1: (7, 7)}, camper=None), lambda g, s: g.__setitem__((1, 1, BUFF_SIZE), 1))


def test_v22_is_registered_and_plays_unit_moves():
    policy = POLICY_REGISTRY["strategic_v22"]()
    assert isinstance(policy, StrategicHeuristicV22)
    for obs in _fixtures():
        actions = policy.act(obs)
        assert set(actions) == {f"unit_{i}" for i in range(5)}
        for move in actions.values():
            norm = float(np.linalg.norm(move))
            assert np.isfinite(move).all() and (norm == 0.0 or norm == pytest.approx(1.0, abs=1e-5))


def test_all_switches_off_is_the_any_angle_v21():
    base, v22 = StrategicHeuristicV21AnyAngle(), StrategicHeuristicV22(**ALL_OFF)
    for obs in _fixtures():
        a, b = base.act(obs), v22.act(obs)
        for name in a:
            assert np.array_equal(a[name], b[name])


def test_existing_heuristics_are_unchanged_by_v22():
    # V22 retunes threat_radius / danger_tiles on its own instances only
    v21 = StrategicHeuristicV21AnyAngle()
    before = [v21.act(obs) for obs in _fixtures()]
    v22 = StrategicHeuristicV22()
    for obs in _fixtures():
        v22.act(_with(obs, lambda g, s: g.__setitem__((21, 21, BUFF_SIZE), 1)))
    v21b = StrategicHeuristicV21AnyAngle()
    assert v21b.threat_radius == pytest.approx(0.16) and v21b.danger_tiles == pytest.approx(3.0)
    after = [v21b.act(obs) for obs in _fixtures()]
    for x, y in zip(before, after):
        assert all(np.array_equal(x[n], y[n]) for n in x)
    assert POLICY_REGISTRY["strategic_v21_anyangle"] is StrategicHeuristicV21AnyAngle


def test_a_walker_holding_a_battery_keeps_walking_during_the_race():
    # unit 1 is the Collector nearest the sanctuary and holds a battery; no Hunter of ours exists
    obs = _with(_obs({1: (9, 9), 2: (6, 6)}, camper=None), lambda g, s: _hold(s, 1))
    v21 = StrategicHeuristicV21Teach().teach(obs)
    v22 = StrategicHeuristicV22Teach().teach(obs)
    assert v21["unit_1"].kind == "deposit"
    assert v22["unit_1"].kind == "transform"
    assert StrategicHeuristicV22Teach(race_cargo=False).teach(obs)["unit_1"].kind == "deposit"
    # a special item is not thrown away for the race
    special = _with(_obs({1: (9, 9), 2: (6, 6)}, camper=None), lambda g, s: (s.__setitem__((1, 3), 0), s.__setitem__((1, 6), 1)))
    assert StrategicHeuristicV22Teach().teach(special)["unit_1"].kind == "deposit"


def test_walkers_route_around_field_items_when_it_costs_nothing():
    # unit 1 at (7, 9) -> sanctuary (11, 11): 2 diagonal + 2 straight steps in any order, so south
    # and south-east both lie on a shortest path; a battery on (8, 10) costs nothing to avoid
    obs = _with(_obs({1: (7, 9)}, camper=None), lambda g, s: g.__setitem__((8, 10, 8), 5 / 15))
    v21 = StrategicHeuristicV21Teach().teach(obs)["unit_1"]
    v22 = StrategicHeuristicV22Teach().teach(obs)["unit_1"]
    assert v21.kind == v22.kind == "transform"
    s_, se = 6, 7  # compass indices of south (row +1) and south-east (row +1, col +1)
    assert v21.mask[s_] and v21.mask[se]
    assert v22.valid and v22.mask[s_] and not v22.mask[se]
    # a battery that costs a longer walk to avoid is walked over (and the walker keeps going)
    longer = _with(_obs({1: (7, 7)}, camper=None), lambda g, s: g.__setitem__((8, 8, 8), 5 / 15))
    assert StrategicHeuristicV22Teach().teach(longer)["unit_1"].mask[se]


def test_region_approach_heads_for_the_nearest_point_of_the_sanctuary():
    obs = _obs({1: (5, 8)}, camper=None)
    policy = StrategicHeuristicV22()
    move = policy.act(obs)["unit_1"]
    # sanctuary tiles rows/cols 11-12: the nearest region point from (5, 8) is (10.6, 10.6) (inset 0.1)
    expected = np.array([10.6 - 8.0, -(10.6 - 5.0)])
    assert np.allclose(move, expected / np.linalg.norm(expected), atol=1e-4)
    assert policy.anyangle_stats.get("region", 0) >= 1
    off = StrategicHeuristicV22(region_approach=False).act(obs)["unit_1"]
    assert not np.allclose(off, move, atol=1e-3)


def test_size_and_speed_items_are_counted():
    teacher = StrategicHeuristicV22Teach()
    plain = _obs({1: (7, 7)}, camper=None)
    teacher.teach(plain)
    assert teacher.sizes == (1.0, 1.0) and teacher.threat_radius == pytest.approx(0.16)

    def items(g, s):
        g[1, 1, BUFF_SIZE] = 1      # our storage: our units +50%
        g[20, 20, BUFF_SIZE] = 1    # theirs: their units +50%
        g[20, 21, BUFF_SIZE] = 1    # twice: +100%
        g[2, 2, DEBUFF_SIZE] = 1    # ours: theirs -30%
        g[1, 2, BUFF_SPEED] = 1
        g[2, 1, BUFF_SPEED] = 1
        g[21, 21, DEBUFF_SPEED] = 1
    teacher.teach(_with(_obs({1: (7, 7)}, camper=None), items))
    assert teacher.sizes == pytest.approx((1.5, 1.7))
    extra = 0.325 * 1.7 + 0.275 * 1.5 - 0.6
    assert teacher.threat_radius == pytest.approx(0.16 + extra / 12)
    assert teacher.danger_tiles == pytest.approx(3.0 + extra)
    ctx = teacher._teach_ctx
    assert teacher._speed(ctx, HUNTER) == pytest.approx(6.0 * 2.0)  # two speed buffs add
    assert teacher._speed(ctx, COLLECTOR) == pytest.approx(4.0 * max(0.1, 2.0 - 0.9))
    assert teacher._speed(ctx, COLLECTOR, enemy=True) == pytest.approx(4.0)
    assert teacher.contact_distance(ctx, HUNTER, True, COLLECTOR, False) == pytest.approx(0.325 * 1.7 + 0.275 * 1.5)
    # back to no items: the radii return to V21's
    teacher.teach(plain)
    assert teacher.threat_radius == pytest.approx(0.16) and teacher.danger_tiles == pytest.approx(3.0)


def test_a_walker_a_step_from_the_sanctuary_commits_past_a_hunter():
    # an enemy Hunter on (13, 13) covers every sanctuary tile (V21: no safe route, no candidate)
    obs = _obs({1: (10, 10)}, camper=(13, 13))
    v21 = StrategicHeuristicV21Teach().teach(obs)["unit_1"]
    teacher = StrategicHeuristicV22Teach()
    v22 = teacher.teach(obs)["unit_1"]
    assert v21.kind != "transform"
    assert v22.kind == "transform" and v22.valid and not v22.evading and teacher.committed == ["unit_1"]
    assert v22.mask[7]  # south-east, straight in
    # too far to make it before the Hunter could reach it: no commitment
    far = _obs({1: (6, 6)}, camper=(13, 13))
    assert StrategicHeuristicV22Teach().teach(far)["unit_1"].kind != "transform"
    assert StrategicHeuristicV22Teach(commit=False).teach(obs)["unit_1"].kind != "transform"


def _tied_batteries(g, s):
    g[..., 8] = 0
    g[10, 2, 8] = g[2, 10, 8] = 5 / 15  # mirror images about the diagonal through unit 1 and our storage
    for row in (0, 2, 3):  # the team already has its Hunters: unit 1 works the economy
        s[row, 9:12] = (0, 1, 0)
        s[row, :2] = _xy(16, 3 + row)


def test_tied_goals_pulling_opposite_ways_resolve_instead_of_freezing():
    obs = _with(_obs({1: (6, 6)}, camper=None), _tied_batteries)
    v21 = StrategicHeuristicV21Teach().teach(obs)["unit_1"]
    v22 = StrategicHeuristicV22Teach().teach(obs)["unit_1"]
    assert v21.kind == "battery" and not v21.valid and len(v21.goals) == 2  # V21 stands here for good
    assert v22.kind == "battery" and v22.valid and len(v22.goals) == 1
    move = StrategicHeuristicV22().act(obs)["unit_1"]
    assert float(np.linalg.norm(move)) == pytest.approx(1.0, abs=1e-5)
    assert not StrategicHeuristicV22Teach(break_ties=False).teach(obs)["unit_1"].valid
