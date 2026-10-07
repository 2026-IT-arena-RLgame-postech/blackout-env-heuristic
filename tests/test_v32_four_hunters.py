"""V32 (docs/policy_v2_design_ko.md §58): V31 + V18's exit guard / V19's sanctuary sweep, defaults = V31."""

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV28, StrategicHeuristicV31, StrategicHeuristicV32
from blackout_env.heuristics.mixture import make_heuristic

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play

V28_KW = dict(reserve_stage=0.5, trade_hold="advance")
V31_KW = dict(reserve_stage=0.5, trade_hold="advance", reserve_join="reserve", solo_sentry=True,
              threat_radius=0.13, camp_engage_radius=6.5)
ROLES = dict(hunter_quota=4, exit_guards=1, guard_min_hunters=3, sweep=True, cargo_transform=True, late_hunter_quota=3)


def test_v32_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v32"]()
    assert isinstance(p, StrategicHeuristicV32) and p.exit_guards == 0 and not p.sweep
    q = make_heuristic("strategic_v32", **V31_KW, **ROLES)
    assert q.hunter_quota == 4 and q.exit_guards == 1 and q.guard_min_hunters == 3 and q.sweep


def test_default_is_v31_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV31(**V31_KW).act(obs)
        y = StrategicHeuristicV32(**V31_KW).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v31_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV31(**V31_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV32(**V31_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(400):
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


def test_roles_fire_and_stay_observation_only():
    """Four Hunters by the race, the guard / sweep roles appear in a match, and a fresh instance plays the same
    moves on the same observation."""
    from blackout_env.sim.state import SimState, obs_from_raw

    sim = SimState.from_seed(4_000_003)
    a, b = StrategicHeuristicV32(**V31_KW, **ROLES), StrategicHeuristicV28(**V28_KW)
    seen_guard = seen_sweep = False
    for t in range(1500):
        if t % 50 == 0:
            obs = obs_from_raw(sim.raw_state(), sim.raw_map())
            mine = {f"unit_{i}": obs[f"unit_{i}"] for i in range(5)}
            x, y = StrategicHeuristicV32(**V31_KW, **ROLES).act(mine), StrategicHeuristicV32(**V31_KW, **ROLES).act(mine)
            assert all(np.array_equal(x[n], y[n]) for n in x)
        _play(sim, a, b)
        seen_guard |= bool(a.guards)
        seen_sweep |= bool(a.sweepers)
        assert not (set(a.guards) & set(a.sweepers)) and not (set(a.guards) & set(a.sentries))
    assert seen_guard and seen_sweep
