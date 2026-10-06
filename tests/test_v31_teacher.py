"""V31 teacher (Q12, docs/policy_v2_design_ko.md §46) = V30 + ``threat_radius=0.13`` + ``camp_engage_radius=6.5``."""

import numpy as np

from blackout_env.heuristics import StrategicHeuristicV28, StrategicHeuristicV30
from blackout_env.heuristics.mixture import make_heuristic

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play

V28_KW = dict(reserve_stage=0.5, trade_hold="advance")
V30_KW = dict(reserve_stage=0.5, trade_hold="advance", reserve_join="reserve", solo_sentry=True)
V31_KW = dict(V30_KW, threat_radius=0.13, camp_engage_radius=6.5)


def test_v31_parameters_reach_the_policy():
    p = make_heuristic("strategic_v30", **V31_KW)
    assert p.threat_radius == 0.13 and p.camp_engage_radius == 6.5 and p.sentry_min_hunters == 1
    d = StrategicHeuristicV30(**V30_KW)
    assert d.threat_radius == 0.16 and d.camp_engage_radius == 4.5


def test_v31_differs_from_v30_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV30(**V30_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV30(**V31_KW), StrategicHeuristicV28(**V28_KW)]]
    for _ in range(3000):
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        if not np.array_equal(out[0][:5], out[1][:5]):
            return
    raise AssertionError("V31 never moved differently from V30")


def test_v31_is_a_function_of_the_observation():
    for obs in _fixtures():
        x, y = StrategicHeuristicV30(**V31_KW).act(obs), StrategicHeuristicV30(**V31_KW).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)
