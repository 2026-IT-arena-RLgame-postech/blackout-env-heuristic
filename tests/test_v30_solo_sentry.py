import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV28, StrategicHeuristicV29, StrategicHeuristicV30

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play

V28_KW = dict(reserve_stage=0.5, trade_hold="advance")
V29_KW = dict(reserve_stage=0.5, trade_hold="advance", reserve_join="reserve")


def _own_hunters(sim):
    from blackout_env.sim.mapdata import HUNTER

    return int((sim.ui[:5, 0] == HUNTER).sum())


def test_v30_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v30"]()
    assert isinstance(p, StrategicHeuristicV30) and p.solo_sentry is False and p.sentry_min_hunters == 2
    assert StrategicHeuristicV30(solo_sentry=True).sentry_min_hunters == 1


def test_default_is_v29_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV29(**V29_KW).act(obs)
        y = StrategicHeuristicV30(**V29_KW).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v29_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV29(**V29_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV30(**V29_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(400):  # race, trades, second race
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


def test_solo_sentry_is_v29_with_one_hunter_minimum():
    """The V30 descriptor ``solo_sentry=true`` plays exactly V29 with ``sentry_min_hunters=1`` (the gated setting)."""
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_005), SimState.from_seed(4_000_005)]
    players = [[StrategicHeuristicV29(sentry_min_hunters=1, **V29_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV30(solo_sentry=True, **V29_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(400):
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


def test_a_lone_hunter_is_posted_as_sentry():
    """Against V28 the opening trades leave V29 one Hunter: V29 posts no sentry, V30 posts it."""
    from blackout_env.sim.state import SimState

    seen = {}
    for name, ours in (("v29", StrategicHeuristicV29(**V29_KW)), ("v30", StrategicHeuristicV30(solo_sentry=True, **V29_KW))):
        sim = SimState.from_seed(4_000_003)
        opp = StrategicHeuristicV28(**V28_KW)
        lone, prev = [], 0
        for _ in range(200):
            _play(sim, ours, opp)
            now = _own_hunters(sim)
            if now == 1 and prev == 1:  # ``sentries`` is set when the Hunter goals are replanned
                lone.append(bool(ours.sentries))
            prev = now
        seen[name] = lone
    assert seen["v29"] and seen["v30"]
    assert not any(seen["v29"][10:]) and any(seen["v30"])


def test_v30_is_a_function_of_the_observation():
    played = StrategicHeuristicV30(solo_sentry=True, **V29_KW)
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV30(solo_sentry=True, **V29_KW).act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)
