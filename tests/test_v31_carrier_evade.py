import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV28, StrategicHeuristicV30, StrategicHeuristicV31

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play

V28_KW = dict(reserve_stage=0.5, trade_hold="advance")
V30_KW = dict(reserve_stage=0.5, trade_hold="advance", reserve_join="reserve", solo_sentry=True)


def _row(x, y, enemy, cls):
    """A unit state row: position, team sign at column 2, class one-hot in the last three columns."""
    r = np.zeros(12, np.float32)
    r[:2] = (x, y)
    r[2] = -1.0 if enemy else 1.0
    r[9 + cls] = 1.0
    return r


def test_v31_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v31"]()
    assert isinstance(p, StrategicHeuristicV31) and p.carrier_col_scale == 1.0 and p.carrier_hunter_scale == 1.0


def test_default_is_v30_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV30(**V30_KW).act(obs)
        y = StrategicHeuristicV31(**V30_KW).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v30_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV30(**V30_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV31(**V30_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(400):
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


def test_carrier_threats_filter():
    from blackout_env.heuristics.strategic import CARRIER, COLLECTOR, HUNTER

    p = StrategicHeuristicV31(carrier_col_scale=0.5, carrier_hunter_scale=0.6, **V30_KW)
    r = p.threat_radius
    me = _row(0.0, 0.0, False, CARRIER)
    states = np.stack([me,
                       _row(0.4 * r, 0.0, True, COLLECTOR),   # near Collector: kept
                       _row(0.7 * r, 0.0, True, COLLECTOR),   # far Collector: dropped (0.5)
                       _row(0.0, 0.5 * r, True, HUNTER),      # near Hunter: kept
                       _row(0.0, 0.8 * r, True, HUNTER),      # far Hunter: dropped (0.6)
                       _row(0.9 * r, 0.0, False, COLLECTOR)])  # teammate: kept
    kept = p._carrier_threats(me, states)
    assert len(kept) == 4 and np.array_equal(kept[[1, 2]], states[[1, 3]])
    # other classes and the defaults see every enemy
    col = _row(0.0, 0.0, False, COLLECTOR)
    assert p._carrier_threats(col, states) is states
    assert StrategicHeuristicV31(**V30_KW)._carrier_threats(me, states) is states


def test_col_scale_zero_changes_carrier_moves_only():
    """Some Carrier move differs from V30 in a match; until the first difference both play the same."""
    from blackout_env.sim.mapdata import CARRIER
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV30(**V30_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV31(carrier_col_scale=0.0, **V30_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(3000):
        cls = sims[0].ui[:5, 0].copy()
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        diff = np.nonzero(np.any(out[0][:5] != out[1][:5], axis=1))[0]
        if len(diff):
            assert all(cls[u] == CARRIER for u in diff), (t, diff, cls)
            return
    raise AssertionError("carrier_col_scale=0 never changed a move")


def test_v31_is_a_function_of_the_observation():
    played = StrategicHeuristicV31(carrier_col_scale=0.0, carrier_hunter_scale=0.6, **V30_KW)
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV31(carrier_col_scale=0.0, carrier_hunter_scale=0.6, **V30_KW).act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)
