import numpy as np
import pytest

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV27, StrategicHeuristicV28

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play


def _dist(sim, u):
    x, y = sim.uf[u]
    return float(np.hypot(max(11 - x, x - 13, 0), max(11 - y, y - 13, 0)))


def test_v28_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v28"]()
    assert isinstance(p, StrategicHeuristicV28) and p.trade_hold == "off"
    with pytest.raises(ValueError):
        StrategicHeuristicV28(trade_hold="sometimes")


def test_default_is_v27_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV27(reserve_stage=1.5).act(obs)
        y = StrategicHeuristicV28(reserve_stage=1.5).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v27_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV27(reserve_stage=0.5), StrategicHeuristicV27(reserve_stage=0.5)],
               [StrategicHeuristicV28(reserve_stage=0.5), StrategicHeuristicV27(reserve_stage=0.5)]]
    for t in range(300):  # race, staging, trades, second race
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


@pytest.mark.parametrize("mode", ["hold", "advance"])
def test_reserve_does_not_back_off_from_the_pending_trade(mode):
    """Mirror of V27(0.5): at decision 99 both sides' walkers are Hunters on the sanctuary; V27's reserve steps
    back (evades), V28's holds ("hold") or steps in ("advance") and turns into a Hunter no later than the enemy's."""
    from blackout_env.sim.mapdata import HUNTER
    from blackout_env.sim.state import SimState

    sim = SimState.from_seed(4_000_003)
    a, b = StrategicHeuristicV28(reserve_stage=0.5, trade_hold=mode), StrategicHeuristicV27(reserve_stage=0.5)
    d_before = held = None
    first = {0: None, 1: None}
    for t in range(140):
        cls_before = sim.ui[:, 0].copy()
        if (cls_before[:5] == HUNTER).any() and d_before is None:
            res = [u for u in range(5) if cls_before[u] != HUNTER and _dist(sim, u) < 1.0]
            assert res, "no staged reserve at the trade"
            held, d_before = res[0], _dist(sim, res[0])
        _play(sim, a, b)
        if held is not None and d_before is not None and t < 101 and a.trade_held:
            assert _dist(sim, held) <= d_before + 1e-6
        for team in (0, 1):
            if t > 100 and first[team] is None and (sim.ui[5 * team:5 * team + 5, 0] == HUNTER).any():
                first[team] = t
    assert first[0] is not None and (first[1] is None or first[0] <= first[1])


def test_v28_is_a_function_of_the_observation():
    played = StrategicHeuristicV28(reserve_stage=0.5, trade_hold="advance")
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV28(reserve_stage=0.5, trade_hold="advance").act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)
