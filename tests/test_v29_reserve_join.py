import numpy as np
import pytest

from blackout_env.heuristics import (POLICY_REGISTRY, StrategicHeuristicV23, StrategicHeuristicV28,
                                     StrategicHeuristicV29)

from .test_v22_teacher import _fixtures
from .test_v27_reserve import _play

V28_KW = dict(reserve_stage=0.5, trade_hold="advance")


def _hunters_at(sim, a, b, decision):
    from blackout_env.sim.mapdata import HUNTER

    for _ in range(decision):
        _play(sim, a, b)
    return int((sim.ui[:5, 0] == HUNTER).sum()), int((sim.ui[5:, 0] == HUNTER).sum())


def test_v29_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v29"]()
    assert isinstance(p, StrategicHeuristicV29) and p.reserve_join == "off"
    with pytest.raises(ValueError):
        StrategicHeuristicV29(reserve_join="sometimes")


def test_default_is_v28_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV28(**V28_KW).act(obs)
        y = StrategicHeuristicV29(**V28_KW).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v28_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV28(**V28_KW), StrategicHeuristicV28(**V28_KW)],
               [StrategicHeuristicV29(**V28_KW), StrategicHeuristicV28(**V28_KW)]]
    for t in range(300):  # race, staging, trades, second race
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


@pytest.mark.parametrize("mode", ["always", "tie"])
def test_reserve_joins_a_tied_race(mode):
    """Against V28 (tied race at decision 99) the reserve walks in with the walkers: four Hunters against three,
    and after the three trades ours is the only Hunter left."""
    from blackout_env.sim.state import SimState

    sim = SimState.from_seed(4_000_003)
    a, b = StrategicHeuristicV29(reserve_join=mode, **V28_KW), StrategicHeuristicV28(**V28_KW)
    assert _hunters_at(sim, a, b, 99) == (4, 3)
    for _ in range(3):
        _play(sim, a, b)
    from blackout_env.sim.mapdata import HUNTER

    assert (sim.ui[:5, 0] == HUNTER).sum() >= 1 and (sim.ui[5:, 0] == HUNTER).sum() == 0


def test_tie_mode_holds_when_the_race_is_won():
    """V23's walkers are ~0.15 tiles behind (first Hunter at 100): the race is won, the reserve holds as V28's."""
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV28(**V28_KW), StrategicHeuristicV23()],
               [StrategicHeuristicV29(reserve_join="tie", **V28_KW), StrategicHeuristicV23()]]
    for t in range(110):
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t
        assert t < 80 or not players[1][0].joining  # early on V23's walkers bend differently; the gap opens by ~3 s
    sim = SimState.from_seed(4_000_003)
    assert _hunters_at(sim, StrategicHeuristicV29(reserve_join="always", **V28_KW), StrategicHeuristicV23(), 99)[0] == 4


def test_v29_is_a_function_of_the_observation():
    played = StrategicHeuristicV29(reserve_join="tie", **V28_KW)
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV29(reserve_join="tie", **V28_KW).act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)
