import numpy as np
import pytest

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV25, StrategicHeuristicV26

from .test_v20_anticamp import _obs
from .test_v22_teacher import _fixtures

COLLECTOR, HUNTER = 0, 1
OFF = dict(late_trade=False, trade_walk_reach=0.0)


def _hunter(obs, row):
    states = next(iter(obs.values()))["agent_states"]
    states[row, 9:12] = 0
    states[row, 9 + HUNTER] = 1
    return obs


def test_v26_is_registered():
    assert isinstance(POLICY_REGISTRY["strategic_v26"](), StrategicHeuristicV26)
    with pytest.raises(ValueError):
        StrategicHeuristicV26(trade_mode="sometimes")


def test_with_the_rules_off_it_is_v25_on_fixtures():
    for obs in _fixtures():
        x, y = StrategicHeuristicV25().act(obs), StrategicHeuristicV26(**OFF).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_with_the_rules_off_it_is_v25_in_a_sim_match():
    from blackout_env.sim.state import obs_from_raw
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV25(), StrategicHeuristicV25()], [StrategicHeuristicV26(**OFF), StrategicHeuristicV25()]]
    for t in range(400):  # past the race and the first Hunter contacts
        out = []
        for sim, (a, b) in zip(sims, players):
            obs = obs_from_raw(sim.raw_state(), sim.raw_map())
            mv = np.zeros((10, 2), np.float32)
            for team, pol in ((0, a), (1, b)):
                acts = pol.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * team, 5 * team + 5)})
                for name, act in acts.items():
                    mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
            out.append(mv)
            sim.step(mv)
        assert np.array_equal(out[0], out[1]), t


def test_our_hunter_goes_for_the_enemy_hunter():
    # our unit 1 is a Hunter, the enemy's unit 5 a Hunter outside its base: one each (parity)
    obs = _hunter(_obs({1: (12, 6), 2: (16, 3)}, camper=(6, 14)), 1)
    v25 = StrategicHeuristicV25().teach(obs)["unit_1"]
    played = StrategicHeuristicV26()
    v26 = played.teach(obs)["unit_1"]
    assert played.traders == ["unit_1"]
    assert v26.kind == "hunt" and v26.goals[0][0] == ((6, 14),)
    assert not (v25.kind == "hunt" and v25.goals[0][0] == ((6, 14),))
    # "parity": two Hunters of ours against one of theirs -> no trade
    obs2 = _hunter(_hunter(_obs({1: (12, 6), 2: (16, 3)}, camper=(6, 14)), 1), 2)
    two = StrategicHeuristicV26()
    two.teach(obs2)
    assert two.traders == []
    always = StrategicHeuristicV26(trade_mode="always")
    always.teach(obs2)
    assert len(always.traders) == 1
    # opening only: trade_until in the past
    late = StrategicHeuristicV26(trade_until=-1.0)
    late.teach(obs)
    assert late.traders == []


def test_v26_is_a_function_of_the_observation():
    played = StrategicHeuristicV26(trade_walk_reach=4.0)
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV26(trade_walk_reach=4.0).act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)


def test_deficit_and_camper_modes():
    far = _hunter(_obs({1: (12, 6), 2: (16, 3)}, camper=(6, 14)), 1)   # one each, theirs far from our spawn
    near = _hunter(_obs({1: (12, 6), 2: (16, 3)}, camper=(4, 5)), 1)   # theirs camping our exit
    for obs, mode, traded in ((far, "deficit", False), (far, "camper", False), (near, "camper", True)):
        played = StrategicHeuristicV26(trade_mode=mode)
        played.teach(obs)
        assert (played.traders == ["unit_1"]) is traded, (mode, played.traders)
