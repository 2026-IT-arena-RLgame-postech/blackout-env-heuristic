import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV23, StrategicHeuristicV24
from blackout_env.sim import SimState, core
from blackout_env.sim.mapdata import CARRIER, HUNTER

from .test_v22_teacher import _fixtures

OFF = dict(relief=False, reenter=False, base_exit=False, dash=False)
TEAM_A = [f"unit_{i}" for i in range(5)]


def _same(a, b, fixtures):
    for obs in fixtures:
        x, y = a.act(obs), b.act(obs)
        assert set(x) == set(y)
        for name in x:
            assert np.array_equal(x[name], y[name])


def _late(sim: SimState, seconds: float = 60.0) -> SimState:
    """Past the opening race (V22's ``_racing``: the first 10 s or no own Hunter)."""
    sim.gf[core.GF_EP] = np.float32(seconds)
    sim.gf[core.GF_ABS] = np.float32(seconds % 20.0)
    sim.g[core.G_TICK] = int(seconds * 50)
    return sim


def _pinned() -> SimState:
    """S1.5 root 950635_A: our Carrier with a battery in the pocket under our base, an enemy Hunter
    1.6 tiles away, our Hunters far off."""
    sim = _late(SimState.empty())
    sim.place_unit(0, 2.38, 15.5, CARRIER)
    sim.give_item(0, core.BATTERY, 8)
    sim.place_unit(1, 8.5, 15.5, HUNTER)
    sim.place_unit(2, 13.5, 10.5, HUNTER)
    sim.place_unit(3, 18.5, 5.3, HUNTER)
    sim.place_unit(9, 3.96, 15.5, HUNTER)
    return sim


def _inside_storage() -> SimState:
    """S1.5 root 950506_B (mirrored to team A): our Carrier picked a battery up while standing in
    our storage, so no region entry ever deposits it."""
    sim = _late(SimState.empty())
    sim.place_unit(0, 3.5, 21.5, CARRIER)  # Storage A1 (3, 21)-(3, 22); place_unit fires no entry
    sim.give_item(0, core.BATTERY, 2)
    return sim


def _team_a(sim):
    obs = sim.obs()
    return {n: obs[n] for n in TEAM_A}


def _sim_fixtures():
    yield _team_a(_pinned())
    yield _team_a(_inside_storage())


def test_v24_is_registered():
    assert isinstance(POLICY_REGISTRY["strategic_v24"](), StrategicHeuristicV24)


def test_all_rules_off_is_v23():
    _same(StrategicHeuristicV23(), StrategicHeuristicV24(**OFF), list(_fixtures()))
    _same(StrategicHeuristicV23(), StrategicHeuristicV24(**OFF), list(_sim_fixtures()))


def test_v24_is_a_function_of_the_observation():
    # one instance per map: the parents cache static map fields for the episode
    for fixtures in (list(_fixtures()), list(_sim_fixtures())):
        played = StrategicHeuristicV24()
        for obs in fixtures:
            x, y = played.act(obs), StrategicHeuristicV24().act(obs)
            for name in x:
                assert np.array_equal(x[name], y[name])
    # and along a played sequence: a long-lived instance equals a fresh one at every tick
    sim = _pinned()
    long_lived = StrategicHeuristicV24()
    for _ in range(40):
        obs = _team_a(sim)
        x, y = long_lived.act(obs), StrategicHeuristicV24().act(obs)
        for name in x:
            assert np.array_equal(x[name], y[name])
        moves = np.zeros((10, 2), np.float32)
        for name, a in x.items():
            moves[int(name.split("_")[1])] = a
        sim.step(moves)


def test_relief_sends_the_nearest_hunter_at_the_pinning_hunter():
    obs = _team_a(_pinned())
    v24 = StrategicHeuristicV24()
    v24.act(obs)
    assert v24.relievers == ["unit_1"]
    v23 = StrategicHeuristicV24(relief=False)
    v23.act(obs)
    assert v23.relievers == []


def _play(sim, policy, steps):
    for _ in range(steps):
        obs = _team_a(sim)
        moves = np.zeros((10, 2), np.float32)  # team B stands still
        for name, a in policy.act(obs).items():
            moves[int(name.split("_")[1])] = a
        sim.step(moves)
    return sim


def test_relief_trades_hunters_and_frees_the_cargo():
    sim = _play(_pinned(), StrategicHeuristicV24(reenter=False), 75)
    assert sim.ui[9, core.U_CLS] == core.COLLECTOR  # the pinning Hunter died (and respawned)
    assert sim.ui[1, core.U_CLS] == core.COLLECTOR  # so did ours: a unit back on the economy
    stay = _play(_pinned(), StrategicHeuristicV24(relief=False, reenter=False), 75)
    assert stay.ui[9, core.U_CLS] == HUNTER


def test_relief_needs_two_hunters():
    sim = _pinned()
    sim.place_unit(2, 1.5, 22.5, core.COLLECTOR)
    sim.place_unit(3, 1.5, 22.5, core.COLLECTOR)
    v24 = StrategicHeuristicV24()
    v24.act(_team_a(sim))
    assert v24.relievers == []


def test_reenter_steps_out_and_deposits():
    obs = _team_a(_inside_storage())
    assert np.array_equal(StrategicHeuristicV23().act(obs)["unit_0"], np.zeros(2, np.float32))  # V23 stands
    v24 = StrategicHeuristicV24()
    assert np.linalg.norm(v24.act(obs)["unit_0"]) > 0.5 and v24.reentering == ["unit_0"]
    sim = _play(_inside_storage(), StrategicHeuristicV24(), 25)
    assert sim.scores[0] == 2 and sim.ui[0, core.U_HELD] < 0  # deposited
    stuck = _play(_inside_storage(), StrategicHeuristicV23(), 25)
    assert stuck.ui[0, core.U_HELD] >= 0


def test_relief_waits_for_the_late_game():
    sim = _pinned()
    _late(sim, 20.0)
    v24 = StrategicHeuristicV24()
    v24.act(_team_a(sim))
    assert v24.relievers == []


def _camped() -> SimState:
    """Our Collector on the edge of our base (x 1-4, y 19-22), an enemy Hunter camping the exit."""
    sim = _late(SimState.empty())
    sim.place_unit(1, 4.3, 19.7)
    sim.place_unit(9, 5.4, 18.6, HUNTER)
    return sim


def test_base_exit_ignores_the_camper_inside_our_base():
    obs = _team_a(_camped())
    v23_move = StrategicHeuristicV24(base_exit=False).act(obs)["unit_1"]
    v24_move = StrategicHeuristicV24().act(obs)["unit_1"]
    away = np.array([4.3 - 5.4, 19.7 - 18.6])
    assert float(v23_move @ away) > 0  # V23 backs away from the camper
    assert not np.array_equal(v23_move, v24_move)
    early = _team_a(_late(_camped(), 20.0))  # before late_after it is V23
    assert np.array_equal(StrategicHeuristicV24().act(early)["unit_1"], StrategicHeuristicV24(base_exit=False).act(early)["unit_1"])
