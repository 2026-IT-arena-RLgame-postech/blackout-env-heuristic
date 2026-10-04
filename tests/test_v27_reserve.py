import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV25, StrategicHeuristicV27

from .test_v22_teacher import _fixtures


def _play(sim, a, b):
    from blackout_env.sim.state import obs_from_raw

    obs = obs_from_raw(sim.raw_state(), sim.raw_map())
    mv = np.zeros((10, 2), np.float32)
    for team, pol in ((0, a), (1, b)):
        acts = pol.act({f"unit_{i}": obs[f"unit_{i}"] for i in range(5 * team, 5 * team + 5)})
        for name, act in acts.items():
            mv[int(name.split("_")[1])] = np.clip(np.asarray(act, np.float32), -1, 1)
    sim.step(mv)
    return mv


def test_v27_is_registered_and_off_by_default():
    p = POLICY_REGISTRY["strategic_v27"]()
    assert isinstance(p, StrategicHeuristicV27) and p.reserve_stage == 0.0


def test_default_is_v25_on_fixtures():
    for obs in _fixtures():
        x, y = StrategicHeuristicV25().act(obs), StrategicHeuristicV27().act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)


def test_default_is_v25_in_a_sim_match():
    from blackout_env.sim.state import SimState

    sims = [SimState.from_seed(4_000_003), SimState.from_seed(4_000_003)]
    players = [[StrategicHeuristicV25(), StrategicHeuristicV25()], [StrategicHeuristicV27(), StrategicHeuristicV25()]]
    for t in range(400):  # past the race, the trades and the second race
        out = [_play(sim, a, b) for sim, (a, b) in zip(sims, players)]
        assert np.array_equal(out[0], out[1]), t


def test_reserve_walks_behind_the_walkers_and_holds():
    from blackout_env.sim.mapdata import HUNTER
    from blackout_env.sim.state import SimState

    sim = SimState.from_seed(4_000_003)
    a, b = StrategicHeuristicV27(reserve_stage=2.0), StrategicHeuristicV25()
    held = []
    for t in range(95):
        mv = _play(sim, a, b)
        assert len(a.staged) <= 1
        if a.staged:
            u = int(a.staged[0].split("_")[1])
            if not mv[u].any():
                held.append(t)
    assert held, "the reserve never stood"
    # it stood within reach of the sanctuary while the walkers were still walking (no Hunter yet)
    assert not (sim.ui[:, 0] == HUNTER).any()
    x, y = sim.uf[u]
    d = np.hypot(max(11 - x, x - 13, 0), max(11 - y, y - 13, 0))
    assert 1.0 < d <= 2.5


def test_v27_is_a_function_of_the_observation():
    played = StrategicHeuristicV27(reserve_stage=2.0)
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV27(reserve_stage=2.0).act(obs)
        assert all(np.array_equal(x[n], y[n]) for n in x)
