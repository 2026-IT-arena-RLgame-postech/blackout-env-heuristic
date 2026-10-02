"""Replay of a committed Unity trace through the simulator, determinism, observation format.

tests/data/sim_trace_short.npz is reports/sim/traces/trace_007.npz (strategic_v21 vs the P5a
student, 595 decisions, team B wins 42-100 on a deposit): recorded moves of both teams, Unity's raw
broadcast after every decision."""

from pathlib import Path

import numpy as np

from blackout_env.sim import SimState, obs_from_raw, raw_from_view
from blackout_env.sim.trace import Trace, obs_events, replay

FIXTURE = Path(__file__).resolve().parent / "data" / "sim_trace_short.npz"


def _assert_identical(tr, s, m):
    n = len(s)
    assert n == tr.steps + 1 or n == len(tr.raw_state)
    for k in range(n):
        assert np.array_equal(s[k], tr.raw_state[k]), f"raw state differs at step {k}"
        assert np.array_equal(m[k], tr.raw_map[k]), f"map differs at step {k}"


def test_replay_from_seed_is_bit_exact():
    tr = Trace.load(FIXTURE)
    s, m, sim = replay(tr, exact_seed=True)
    _assert_identical(tr, s, m)
    # the deciding deposit (96 -> 100) is the last recorded move; its broadcast was the reset one
    assert sim.over and sim.scores == (42, 100) and sim.winner == 1


def test_replay_from_first_observation_is_bit_exact():
    tr = Trace.load(FIXTURE)
    s, m, _ = replay(tr, exact_seed=False)
    _assert_identical(tr, s, m)


def test_mid_match_start_tracks_unity():
    tr = Trace.load(FIXTURE)
    t0 = 300
    s, m, _ = replay(tr, t0=t0, steps=125, exact_seed=False)
    u = tr.raw_state[t0:t0 + len(s)]
    pos = lambda r: (r[:, :40].reshape(-1, 10, 4)[:, :, :2].astype(np.float64) + 1) * 12
    assert np.abs(pos(s) - pos(u)).max() < 1e-3
    assert obs_events(s, m) == obs_events(u, tr.raw_map[t0:t0 + len(s)])


def test_determinism_and_copy_independence():
    tr = Trace.load(FIXTURE)
    a = SimState.from_raw(tr.raw_state[100], tr.raw_map[100])
    a.reseed(5)
    b = a.copy()
    fork = a.copy()
    for t in range(100, 300):
        a.step(tr.moves[t])
        b.step(tr.moves[t])
    assert np.array_equal(a.raw_state(), b.raw_state()) and np.array_equal(a.raw_map(), b.raw_map())
    assert np.array_equal(a.it, b.it) and np.array_equal(a.rng, b.rng)
    # the fork did not move
    assert np.array_equal(fork.raw_state(), SimState.from_raw(tr.raw_state[100], tr.raw_map[100]).raw_state())
    fork.load(a)
    assert np.array_equal(fork.raw_state(), a.raw_state())


def test_observation_format_matches_env():
    tr = Trace.load(FIXTURE)
    sim = SimState.from_raw(tr.raw_state[50], tr.raw_map[50])
    obs = sim.obs()
    ref = obs_from_raw(tr.raw_state[50], tr.raw_map[50])
    assert set(obs) == {f"unit_{i}" for i in range(10)}
    for name in obs:
        for key in ("graphic", "team_state", "agent_states"):
            assert obs[name][key].dtype == np.float32
            assert np.array_equal(obs[name][key], ref[name][key])
    assert obs["unit_0"]["graphic"].shape == (24, 24, 13)
    assert obs["unit_0"]["agent_states"].shape == (10, 12)
    # a team B view inverts back to the same raw broadcast
    raw, packed = raw_from_view(obs["unit_7"])
    assert np.array_equal(raw, tr.raw_state[50]) and np.array_equal(packed, tr.raw_map[50])
