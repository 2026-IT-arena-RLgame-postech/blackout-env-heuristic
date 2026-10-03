"""examples/unity_search_eval.py pieces that do not need a Unity player:

* the acting inputs built from an env observation (``_inputs``: ``canonical_obs``) are the batch
  renderer's inputs for the same state, both teams -- so the Unity player's r5 sees what the
  search's r5 sees on an in-sync shadow;
* the shadow resync (``SimState.from_history`` carrying RNG and spawn timer) reproduces the broadcast
  it was rebuilt from and the hidden state it was handed."""

from pathlib import Path

import numpy as np

from blackout_env.sim import SimState, core
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
from blackout_env.sim.trace import Trace, init_sim

FIXTURE = Path(__file__).resolve().parent / "data" / "sim_trace_short.npz"


def _walk(every=60, limit=500):
    tr = Trace.load(FIXTURE)
    sim = init_sim(tr, 0, None, True, "trace", 0)
    out = []
    for k in range(min(len(tr.moves), limit)):
        out.append((sim.copy(), tr.moves[k]))
        sim.step(tr.moves[k])
        if sim.over:
            break
    return out


def test_inputs_from_obs_equal_batch_render():
    from examples.unity_search_eval import _inputs

    states = [s for k, (s, _) in enumerate(_walk()) if k % 60 == 0]
    bs = BatchSim(states)
    n = len(states)
    g = np.zeros((n, 2, N_CH, 24, 24), np.float32)
    t = np.zeros((n, 2, 4), np.float32)
    a = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
    bs.render(bs.all, g, t, a)
    for b, s in enumerate(states):
        obs = s.obs()
        for side, name in ((0, "unit_0"), (1, "unit_7")):
            bufs = (np.zeros((1, 2, N_CH, 24, 24), np.float32), np.zeros((1, 2, 4), np.float32),
                    np.zeros((1, 2, 10, N_AGENT_COLS), np.float32))
            _inputs(obs[name], side, bufs)
            assert bufs[0][0, side].tobytes() == g[b, side].tobytes()
            assert bufs[1][0, side].tobytes() == t[b, side].tobytes()
            assert bufs[2][0, side].tobytes() == a[b, side].tobytes()


def test_resync_from_history_keeps_hidden_state():
    walk = _walk(limit=300)
    k = 200
    hist = walk[k - 25:k + 1]
    raw_s = np.stack([s.raw_state() for s, _ in hist])
    raw_m = np.stack([s.raw_map() for s, _ in hist])
    moves = np.stack([m for _, m in hist[:-1]])
    truth = hist[-1][0]
    kw = {"rng_state": truth.rng.copy(), "spawn_timer": float(truth.gf[core.GF_SPAWN])}
    sim = SimState.from_history(raw_s, raw_m, moves, **kw)
    assert np.array_equal(sim.raw_state(), truth.raw_state())
    assert np.array_equal(sim.raw_map(), truth.raw_map())
    assert np.array_equal(sim.rng, truth.rng)
    # 1 s on from the rebuilt state: same broadcasts as the true continuation
    for s_true, m in walk[k:k + 25]:
        assert np.array_equal(sim.raw_state(), s_true.raw_state())
        sim.step(m)
