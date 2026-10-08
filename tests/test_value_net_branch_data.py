"""GPU/batched S1.6 branch data for a rollout net's continuations (examples/value_net_branch_data.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, NetPolicy  # noqa: E402
from blackout_env.sim.rollout_net import RolloutNet  # noqa: E402
from blackout_env.sim.search import parse_branches  # noqa: E402
from blackout_env.value.branch import load_raw  # noqa: E402
from blackout_env.value.evaluate import encode_view, team_view  # noqa: E402

from tests.test_sim_batch import _trace_states  # noqa: E402

import value_net_branch_data as vd  # noqa: E402


def test_encode_rendered_is_the_value_training_encoding():
    states = _trace_states(40)
    bs = BatchSim(states)
    n = len(states)
    g = np.zeros((n, 2, N_CH, 24, 24), np.float32)
    t = np.zeros((n, 2, 4), np.float32)
    a = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
    bs.render(bs.all, g, t, a)
    for team in (0, 1):
        enc = vd.encode_rendered(g[:, team])
        for b, s in enumerate(states):
            ge, te, ae = encode_view(team_view(s.obs(), team))
            assert np.array_equal(enc[b], ge)
            assert np.array_equal(t[b, team], te) and np.array_equal(a[b, team], ae)


def test_play_chunk_rows_are_consistent(tmp_path):
    torch.manual_seed(0)
    net = NetPolicy(RolloutNet(arch="attn", d=32, layers=1, heads=2), device="cpu")
    args = SimpleNamespace(cap_s=12.0, window=10, seed0=1, verbose=False, net="random", branches="alt0,samp:1.0x1")
    branches = parse_branches(args.branches)
    seeds, sides = [11, 11, 12], [0, 1, 0]
    res = vd.play_chunk(seeds, sides, "self", net, None, branches, args)
    vd.save_matches(tmp_path, seeds, sides, "self", res, args)
    d = load_raw(tmp_path / "raw")
    r, st = d["rows"], d["states"]
    assert set(np.unique(r["kind"])) == {0, 1}
    assert set(np.unique(r["branch"][r["kind"] == 1])) == {0, 1, 2}
    lead_own = (st["own"]["team_state"][:, 0] - st["own"]["team_state"][:, 1]) * 100
    lead_mir = (st["mir"]["team_state"][:, 0] - st["mir"]["team_state"][:, 1]) * 100
    np.testing.assert_allclose(lead_own[r["idx"]], r["lead"], atol=1e-3)  # rows' leads from the learner side
    np.testing.assert_allclose(lead_mir, -lead_own, atol=1e-3)            # mir = the other team's view
    live = r["next_idx"] >= 0
    np.testing.assert_allclose(lead_own[r["next_idx"][live]], r["lead5"][live], atol=1e-3)
    assert (live | r["ended"]).all()
    # the base branch of a root = the plain net's own trajectory for the window: in self-play the
    # root match's state at root + window is that same state (traj rows every second)
    t_root = r["root"][r["kind"] == 1]
    assert (t_root >= 0).all() and (r["root"][r["kind"] == 0] == -1).all()


def test_packed_graphic_roundtrip_and_load_raw(tmp_path):
    from blackout_env.value.branch import pack_graphic_u8, unpack_graphic_u8

    states = _trace_states(40)
    enc = np.stack([encode_view(team_view(s.obs(), t))[0] for s in states for t in (0, 1)])
    assert np.array_equal(unpack_graphic_u8(pack_graphic_u8(enc)), enc)
    torch.manual_seed(0)
    net = NetPolicy(RolloutNet(arch="attn", d=32, layers=1, heads=2), device="cpu")
    args = SimpleNamespace(cap_s=6.0, window=10, seed0=1, verbose=False, net="random", branches="alt0")
    res = vd.play_chunk([11], [1], "self", net, None, parse_branches(args.branches), args)
    vd.save_matches(tmp_path, [11], [1], "self", res, args)
    a, b = load_raw(tmp_path / "raw"), load_raw(tmp_path / "raw", packed=True)
    for v in ("own", "mir"):
        assert np.array_equal(unpack_graphic_u8(b["states"][v]["graphic"]), a["states"][v]["graphic"])
