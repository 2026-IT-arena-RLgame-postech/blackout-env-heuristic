"""B1 pipelined search rollouts (blackout_env/sim/pipeline.py): bit-identical with ``BatchRollout`` --
the device decode, every branch kind, opponent variants, the external base overlay, 2-ply, several
forward chunks, and whole ``play_matches`` runs -- and the fallbacks."""

import math

import numpy as np
import pytest
import torch

from blackout_env.sim.batch import BatchSim, decode_moves
from blackout_env.sim.pipeline import PipelinedRollout, _Dev, moves_from_bins
from blackout_env.sim.search import MatchConfig, ModNetPolicy, Searcher, ValueNetLeaf, parse_branches, play_matches

from tests.test_sim_search import _net, _states, _value

BRANCHES = "alt*,alt1:1,pair03,rot2:45,samp:1.0x2,samp:0.5x1@4,stop3"


def _devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("width", [32, 33, 8])
@pytest.mark.parametrize("angle", [True, False])
def test_device_decode_matches_decode_moves(width, angle):
    rng = np.random.default_rng(width)
    m = 64
    q = rng.normal(size=(m, 5, width)).astype(np.float32)
    q[:4, :, :3] = 7.0                      # ties: the first max wins
    q[4:8, :, -1] = 9.0                      # STAND (odd width) / the last bin
    k = width - width % 2
    off = (rng.normal(size=(m, 5, k)) * 0.1).astype(np.float32) if angle else None
    team_b = rng.random(m) < 0.5
    ref = decode_moves(q, off, team_b)
    for dev in _devices():
        d = _Dev(dev)
        j, o = d.decode(d.put(q), None if off is None else d.put(off), d.put(team_b))
        got = moves_from_bins(j.cpu().numpy(), None if o is None else o.cpu().numpy(), width)
        assert got.dtype == ref.dtype and np.array_equal(got, ref), dev


def _on(net, device, max_batch):
    net.max_batch = max_batch
    if device != "cpu":
        net.own.to(device)
        net.device = torch.device(device)
    return net


def _searchers(device="cpu", max_batch=6, chunk=None, base_seed=None, pipeline_graphs=False, **kw):
    """(reference, pipelined) searchers over equal nets; ``max_batch`` views per forward chunk (the rollout
    net), 5 (the base net)."""
    out, branches = [], parse_branches(kw.pop("branches", BRANCHES))
    for pipe in (False, chunk or True):
        base = None if base_seed is None else _on(_net(base_seed), device, 5)
        out.append(Searcher(ModNetPolicy(_on(_net(21), device, max_batch)), ValueNetLeaf(_value(), device),
                            branches,
                            window=kw.get("window", 5), leaf_decisions=kw.get("leaf", 6), margin=0.0,
                            base_policy=base, overlay=kw.get("overlay", True), pipeline=pipe,
                            pipeline_graphs=pipeline_graphs and pipe is not False,
                            **{k: v for k, v in kw.items() if k not in ("window", "leaf", "overlay")}))
    return out


def _same(a, b):
    assert np.array_equal(a.values, b.values) and np.array_equal(a.choice, b.choice)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("opts", [
    {},                                                               # 1-ply, every branch kind
    {"opp_branches": parse_branches("samp:1.0x3"), "opp_agg": "mean", "opp_left": 11},  # o4f
    {"base_seed": 5},                                                 # external base, overlay
    {"base_seed": 5, "overlay": False},
    {"base_seed": 5, "opp_branches": parse_branches("samp:1.0x3"), "opp_agg": "mean", "opp_left": 11},
    {"children": parse_branches("alt0,samp:1.0x1"), "child_window": 3, "expand": 2},  # 2-ply (bsim= runs)
])
def test_pipelined_search_is_bit_identical(device, opts):
    roots = _states(3)
    teams, seeds = [0, 1, 1], [11, 12, 13]
    ref, pipe = _searchers(device, **opts)
    assert isinstance(pipe.roll, PipelinedRollout) and pipe.roll.chunk_games == 3
    a = ref.search(roots, teams, seeds)
    b = pipe.search(roots, teams, seeds)
    _same(a, b)
    assert a.timing["game_decisions"] == b.timing["game_decisions"]
    assert pipe.roll.timing.get("wait") is not None  # really pipelined, not the fallback
    # again (buffers reused, counters reconfigured)
    _same(ref.search(roots[:2], teams[:2], seeds[:2]), pipe.search(roots[:2], teams[:2], seeds[:2]))


def test_pipelined_rollout_final_states_identical():
    ref, pipe = _searchers(base_seed=7, opp_branches=parse_branches("samp:1.0x3"), opp_agg="mean", opp_left=11)
    roots = _states(2)
    for s in (ref, pipe):
        s._configure(len(roots) * 2, np.array([0, 0, 1, 1]), parse_branches("alt2,samp:1.0x1") * 2, 5)
    r1 = ref.roll.run([r for r in roots for _ in range(2)], 12, reseeds=[1, 2, 3, 4])
    r2 = pipe.roll.run([r for r in roots for _ in range(2)], 12, reseeds=[1, 2, 3, 4])
    from blackout_env.sim.batch import _SIM_ARRAYS

    for name in _SIM_ARRAYS:
        assert np.array_equal(getattr(r1.bsim, name), getattr(r2.bsim, name)), name
    for k in ("scores", "over", "steps", "ticks", "first_hunter"):
        assert np.array_equal(getattr(r1, k), getattr(r2, k)), k


def test_play_matches_identical():
    cfg = MatchConfig(every=10, window=10, cap_s=2, search_from=5)
    res = []
    for pipe in (False, True):
        net = _net(31)
        net.max_batch = 8
        s = Searcher(ModNetPolicy(net), ValueNetLeaf(_value()), parse_branches("alt*,samp:1.0x2"), window=10,
                     leaf_decisions=5, margin=0.5, base_policy=_net(32), opp_branches=parse_branches("samp:1.0x3"),
                     opp_agg="mean", opp_left=15, pipeline=pipe)
        res.append(play_matches([41, 42, 43], [0, 1, 0], ModNetPolicy(net), _net(33), s, cfg, base=_net(32)))
    for k in ("margin", "ticks", "chosen", "gain_sum", "searches", "first_ours", "first_opp"):
        assert np.array_equal(res[0][k], res[1][k]), k


def test_fallbacks_and_finer_chunks():
    roots = _states(2)
    # an extra opponent net in use -> BatchRollout (packed): still identical
    kw = dict(window=4, leaf_decisions=4, opp_branches=parse_branches("samp:1.0x1"), opp_agg="mean", opp_net_idx=[0, 1])
    a = Searcher(ModNetPolicy(_net(1), opp_nets=[_net(2)]), ValueNetLeaf(_value()), parse_branches("alt0"), **kw)
    b = Searcher(ModNetPolicy(_net(1), opp_nets=[_net(2)]), ValueNetLeaf(_value()), parse_branches("alt0"), pipeline=True, **kw)
    _same(a.search(roots, [0, 1], [3, 4]), b.search(roots, [0, 1], [3, 4]))
    assert "wait" not in b.roll.timing
    # finer chunks than the forward's: runs, same shapes (bit-exactness not promised on GPUs)
    ref, fine = _searchers(max_batch=64, chunk=2)
    x, y = ref.search(roots, [0, 1], [3, 4]), fine.search(roots, [0, 1], [3, 4])
    assert x.values.shape == y.values.shape and np.allclose(x.values, y.values, atol=1e-3)


def test_moves_from_bins_stand():
    j = np.array([[32, 0, 8, 16, 24]])
    mv = moves_from_bins(j, np.zeros((1, 5), np.float32), 33)
    assert np.all(mv[0, 0] == 0) and np.allclose(mv[0, 2], [math.cos(math.pi / 2), 1.0], atol=1e-7)


@pytest.mark.parametrize("device", _devices())
def test_games_ending_mid_rollout(device):
    """Roots whose episode clock runs out inside the rollout at different decisions: chunk membership shifts
    as games drop out (the roll-over stops, the next step re-chunks) -- still bit-identical."""
    from blackout_env.sim import core

    roots = _states(4)
    for k, left_s in enumerate((0.13, 10.0, 0.31, 0.05)):
        roots[k].gf[core.GF_EP] = core.EPISODE_SECONDS - left_s
    for opts in ({}, {"base_seed": 5, "window": 3}):
        ref, pipe = _searchers(device, max_batch=4, **opts)
        teams, seeds = [0, 1, 1, 0], [1, 2, 3, 4]
        a, b = ref.search(roots, teams, seeds), pipe.search(roots, teams, seeds)
        _same(a, b)
        assert a.timing["game_decisions"] == b.timing["game_decisions"] < 4 * 8 * 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("opts", [{}, {"base_seed": 5, "opp_branches": parse_branches("samp:1.0x3"), "opp_agg": "mean",
                                       "opp_left": 11}])
def test_cuda_graph_chunks_bit_identical(opts):
    roots = _states(3)
    teams, seeds = [0, 1, 1], [11, 12, 13]
    ref, pipe = _searchers("cuda", pipeline_graphs=True, **opts)
    assert pipe.roll.graphs
    for _ in range(2):  # capture, then replay
        _same(ref.search(roots, teams, seeds), pipe.search(roots, teams, seeds))
    assert pipe.roll._graphs
