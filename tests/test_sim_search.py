"""S3 search policy (blackout_env/sim/search.py): branch modifications, leaf encoding, and that a
searcher which can only pick the base branch plays exactly the plain rollout net."""

import math

import numpy as np
import torch

from blackout_env.sim import SimState
from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchRollout, BatchSim, NetPolicy
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.sim.search import (K_ALT, K_BASE, K_ROT, K_SAMP, K_STOP, Branch, MatchConfig, ModNetPolicy,
                                     Searcher, ValueNetLeaf, parse_branches, play_matches, race)


def _net(seed=0):
    torch.manual_seed(seed)
    return NetPolicy(RolloutNet(arch="attn", d=32, layers=1, heads=2, hidden=32, c1=8, c2=8, c3=8, glob=16).eval())


def _states(n=4, steps=30):
    """Mid-opening states: a few random decisions from different seeds."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        s = SimState.from_seed(300 + k)
        for _ in range(steps):
            s.step(rng.normal(size=(10, 2)).astype(np.float32).clip(-1, 1))
        out.append(s)
    return out


def _inputs(bs):
    g = np.zeros((bs.n, 2, N_CH, 24, 24), np.float32)
    t = np.zeros((bs.n, 2, 4), np.float32)
    a = np.zeros((bs.n, 2, 10, N_AGENT_COLS), np.float32)
    bs.render(bs.all, g, t, a)
    return g, t, a


def test_parse_branches():
    br = parse_branches("alt*,rot2:45,samp:0.5x2,stop1")
    assert br[0].kind == K_BASE
    names = [b.name for b in br]
    assert names == ["base", "alt0", "alt1", "alt2", "alt3", "alt4", "rot2+45", "rot2-45", "samp0.5s0", "samp0.5s1", "stop1"]


def test_modifications_apply_to_one_team_and_expire():
    net = _net()
    bs = BatchSim(_states())
    inp = _inputs(bs)
    ref = net.moves(bs, bs.all, (0, 1), inp)
    pol = ModNetPolicy(net)
    pol.configure(bs.n)
    got = pol.moves(bs, bs.all, (0, 1), inp)  # nothing configured = the plain net
    for tm in (0, 1):
        assert np.array_equal(got[tm], ref[tm])
    # game 0: team A alt unit 1; game 1: team B stop unit 3; game 2: team A rot unit 0 +90; game 3: team B sample
    pol.configure(bs.n, team=[0, 1, 0, 1], kind=[K_ALT, K_STOP, K_ROT, K_SAMP], unit=[1, 3, 0, 0],
                  param=[0, 0, math.pi / 2, 1.0], seed=[0, 0, 0, 5], left=[2, 2, 2, 2])
    got = pol.moves(bs, bs.all, (0, 1), inp)
    assert np.array_equal(got[1][0], ref[1][0]) and np.array_equal(got[0][1], ref[0][1])  # other team untouched
    assert np.array_equal(np.delete(got[0][0], 1, 0), np.delete(ref[0][0], 1, 0))
    cos = float(np.dot(got[0][0, 1], ref[0][0, 1]))
    assert cos < math.cos(1.5 * 2 * math.pi / 32)  # outside the top bin +- 1 (11.25 deg bins)
    assert np.all(got[1][1, 3] == 0) and np.array_equal(np.delete(got[1][1], 3, 0), np.delete(ref[1][1], 3, 0))
    v, w = ref[0][2, 0], got[0][2, 0]
    assert np.allclose(w, [-v[1], v[0]], atol=1e-6)
    assert np.isclose(np.linalg.norm(got[1][3], axis=-1), 1, atol=1e-5).all()
    # same seed and step -> same samples
    pol2 = ModNetPolicy(net)
    pol2.configure(bs.n, team=[1] * 4, kind=[K_SAMP] * 4, param=[1.0] * 4, seed=[5] * 4, left=[1] * 4)
    assert np.array_equal(pol2.moves(bs, bs.all, (0, 1), inp)[1][3], got[1][3])
    pol.moves(bs, bs.all, (0, 1), inp)  # second decision: left 2 -> 0
    after = pol.moves(bs, bs.all, (0, 1), inp)
    for tm in (0, 1):
        assert np.array_equal(after[tm], ref[tm])


def test_leaf_inputs_match_value_evaluator():
    from blackout_env.value.evaluate import ValueEvaluator
    from blackout_env.value.model import ValueNet

    torch.manual_seed(0)
    vnet = ValueNet(hidden_size=64).eval()
    leaf = ValueNetLeaf(vnet)
    states = _states(3)
    bs = BatchSim(states)
    teams = np.array([0, 1, 0])
    got = leaf(bs, teams)
    ev = ValueEvaluator(vnet)
    ref = ev.evaluate_obs([s.obs() for s in states], teams, symmetric=True)["final"] * 100
    assert np.allclose(got, ref, atol=1e-3)


def test_base_only_search_plays_the_plain_net():
    net = _net(1)
    leaf = ValueNetLeaf(_value())
    seeds, sides = [11, 12, 11, 12], [0, 0, 1, 1]
    cfg = MatchConfig(every=10, window=10, cap_s=3)
    plain = play_matches(seeds, sides, (p := ModNetPolicy(net)), p, None, cfg)
    s = Searcher(ModNetPolicy(net), leaf, [Branch(K_BASE)], window=10, leaf_decisions=10)
    srch = play_matches(seeds, sides, (q := ModNetPolicy(net)), q, s, cfg)
    assert np.array_equal(plain["margin"], srch["margin"]) and np.array_equal(plain["ticks"], srch["ticks"])
    assert (srch["searches"] == 8).all() and (srch["chosen"][:, 0] == 8).all()


def test_search_picks_best_leaf_and_terminal_is_exact():
    net = _net(2)
    s = Searcher(ModNetPolicy(net), ValueNetLeaf(_value()), parse_branches("alt*"), window=5, leaf_decisions=5)
    roots = _states(2)
    out = s.search(roots, [0, 1], [1, 2])
    assert out.values.shape == (2, 6)
    assert np.array_equal(out.choice, np.where(out.values.max(1) > out.values[:, 0], out.values.argmax(1), 0))


def test_race():
    assert race(np.array([10, 10, -1, 5, -1]), np.array([20, 10, 3, -1, -1])).tolist() == [1, 0, -1, 1, 0]


def _value():
    from blackout_env.value.model import ValueNet

    torch.manual_seed(3)
    return ValueNet(hidden_size=64).eval()
