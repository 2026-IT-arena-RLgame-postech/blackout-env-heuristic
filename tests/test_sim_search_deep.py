"""S4 search options (blackout_env/sim/search.py, docs §26): alt ranks, two-unit pairs, the
opponent slot, opponent variants, 2-ply / best-first trees and ``take_games``."""

import numpy as np
import torch

from blackout_env.sim.batch import BatchRollout, BatchSim, NetPolicy
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.sim.search import (K_ALT, K_ALTM, K_SAMP, Branch, ModNetPolicy, Searcher, ValueNetLeaf,
                                     parse_branches, take_games)
from tests.test_sim_search import _inputs, _states, _value


def _net(seed=0, per_sample=True):
    torch.manual_seed(seed)
    net = RolloutNet(arch="attn", d=32, layers=1, heads=2, hidden=32, c1=8, c2=8, c3=8, glob=16).eval()
    return NetPolicy(net, per_sample=per_sample)


def _bins(net, bs, inp):
    """Arg-max bin per (game, team, unit) of a plain forward, in the canonical frame."""
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH

    g, t, a = inp
    q, _ = net.forward(g.reshape(-1, N_CH, 24, 24), t.reshape(-1, 4), a.reshape(-1, 10, N_AGENT_COLS))
    return q.reshape(bs.n, 2, 5, -1)


def test_parse_s4_branches():
    names = [b.name for b in parse_branches("alt*:1,pair*,pair03,samp:1.0x2@4")]
    assert names[:6] == ["base", "alt0r1", "alt1r1", "alt2r1", "alt3r1", "alt4r1"]
    assert names[6:16] == ["pair01", "pair02", "pair03", "pair04", "pair12", "pair13", "pair14", "pair23", "pair24", "pair34"]
    assert names[16:] == ["pair03", "samp1s4", "samp1s5"]
    assert [b.name for b in parse_branches("alt*")] == ["base", "alt0", "alt1", "alt2", "alt3", "alt4"]


def test_alt_rank_and_pair():
    net = _net()
    bs = BatchSim(_states())
    inp = _inputs(bs)
    ref = net.moves(bs, bs.all, (0, 1), inp)
    q = _bins(net, bs, inp)
    k = q.shape[-1]
    pol = ModNetPolicy(net)
    # games: alt unit 2 rank 0, alt unit 2 rank 1, pair (1, 3), alt unit 1 then unit 3 checked against pair
    pol.configure(bs.n, team=[0, 0, 0, 0], kind=[K_ALT, K_ALT, K_ALTM, K_ALT], unit=[2, 2, 0b1010, 1],
                  param=[0, 1, 0, 0], left=[1] * 4)
    got = pol.moves(bs, bs.all, (0, 1), inp)
    for b in range(4):
        assert np.array_equal(got[1][b], ref[1][b])  # team B untouched

    top = q[0, 0, 2].argmax()
    masked = q[0, 0, 2].copy()
    masked[[(top - 1) % k, top, (top + 1) % k]] = -np.inf
    second = masked.argmax()
    masked2 = masked.copy()
    masked2[[(second - 1) % k, second, (second + 1) % k]] = -np.inf
    third = masked2.argmax()
    assert third not in ((top - 1) % k, top, (top + 1) % k, (second - 1) % k, second, (second + 1) % k)
    assert not np.allclose(got[0][0, 2], got[0][1, 2])  # rank 0 and rank 1 differ
    # pair (1, 3): units 1 and 3 changed, the rest = plain
    assert np.array_equal(got[0][2][[0, 2, 4]], ref[0][2][[0, 2, 4]])
    assert not np.allclose(got[0][2][1], ref[0][2][1]) and not np.allclose(got[0][2][3], ref[0][2][3])
    # each pair unit = that unit's single alt
    for u in (1, 3):
        pol.configure(bs.n, team=[0] * 4, kind=[K_ALT] * 4, unit=[u] * 4, left=[1] * 4)
        assert np.array_equal(got[0][2][u], pol.moves(bs, bs.all, (0, 1), inp)[0][2][u])


def test_opponent_slot():
    net = _net()
    bs = BatchSim(_states())
    inp = _inputs(bs)
    ref = net.moves(bs, bs.all, (0, 1), inp)
    pol = ModNetPolicy(net)
    pol.configure(bs.n, team=[0, 1, 0, -1], left=[0] * 4)
    pol.configure_opp(kind=[K_SAMP] * 4, param=[1.0] * 4, seed=[3] * 4, left=[1] * 4)
    got = pol.moves(bs, bs.all, (0, 1), inp)
    assert np.array_equal(got[0][0], ref[0][0]) and not np.array_equal(got[1][0], ref[1][0])  # opp of team 0 = B
    assert np.array_equal(got[1][1], ref[1][1]) and not np.array_equal(got[0][1], ref[0][1])
    assert np.array_equal(got[0][3], ref[0][3]) and np.array_equal(got[1][3], ref[1][3])      # no team: untouched
    after = pol.moves(bs, bs.all, (0, 1), inp)  # expired
    for tm in (0, 1):
        assert np.array_equal(after[tm], ref[tm])


def test_take_games_continues_exactly():
    net = _net()
    pol = ModNetPolicy(net)
    roll = BatchRollout(pol, pol)
    roots = _states(2)
    pol.configure(2)
    one = roll.run(roots, 12, reseeds=[5, 6])
    pol.configure(2)
    half = roll.run(roots, 5, reseeds=[5, 6])
    cont = take_games(half.bsim, [1, 0, 1])
    pol.configure(3)
    rest = roll.run(None, 7, bsim=cont)
    for name in ("uf", "ui", "g", "gf", "rng"):
        assert np.array_equal(getattr(rest.bsim, name)[[1, 0]], getattr(one.bsim, name)[[0, 1]]) or name == "uf" and np.allclose(
            getattr(rest.bsim, name)[[1, 0]], getattr(one.bsim, name)[[0, 1]])
        assert np.array_equal(getattr(rest.bsim, name)[0], getattr(rest.bsim, name)[2])


def test_opp_variants_and_tree_reduce_to_one_ply():
    net = _net(2)
    leaf = ValueNetLeaf(_value())
    roots = _states(2)
    br = parse_branches("alt*")
    one = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5).search(roots, [0, 1], [1, 2])
    # opponent variants whose modification lasts 0 decisions = the base opponent each time
    opp = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5, opp_branches=parse_branches("samp:1.0x2"),
                   opp_left=0).search(roots, [0, 1], [1, 2])
    assert np.allclose(opp.values, one.values, atol=1e-5) and opp.timing["raw"].shape == (2, 6, 3)
    # live opponent variants: min <= base-opponent value, mean between min and max
    s = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5, opp_branches=parse_branches("samp:1.0x2"))
    mn = s.search(roots, [0, 1], [1, 2])
    raw = mn.timing["raw"]
    assert np.allclose(raw[..., 0], one.values, atol=1e-5) and np.allclose(mn.values, raw.min(-1))
    # 2-ply with 0-decision children: every child = the base continuation = the 1-ply leaf
    t = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5, children=parse_branches("alt*"),
                 child_window=0).search(roots, [0, 1], [1, 2])
    assert np.allclose(t.values, one.values, atol=1e-5)
    # best-first top-1: base + one other expanded, the rest -inf; its 1-ply values = the 1-ply search
    t = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5, children=parse_branches("alt*"),
                 expand=1).search(roots, [0, 1], [1, 2])
    assert np.allclose(t.timing["v1"], one.values, atol=1e-5)
    assert (np.isfinite(t.values).sum(1) == 2).all() and np.isfinite(t.values[:, 0]).all()
    full = Searcher(ModNetPolicy(net), leaf, br, window=5, leaf_decisions=5, children=parse_branches("alt*")).search(
        roots, [0, 1], [1, 2])
    fin = np.isfinite(t.values)
    assert np.allclose(t.values[fin], full.values[fin], atol=1e-5)
    assert t.timing["games"] == 2 * 6 + 2 * 2 * 6 and full.timing["games"] == 2 * 6 + 2 * 6 * 6
