"""Search with an external acting policy (docs §32, blackout_env/sim/search.py ``ExternalBasePolicy``,
``play_matches(base=...)``): off = unchanged; base = the rollout net itself = unchanged; a base-only
searcher plays exactly the base policy; overlay masks; the search window options."""

import numpy as np
import torch

from blackout_env.sim.batch import NetPolicy
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.sim.search import (K_ALT, K_ALTM, K_BASE, K_ROT, K_SAMP, K_STOP, Branch, MatchConfig,
                                     ModNetPolicy, Searcher, ValueNetLeaf, branch_units, overlay_base,
                                     parse_branches, play_matches)

from test_sim_search import _net, _states, _value


def test_branch_units_and_overlay():
    m = branch_units([K_BASE, K_ALT, K_ROT, K_STOP, K_SAMP, K_ALTM], [3, 1, 2, 4, 0, 0b01001])
    assert m.tolist() == [[False] * 5, [False, True, False, False, False], [False, False, True, False, False],
                          [False, False, False, False, True], [True] * 5, [True, False, False, True, False]]
    a, b = np.ones((2, 5, 2), np.float32), np.zeros((2, 5, 2), np.float32)
    got = overlay_base(a, b, [K_BASE, K_ALT], [0, 2], overlay=True)
    assert got[0].sum() == 0 and got[1, 2].tolist() == [1, 1] and got[1].sum() == 2
    got = overlay_base(a, b, [K_BASE, K_ALT], [0, 2], overlay=False)
    assert got[0].sum() == 0 and (got[1] == 1).all()


def test_base_equal_to_rollout_net_changes_nothing():
    net = _net(4)
    leaf = ValueNetLeaf(_value())
    br = parse_branches("alt*,rot1:90,samp:1.0x2,pair03")
    roots = _states(3)
    ref = Searcher(ModNetPolicy(net), leaf, br, window=6, leaf_decisions=6).search(roots, [0, 1, 0], [5, 6, 7])
    got = Searcher(ModNetPolicy(net), leaf, br, window=6, leaf_decisions=6, base_policy=net).search(roots, [0, 1, 0], [5, 6, 7])
    assert np.array_equal(ref.values, got.values) and np.array_equal(ref.choice, got.choice)
    seeds, sides = [21, 22, 21, 22], [0, 0, 1, 1]
    cfg = MatchConfig(every=10, window=10, cap_s=2, trace_s=(1,))
    s1 = Searcher(ModNetPolicy(net), leaf, br, window=10, leaf_decisions=5)
    a = play_matches(seeds, sides, (p := ModNetPolicy(net)), p, s1, cfg)
    s2 = Searcher(ModNetPolicy(net), leaf, br, window=10, leaf_decisions=5, base_policy=net)
    b = play_matches(seeds, sides, (q := ModNetPolicy(net)), q, s2, cfg, base=net)
    for k in ("margin", "ticks", "chosen", "first_ours", "lead_at"):
        assert np.array_equal(a[k], b[k]), k


def test_base_only_search_plays_the_base_policy():
    net, ext = _net(5), _net(6)
    opp = _net(7)
    seeds, sides = [31, 32, 31, 32], [0, 0, 1, 1]
    cfg = MatchConfig(every=10, window=10, cap_s=3, trace_s=(1, 2))
    direct = play_matches(seeds, sides, ModNetPolicy(ext), opp, None, cfg)   # the base net as the rollout net
    plain = play_matches(seeds, sides, ModNetPolicy(net), opp, None, cfg, base=ext)
    s = Searcher(ModNetPolicy(net), ValueNetLeaf(_value()), [Branch(K_BASE)], window=10, leaf_decisions=5, base_policy=ext)
    srch = play_matches(seeds, sides, ModNetPolicy(net), opp, s, cfg, base=ext)
    for k in ("margin", "ticks", "first_ours", "first_opp", "lead_at"):
        assert np.array_equal(direct[k], plain[k]), k
        assert np.array_equal(plain[k], srch[k]), k
    assert (srch["searches"] == 8).all()


def test_external_base_window_plays_the_base_then_the_net():
    """The base branch: our team plays the base policy for the window, then the rollout net; the other
    team the rollout net throughout -- the same as a plain two-policy BatchRollout."""
    from blackout_env.sim.batch import BatchRollout

    net, ext = _net(8), _net(9)
    leaf = ValueNetLeaf(_value())
    roots = _states(2)
    for w, ld in ((8, 0), (8, 4)):
        a = Searcher(ModNetPolicy(net), leaf, [Branch(K_BASE)], window=w, leaf_decisions=ld, base_policy=ext).search(roots, [0, 0], [3, 4])
        r = BatchRollout(ext, net).run(roots, w, reseeds=[3, 4])
        if ld:
            r = BatchRollout(net, net).run(None, ld, bsim=r.bsim)
        assert np.allclose(a.values[:, 0], leaf(r.bsim, np.array([0, 0]))), (w, ld)


def test_search_window_bounds():
    net = _net(10)
    s = Searcher(ModNetPolicy(net), ValueNetLeaf(_value()), [Branch(K_BASE)], window=5, leaf_decisions=2)
    cfg = MatchConfig(every=5, window=5, cap_s=2, search_from=10, search_until=30)
    res = play_matches([41, 42], [0, 1], (p := ModNetPolicy(net)), p, s, cfg)
    assert (res["searches"] == 4).all()  # t = 10, 15, 20, 25
