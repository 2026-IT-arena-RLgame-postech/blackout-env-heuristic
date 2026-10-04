"""Opponent-blind search options (docs §34, blackout_env/sim/search.py): the opponent-variant aggregates
(``agg_fn``), extra opponent-model nets (``ModNetPolicy(opp_nets=...)``, ``Searcher(opp_net_idx=...)``) --
off or pointing at the main net = unchanged, otherwise the other team really plays that net -- and the
server / Unity handshake fields (``server_meta``, ``build_searcher(base=...)``)."""

import numpy as np
import pytest

from blackout_env.sim.batch import BatchRollout
from blackout_env.sim.search import (K_BASE, Branch, MatchConfig, ModNetPolicy, Searcher, ValueNetLeaf, agg_fn,
                                     deep_opp_nets, parse_branches, play_matches, search_config, searcher_kwargs)

from tests.test_sim_search import _net, _states, _value


def test_agg_fn():
    v = np.array([[1.0, 3.0, -2.0, 6.0], [0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(agg_fn("mean")(v), [2.0, 0.0])
    assert np.allclose(agg_fn("min")(v), [-2.0, 0.0])
    assert np.allclose(agg_fn("mix:0")(v), agg_fn("mean")(v)) and np.allclose(agg_fn("mix:1")(v), agg_fn("min")(v))
    assert np.allclose(agg_fn("mix:0.25")(v), [0.25 * -2 + 0.75 * 2, 0.0])
    s = agg_fn("softmin:10")(v)
    assert (s <= agg_fn("mean")(v) + 1e-12).all() and (s >= agg_fn("min")(v) - 1e-12).all()
    assert np.allclose(agg_fn("softmin:1e6")(v), [2.0, 0.0], atol=1e-4)
    assert np.allclose(agg_fn("softmin:1e-3")(v), [-2.0, 0.0], atol=1e-2)
    for bad in ("max", "mix:2", "softmin:0", "mean:1"):
        with pytest.raises((ValueError, AssertionError)):
            agg_fn(bad)


def test_opp_nets_off_or_same_net_change_nothing():
    net = _net(11)
    leaf = ValueNetLeaf(_value())
    br = parse_branches("alt*,samp:1.0x2")
    roots = _states(3)
    kw = dict(window=6, leaf_decisions=6, opp_branches=parse_branches("samp:1.0x2"), opp_agg="mean", opp_left=12)
    ref = Searcher(ModNetPolicy(net), leaf, br, **kw).search(roots, [0, 1, 0], [5, 6, 7])
    # an extra net that is never used / that is the main net itself
    a = Searcher(ModNetPolicy(net, opp_nets=[_net(12)]), leaf, br, **kw).search(roots, [0, 1, 0], [5, 6, 7])
    b = Searcher(ModNetPolicy(net, opp_nets=[net]), leaf, br, opp_net_idx=[0, 1, 1], **kw).search(roots, [0, 1, 0], [5, 6, 7])
    for got in (a, b):
        assert np.array_equal(ref.values, got.values) and np.array_equal(ref.choice, got.choice)
    # plain matches with an unused opponent net: unchanged
    cfg = MatchConfig(every=10, window=10, cap_s=2)
    s1 = Searcher(ModNetPolicy(net), leaf, br, window=10, leaf_decisions=5)
    s2 = Searcher(ModNetPolicy(net, opp_nets=[_net(12)]), leaf, br, window=10, leaf_decisions=5)
    r1 = play_matches([21, 22], [0, 1], (p := ModNetPolicy(net)), p, s1, cfg)
    r2 = play_matches([21, 22], [0, 1], (q := ModNetPolicy(net)), q, s2, cfg)
    for k in ("margin", "ticks", "chosen"):
        assert np.array_equal(r1[k], r2[k]), k


def test_opp_net_plays_the_other_team():
    """Variant j with net 1: the other team is the extra net for the whole rollout (window + leaf), our
    team the main net -- a two-policy BatchRollout; variant 0 stays the main net on both sides."""
    net, alt = _net(13), _net(14)
    leaf = ValueNetLeaf(_value())
    roots = _states(2)
    w, ld = 6, 6
    s = Searcher(ModNetPolicy(net, opp_nets=[alt]), leaf, [Branch(K_BASE)], window=w, leaf_decisions=ld,
                 opp_branches=[Branch(K_BASE), Branch(K_BASE)], opp_agg="mean", opp_left=w + ld, opp_net_idx=[0, 1])
    out = s.search(roots, [0, 1], [3, 4])
    raw = out.timing["raw"]                                  # [R, B, variants]
    teams = np.array([0, 1])
    same = BatchRollout(net, net).run(roots, w + ld, reseeds=[3, 4])
    assert np.allclose(raw[:, 0, 0], leaf(same.bsim, teams), atol=1e-4)
    for r, tm in enumerate((0, 1)):                          # our team 0 / 1 vs the alt net
        pa, pb = (net, alt) if tm == 0 else (alt, net)
        two = BatchRollout(pa, pb).run([roots[r]], w + ld, reseeds=[3 + r])
        assert np.allclose(raw[r, 0, 1], leaf(two.bsim, np.array([tm]))[0], atol=1e-4), r
    assert not np.allclose(raw[:, 0, 0], raw[:, 0, 1])
    assert s.config()["opp_net_idx"] == [0, 1]


def test_config_and_kwargs():
    deep = {"opp_branches": "samp:1.0x3", "opp_agg": "mix:0.3", "opp_left": 150, "opp_net_idx": [0, 0, 1, 1],
            "opp_nets": ["checkpoints/rollout_net/r5.pt"]}
    kw = searcher_kwargs(deep)
    assert "opp_nets" not in kw and kw["opp_net_idx"] == [0, 0, 1, 1] and kw["opp_agg"] == "mix:0.3"
    assert deep_opp_nets(deep) == ["checkpoints/rollout_net/r5.pt"] and deep_opp_nets(None) == []
    s = Searcher(None, None, parse_branches("alt*"), **kw)
    sc = {k: v for k, v in deep.items() if k != "opp_nets"}
    assert s.config() == search_config("alt*", 25, **{"leaf_decisions": 125, **sc})
    # all-zero net indices = no mixture (same config as without the option)
    z = Searcher(None, None, parse_branches("alt*"), **searcher_kwargs({"opp_branches": "samp:1.0x1", "opp_net_idx": [0, 0]}))
    assert "opp_net_idx" not in z.config() and z.opp_net_idx is None
    with pytest.raises(AssertionError):
        Searcher(None, None, parse_branches("alt*"), **searcher_kwargs({"opp_branches": "samp:1.0x1", "opp_net_idx": [0, 1, 1]}))


def test_server_meta_only_adds_fields_when_set():
    from blackout_env.sim.search_server import server_meta

    assert server_meta("a/r5.pt", "b/v.pt") == {"rollout": "r5.pt", "value": "v.pt"}
    m = server_meta("a/f3.pt", "b/v.pt", {"opp_nets": ["c/r5.pt"]}, "d/p14_v25.pt")
    assert m == {"rollout": "f3.pt", "value": "v.pt", "opp_nets": ["r5.pt"], "base": "p14_v25.pt", "overlay": True}
