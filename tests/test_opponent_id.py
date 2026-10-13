"""OpponentID: posterior deal of the search's opponent variants; Searcher's per-root net index layout."""

import numpy as np

from blackout_env.sim.search import OpponentID, Searcher


class _Net:
    """Always prefers bin ``b`` (logit 5), K = 32 + STAND."""

    def __init__(self, b):
        self.b = b

    def forward(self, g, t, a):
        q = np.zeros((len(g), 5, 33), np.float32)
        q[..., self.b] = 5.0
        return q, None


def test_posterior_follows_the_net_that_predicts_the_moves():
    oid = OpponentID([_Net(0), _Net(8), _Net(16)], variants=4, temper=1.0, half_life=1e9)
    oid.reset(2)
    inputs = (np.zeros((2, 13, 24, 24), np.float32), np.zeros((2, 4), np.float32), np.zeros((2, 10, 12), np.float32))
    up = np.zeros((2, 5, 2), np.float32)
    up[0, :, 1] = 1.0                       # game 0: every unit moves +y = bin 8
    up[1, :, 0] = 1.0                       # game 1: +x = bin 0
    for _ in range(3):
        oid.observe(np.arange(2), inputs, up)
    a = oid.assign(np.arange(2))
    assert a[0, 0] == 1 and (a[0] == 1).all()
    assert a[1, 0] == 0 and (a[1] == 0).all()


def test_flat_posterior_spreads_the_samples():
    oid = OpponentID([_Net(0), _Net(8)], variants=4, temper=1.0)
    oid.reset(1)
    a = oid.assign(np.arange(1))[0]
    assert a[0] == 0 and sorted(a[1:].tolist()) in ([0, 0, 1], [0, 1, 1])


def test_net_index_layout_is_root_branch_variant():
    s = Searcher.__new__(Searcher)
    s.opp_net_idx = None
    idx = s._net_idx([[0, 1, 2, 3], [4, 4, 4, 4]], r=2, nb=3, no=4)
    assert idx.tolist() == [0, 1, 2, 3] * 3 + [4] * 12
    assert s._net_idx(None, 2, 3, 4) is None
