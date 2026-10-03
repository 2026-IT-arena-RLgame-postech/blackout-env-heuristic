"""Batched search server (blackout_env/sim/search_server.py): state codec, results through the server
equal the in-process search bit for bit (per-sample forwards, so batching cannot change rounding),
concurrent clients get batched, and nothing deadlocks or lingers when either side goes away."""

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from blackout_env.sim import SimState
from blackout_env.sim.batch import NetPolicy
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, choose, parse_branches
from blackout_env.sim.search_server import SearchClient, SearchServer, decode_state, encode_state, view_buckets

BRANCHES = "alt0,alt3,samp:1.0x1"


def _searcher():
    torch.manual_seed(1)
    net = RolloutNet(arch="attn", d=32, layers=1, heads=2, hidden=32, c1=8, c2=8, c3=8, glob=16).eval()
    from blackout_env.value.model import ValueNet

    torch.manual_seed(3)
    vnet = ValueNet(hidden_size=64).eval()
    # batch-1 forwards everywhere: a root's result cannot depend on what it was batched with
    pol = ModNetPolicy(NetPolicy(net, per_sample=True))
    return Searcher(pol, ValueNetLeaf(vnet, batch=1), parse_branches(BRANCHES), window=6, leaf_decisions=8)


def _roots(n=5, steps=40):
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        s = SimState.from_seed(500 + k)
        for _ in range(steps + 7 * k):
            s.step(rng.normal(size=(10, 2)).astype(np.float32).clip(-1, 1))
        out.append(s)
    return out


_DIRS = []


def _sock():
    """A short socket path (AF_UNIX paths are limited to ~104 bytes: no pytest tmp_path on macOS)."""
    d = tempfile.mkdtemp(prefix="bo-ss-", dir="/tmp")
    _DIRS.append(d)
    return str(Path(d) / "s.sock")


@pytest.fixture(autouse=True)
def _cleanup_dirs():
    yield
    while _DIRS:
        shutil.rmtree(_DIRS.pop(), ignore_errors=True)


class _Thread:
    def __init__(self, server):
        self.server = server
        self.result = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        self.result = self.server.serve()


def test_state_codec_roundtrip():
    for s in _roots(3):
        d = decode_state(encode_state(s))
        for name in SimState._ARRAYS:
            assert np.array_equal(getattr(d, name), getattr(s, name)), name
        assert np.array_equal(d.raw_state(), s.raw_state()) and np.array_equal(d.raw_map(), s.raw_map())
        a, b = s.copy(), d
        mv = np.random.default_rng(0).normal(size=(30, 10, 2)).astype(np.float32).clip(-1, 1)
        for m in mv:
            a.step(m)
            b.step(m)
        assert np.array_equal(a.raw_state(), b.raw_state()) and np.array_equal(a.rng, b.rng)


def test_choose_matches_margin_rule():
    v = np.array([[0.0, 2.0, 5.0], [1.0, 3.9, 0.0], [0.0, 0.0, 0.0]])
    assert choose(v, 3.0).tolist() == [2, 0, 0]
    assert choose(v, np.array([6.0, 2.0, 0.0])).tolist() == [0, 1, 0]


def test_view_buckets():
    b = view_buckets(8 * 2 * 32)
    assert b[:8] == list(range(16, 129, 16)) and b[-1] >= 512 and all(x % 16 == 0 for x in b)


def test_server_equals_in_process_and_batches_concurrent_clients():
    path = _sock()
    ref_s = _searcher()
    roots = _roots(5)
    teams = [0, 1, 1, 0, 1]
    reseeds = [11, 22, 33, 44, 55]
    ref = [ref_s.search([r.copy()], [t], [k]) for r, t, k in zip(roots, teams, reseeds)]
    # wide wait so the concurrent requests share a batch
    th = _Thread(SearchServer(_searcher(), path, max_roots=16, max_wait_ms=300, linger_s=0.5, startup_s=30))
    got = [None] * len(roots)

    def client(j):
        cli = SearchClient(path, expect=ref_s.config(), connect_timeout=20)
        got[j] = cli.search([roots[j]], [teams[j]], [reseeds[j]], margin=1.0)
        cli.close()

    ts = [threading.Thread(target=client, args=(j,)) for j in range(len(roots))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    th.t.join(10)
    assert not th.t.is_alive(), "server must exit once its clients are gone"
    assert not Path(path).exists()
    for j, (g, r) in enumerate(zip(got, ref)):
        assert g is not None
        assert g.values.tobytes() == r.values.tobytes(), j        # bit-exact leaf values
        assert g.choice.tolist() == choose(r.values, 1.0).tolist()
    assert max(g.batch_roots for g in got) > 1                     # the requests were batched
    assert th.result["roots"] == len(roots)


def test_multi_root_request_and_stats():
    path = _sock()
    th = _Thread(SearchServer(_searcher(), path, max_roots=4, max_wait_ms=1, linger_s=0.2, startup_s=30))
    cli = SearchClient(path, connect_timeout=20)
    roots = _roots(3)
    rep = cli.search(roots, [0, 1, 0], [1, 2, 3], margin=0.0)
    assert rep.values.shape == (3, len(parse_branches(BRANCHES))) and rep.choice.shape == (3,)
    assert cli.stats()["roots"] == 3
    cli.close()
    th.t.join(10)
    assert not th.t.is_alive()


def test_config_mismatch_raises():
    path = _sock()
    th = _Thread(SearchServer(_searcher(), path, linger_s=0.2, startup_s=30))
    cli = SearchClient(path, expect={"window": 25}, connect_timeout=20)
    with pytest.raises(RuntimeError, match="mismatch"):
        cli.connect()
    th.server.stop = True
    th.t.join(10)


def test_client_falls_back_when_server_missing_or_dies():
    cli = SearchClient(_sock(), connect_timeout=0.5, retry_s=60, log=lambda m: None)
    t0 = time.monotonic()
    assert cli.search(_roots(1), [0], [1]) is None
    assert cli.search(_roots(1), [0], [1]) is None   # inside retry_s: no reconnect attempt
    assert time.monotonic() - t0 < 5 and cli.failures == 1
    # a server that goes away mid-session: the next search returns None instead of hanging
    path = _sock()
    th = _Thread(SearchServer(_searcher(), path, linger_s=0.2, startup_s=30))
    cli = SearchClient(path, connect_timeout=20, timeout=5, log=lambda m: None)
    assert cli.search(_roots(1), [0], [1]) is not None
    th.server.stop = True
    th.t.join(10)
    assert not th.t.is_alive()
    t0 = time.monotonic()
    assert cli.search(_roots(1), [0], [1]) is None
    assert time.monotonic() - t0 < 6


def test_server_exits_when_parent_dies_or_nobody_connects():
    parent = subprocess.Popen(["sleep", "0.5"])
    th = _Thread(SearchServer(_searcher(), _sock(), startup_s=60, parent_pid=parent.pid))
    parent.wait()
    th.t.join(5)
    assert not th.t.is_alive()
    th = _Thread(SearchServer(_searcher(), _sock(), startup_s=0.3))
    th.t.join(5)
    assert not th.t.is_alive()


def test_packed_render_unpacks_to_the_float_inputs():
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, unpack_graphic

    bs = BatchSim(_roots(6, steps=120))
    n = bs.n
    g = np.zeros((n, 2, N_CH, 24, 24), np.float32)
    t = np.zeros((n, 2, 4), np.float32)
    a = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
    bs.render(bs.all, g, t, a)
    p = np.zeros((n, 2, 24, 24), np.int16)
    t2, a2 = np.zeros_like(t), np.zeros_like(a)
    bs.render_packed(bs.all, p, t2, a2)
    u = unpack_graphic(torch.from_numpy(p.reshape(-1, 24, 24))).numpy().reshape(g.shape)
    assert u.tobytes() == g.tobytes() and t2.tobytes() == t.tobytes() and a2.tobytes() == a.tobytes()
    assert g[:, :, 8].any()  # batteries on the map


def test_packed_searcher_equals_float_searcher():
    roots = _roots(3)
    ref = _searcher().search(roots, [0, 1, 0], [5, 6, 7])
    s = _searcher()
    s.roll.packed = True
    s.roll._buf = None
    got = s.search(roots, [0, 1, 0], [5, 6, 7])
    assert got.values.tobytes() == ref.values.tobytes()


def test_unpack_graphic_every_code():
    from blackout_env.sim.batch import unpack_graphic

    base, bat, item = np.meshgrid(np.arange(8), np.arange(16), np.arange(8), indexing="ij")
    p = (base | bat << 3 | item << 7).astype(np.int16).reshape(1, -1, 1)
    g = unpack_graphic(torch.from_numpy(p)).numpy()[0, :, :, 0]  # [13, codes]
    ref = np.zeros_like(g)
    ref[base.ravel(), np.arange(p.size)] = 1.0
    ref[8] = bat.ravel().astype(np.float32) / np.float32(15.0)
    it = item.ravel()
    ok = (it >= 1) & (it <= 4)
    ref[8 + it[ok], np.nonzero(ok)[0]] = 1.0
    assert g.tobytes() == ref.tobytes()
