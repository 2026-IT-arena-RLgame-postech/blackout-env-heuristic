"""Batched S3 search server: many clients (Unity workers, data generators) send search roots, one
process per GPU runs them as one ``Searcher`` batch (docs/policy_v2_design_ko.md §23 "탐색 배치 서버").

Why: a single-root search (1 root x 8 branches x 150 decisions) is launch/latency bound -- ~0.6 s
per search when 12 Unity workers each run their own on a shared GPU, while 128 roots batched in
the lockstep simulator cost ~39 ms per root.  The server owns the rollout net (CUDA graphs per
batch-size bucket), the value net and the batched simulator; a request carries the root states
(``encode_state``: the ``SimState`` arrays + the storage choice, ~62 KB), the searching team, the
reseed and the margin, and gets back every branch's leaf value and the chosen branch.

Protocol: a Unix stream socket, frames = 8-byte little-endian length + pickle (local and trusted:
only this user's processes can open the socket, mode 0600).  Ops: ``hello`` (config handshake:
branches / window / leaf decisions / nets must match what the client expects), ``search``,
``stats``.  Dynamic batching: requests are gathered for ``max_wait_ms`` after the first one
arrives, or until ``max_roots`` roots are queued; requests that arrive while a batch runs go into
the next batch (so the wait only matters when the server is idle).

Robustness (no orphans, no deadlocks):
* the server exits when every client that connected has gone (``linger_s`` later), when nothing
  connected within ``startup_s``, when ``--parent-pid`` dies, or on SIGTERM/SIGINT -- and removes
  its socket file;
* ``SearchClient.search`` never blocks forever: connect retries up to ``connect_timeout``, every
  request has a ``timeout``; on any failure it logs one warning, returns ``None`` (the caller plays
  the plain rollout net = the base branch) and retries the connection after ``retry_s``.

    python -m blackout_env.sim.search_server serve --socket /tmp/bo-search.sock --device cuda \\
        --parent-pid $$            # (examples/unity_search_eval.py --server spawn does this)
    cli = SearchClient("/tmp/bo-search.sock", expect=searcher_config)
    out = cli.search([shadow.copy()], [side], [reseed], margin=3.0)   # SearchReply or None
"""

from __future__ import annotations

import argparse
import os
import pickle
import selectors
import signal
import socket
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .state import SimState

HEADER = struct.Struct("<Q")
PROTO = 1
_LAYOUTS: dict = {}


# ---------------------------------------------------------------------------------- state codec

def encode_state(sim: SimState) -> dict:
    """A picklable, layout-free copy of ``sim``: its arrays + the storage choice (the layout is
    rebuilt / cached on the other side) + the background and AgentAct order."""
    d = {name: getattr(sim, name) for name in SimState._ARRAYS}
    d["chosen"] = tuple(int(c) for c in sim.layout.chosen)
    d["order"] = np.asarray(sim.order)
    d["bg"] = sim.bg
    return d


def decode_state(d: dict) -> SimState:
    from .mapdata import build_layout

    lay = _LAYOUTS.get(d["chosen"])
    if lay is None:
        lay = _LAYOUTS[d["chosen"]] = build_layout(d["chosen"])
    sim = object.__new__(SimState)
    sim.layout, sim.order, sim.bg = lay, np.asarray(d["order"], np.int64), np.asarray(d["bg"])
    sim.M = (lay.tile, lay.region, lay.reg_kind, lay.reg_team, lay.reg_in, lay.reg_out,
             lay.reg_locked, lay.reg_tiles, lay.reg_ntiles, lay.walk, lay.half)
    for name in SimState._ARRAYS:
        setattr(sim, name, np.array(d[name], copy=True))
    return sim


# ---------------------------------------------------------------------------------- framing

def _send(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("search server closed the connection")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket):
    (n,) = HEADER.unpack(_recv_exact(sock, HEADER.size))
    return pickle.loads(_recv_exact(sock, n))


# ---------------------------------------------------------------------------------- server

@dataclass
class _Req:
    conn: socket.socket
    rid: int
    roots: list
    teams: list
    reseeds: list
    margin: float
    t_arrive: float


@dataclass
class _Conn:
    sock: socket.socket
    buf: bytearray = field(default_factory=bytearray)
    pid: int = -1


class SearchServer:
    """``serve()`` until the clients are gone.  ``searcher``: a ``search.Searcher`` (its own margin
    is ignored -- each request brings one).  ``meta``: extra handshake fields (net paths)."""

    def __init__(self, searcher, path: str, max_roots: int = 32, max_wait_ms: float = 3.0,
                 linger_s: float = 5.0, startup_s: float = 600.0, parent_pid: int | None = None,
                 meta: dict | None = None, log=None):
        self.searcher, self.path = searcher, path
        self.max_roots, self.max_wait = int(max_roots), max_wait_ms / 1e3
        self.linger_s, self.startup_s, self.parent_pid = linger_s, startup_s, parent_pid
        self.config = {**searcher.config(), **(meta or {}), "proto": PROTO}
        self.log = log or (lambda m: print(f"[search-server {os.getpid()}] {m}", file=sys.stderr, flush=True))
        self.conns: dict[int, _Conn] = {}
        self.queue: list[_Req] = []
        self.stop = False
        self.stats = {"batches": 0, "roots": 0, "requests": 0, "search_s": 0.0, "max_batch_roots": 0,
                      "batch_hist": {}, "queue_s": 0.0, "clients_seen": 0}

    # -- lifecycle
    def _listen(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            srv.bind(self.path)
        finally:
            os.umask(old)
        srv.listen(256)
        srv.setblocking(False)
        return srv

    def _should_exit(self, now: float, t_start: float, t_empty: float | None) -> str | None:
        if self.stop:
            return "signal"
        if self.parent_pid and not _alive(self.parent_pid):
            return f"parent {self.parent_pid} gone"
        if self.stats["clients_seen"] == 0 and now - t_start > self.startup_s:
            return f"no client within {self.startup_s:.0f} s"
        if self.stats["clients_seen"] and not self.conns and t_empty is not None and now - t_empty > self.linger_s:
            return "all clients gone"
        return None

    def serve(self) -> dict:
        srv = self._listen()
        sel = selectors.DefaultSelector()
        sel.register(srv, selectors.EVENT_READ, None)
        t_start, t_empty = time.monotonic(), None
        self.log(f"listening on {self.path} (max_roots {self.max_roots}, wait {self.max_wait * 1e3:.1f} ms) config {self.config}")
        reason = None
        try:
            while True:
                now = time.monotonic()
                reason = self._should_exit(now, t_start, t_empty)
                if reason:
                    break
                if self.queue:
                    timeout = max(0.0, self.queue[0].t_arrive + self.max_wait - now)
                else:
                    timeout = 0.5
                for key, _ in sel.select(timeout):
                    if key.data is None:
                        self._accept(srv, sel)
                    else:
                        self._read(key.fileobj, sel)
                if not self.conns and t_empty is None:
                    t_empty = time.monotonic()
                elif self.conns:
                    t_empty = None
                now = time.monotonic()
                queued = sum(len(r.roots) for r in self.queue)
                if self.queue and (queued >= self.max_roots or now >= self.queue[0].t_arrive + self.max_wait):
                    self._run_batch()
        finally:
            for c in list(self.conns.values()):
                c.sock.close()
            sel.close()
            srv.close()
            if os.path.exists(self.path):
                os.unlink(self.path)
            self.log(f"exit ({reason}): {self.summary()}")
        return self.stats

    def summary(self) -> str:
        s = self.stats
        b = max(1, s["batches"])
        return (f"{s['batches']} batches, {s['roots']} roots ({s['roots'] / b:.1f}/batch, max {s['max_batch_roots']}), "
                f"{s['search_s'] / b * 1e3:.0f} ms/batch, {s['search_s'] / max(1, s['roots']) * 1e3:.1f} ms/root, "
                f"clients {s['clients_seen']}")

    # -- io
    def _accept(self, srv, sel):
        try:
            sock, _ = srv.accept()
        except BlockingIOError:
            return
        sock.setblocking(False)
        self.conns[sock.fileno()] = _Conn(sock)
        sel.register(sock, selectors.EVENT_READ, "c")
        self.stats["clients_seen"] += 1

    def _drop(self, sock, sel):
        c = self.conns.pop(sock.fileno(), None)
        try:
            sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        sock.close()
        self.queue = [r for r in self.queue if r.conn is not sock]
        if c is not None and not self.conns:
            self.log(f"last client gone ({self.summary()})")

    def _read(self, sock, sel):
        c = self.conns.get(sock.fileno())
        if c is None:
            return
        try:
            chunk = sock.recv(1 << 20)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._drop(sock, sel)
            return
        c.buf += chunk
        while len(c.buf) >= HEADER.size:
            (n,) = HEADER.unpack_from(c.buf)
            if len(c.buf) < HEADER.size + n:
                break
            msg = pickle.loads(bytes(c.buf[HEADER.size:HEADER.size + n]))
            del c.buf[:HEADER.size + n]
            self._handle(c, msg, sel)

    def _reply(self, sock, obj) -> bool:
        try:
            sock.setblocking(True)
            sock.settimeout(10.0)
            _send(sock, obj)
            sock.setblocking(False)
            return True
        except OSError:
            return False

    def _handle(self, c: _Conn, msg: dict, sel):
        op = msg.get("op")
        if op == "hello":
            c.pid = int(msg.get("pid", -1))
            self._reply(c.sock, {"ok": True, "config": self.config, "pid": os.getpid()})
        elif op == "search":
            roots = [decode_state(d) for d in msg["roots"]]
            self.queue.append(_Req(c.sock, msg["id"], roots, list(msg["teams"]), list(msg["reseeds"]),
                                   float(msg.get("margin", 0.0)), time.monotonic()))
        elif op == "stats":
            self._reply(c.sock, {"ok": True, "stats": self.stats, "config": self.config})
        else:
            self._reply(c.sock, {"ok": False, "error": f"unknown op {op!r}"})

    def _run_batch(self):
        take, n = [], 0
        while self.queue and (not take or n + len(self.queue[0].roots) <= self.max_roots):
            r = self.queue.pop(0)
            take.append(r)
            n += len(r.roots)
        roots = [s for r in take for s in r.roots]
        teams = [t for r in take for t in r.teams]
        reseeds = [k for r in take for k in r.reseeds]
        t0 = time.monotonic()
        out = self.searcher.search(roots, teams, reseeds)
        dt = time.monotonic() - t0
        from .search import choose

        s = self.stats
        s["batches"] += 1
        s["roots"] += n
        s["requests"] += len(take)
        s["search_s"] += dt
        s["max_batch_roots"] = max(s["max_batch_roots"], n)
        s["batch_hist"][n] = s["batch_hist"].get(n, 0) + 1
        k = 0
        for r in take:
            m = len(r.roots)
            vals = out.values[k:k + m]
            k += m
            q = t0 - r.t_arrive
            s["queue_s"] += q
            self._reply(r.conn, {"ok": True, "id": r.rid, "values": vals, "choice": choose(vals, r.margin),
                                 "batch_roots": n, "server_s": dt, "queue_s": q})


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------------- client

@dataclass
class SearchReply:
    choice: np.ndarray    # [R]
    values: np.ndarray    # [R, B] leaf values (points, the searching team)
    batch_roots: int      # roots in the server batch this request ran in
    server_s: float       # that batch's search time
    queue_s: float        # time the request waited in the server queue
    wall_s: float         # client round trip


class SearchClient:
    """Blocking client.  ``expect``: handshake fields that must equal the server's (e.g.
    ``Searcher.config()``); a mismatch raises (a misconfigured server is a bug, not an outage).
    Outages (no server, dead server, timeout) never raise from ``search``: it returns None."""

    def __init__(self, path: str, expect: dict | None = None, timeout: float = 60.0,
                 connect_timeout: float = 120.0, retry_s: float = 30.0, log=None):
        self.path, self.expect = path, dict(expect or {})
        self.timeout, self.connect_timeout, self.retry_s = timeout, connect_timeout, retry_s
        self.log = log or (lambda m: print(f"[search-client {os.getpid()}] {m}", file=sys.stderr, flush=True))
        self.sock: socket.socket | None = None
        self.down_until = 0.0
        self.failures = 0
        self.server_config: dict | None = None
        self._rid = 0
        self._first = True

    def connect(self, wait: float | None = None) -> bool:
        deadline = time.monotonic() + (self.connect_timeout if wait is None else wait)
        last = None
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect(self.path)
                _send(s, {"op": "hello", "pid": os.getpid(), "proto": PROTO})
                rep = _recv(s)
                break
            except (OSError, ConnectionError, EOFError) as e:
                last = e
                try:
                    s.close()
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    self.log(f"WARNING cannot reach search server {self.path}: {last!r}")
                    return False
                time.sleep(0.2)
        cfg = rep.get("config", {})
        bad = {k: (v, cfg.get(k)) for k, v in self.expect.items() if cfg.get(k) != v}
        if bad:
            s.close()
            raise RuntimeError(f"search server config mismatch (expected, server): {bad}")
        self.sock, self.server_config = s, cfg
        return True

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _fail(self, why) -> None:
        self.failures += 1
        self.close()
        self.down_until = time.monotonic() + self.retry_s
        self.log(f"WARNING search server unavailable ({why!r}); playing the plain rollout net, retry in {self.retry_s:.0f} s")

    def search(self, roots, teams, reseeds, margin: float = 0.0) -> SearchReply | None:
        if self.sock is None:
            if time.monotonic() < self.down_until:
                return None
            wait = self.connect_timeout if self._first else 2.0
            self._first = False
            if not self.connect(wait):
                self._fail("connect failed")
                return None
        self._rid += 1
        t0 = time.perf_counter()
        try:
            _send(self.sock, {"op": "search", "id": self._rid, "roots": [encode_state(s) for s in roots],
                              "teams": [int(t) for t in teams], "reseeds": [int(k) for k in reseeds],
                              "margin": float(margin)})
            rep = _recv(self.sock)
        except (OSError, ConnectionError, EOFError, pickle.UnpicklingError) as e:
            self._fail(e)
            return None
        if not rep.get("ok") or rep.get("id") != self._rid:
            self._fail(f"bad reply {rep.get('error', rep.get('id'))}")
            return None
        return SearchReply(np.asarray(rep["choice"]), np.asarray(rep["values"]), int(rep["batch_roots"]),
                           float(rep["server_s"]), float(rep["queue_s"]), time.perf_counter() - t0)

    def stats(self) -> dict | None:
        try:
            _send(self.sock, {"op": "stats"})
            return _recv(self.sock)["stats"]
        except (OSError, ConnectionError, EOFError, AttributeError):
            return None


# ---------------------------------------------------------------------------------- building / CLI

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"


def view_buckets(max_views: int) -> list[int]:
    """CUDA-graph batch buckets (in views = 2 x games): multiples of 16 to 128, then ~x1.25 steps."""
    out, b = list(range(16, 129, 16)), 128
    while b < max_views:
        b = int(-(-b * 1.25 // 16) * 16)
        out.append(b)
    return out


def build_searcher(rollout: str = R5, value: str = VALUE, branches: str = "alt*,samp:1.0x2", window: int = 25,
                   leaf_decisions: int = 125, device: str = "cuda", cuda_graphs: bool = True,
                   max_roots: int = 32, repo: str | None = None):
    """The Searcher the Unity eval uses (unity_search_eval._init_worker), graphs bucketed for a server."""
    from .batch import NetPolicy
    from .rollout_net import RolloutNet
    from .search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches

    def at(p):
        return p if repo is None or os.path.isabs(p) else os.path.join(repo, p)

    br = parse_branches(branches)
    buckets = view_buckets(2 * len(br) * max_roots) if cuda_graphs else None
    pol = ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(at(rollout)), device=device, cuda_graphs=cuda_graphs,
                                 graph_buckets=buckets))
    return Searcher(pol, ValueNetLeaf(at(value), device), br, window=window, leaf_decisions=leaf_decisions)


def _serve(args):
    import torch

    torch.set_num_threads(args.threads)
    import numba

    numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
    searcher = build_searcher(args.rollout, args.value, args.branches, args.window, args.leaf, args.device,
                              not args.no_cuda_graphs, args.max_roots, repo=args.repo)
    # warm-up off the clock: numba compile, the bucketed CUDA graphs for 1 and max_roots roots
    for n in (1, args.max_roots):
        searcher.search([SimState.from_seed(1 + k) for k in range(n)], [k % 2 for k in range(n)], list(range(n)))
    server = SearchServer(searcher, args.socket, args.max_roots, args.max_wait_ms, args.linger_s, args.startup_s,
                          args.parent_pid, meta={"rollout": os.path.basename(args.rollout), "value": os.path.basename(args.value)})
    signal.signal(signal.SIGTERM, lambda *a: setattr(server, "stop", True))
    signal.signal(signal.SIGINT, lambda *a: setattr(server, "stop", True))
    stats = server.serve()
    if args.stats:
        import json

        with open(args.stats, "w") as fh:
            json.dump({**stats, "batch_hist": {str(k): v for k, v in sorted(stats["batch_hist"].items())},
                       "config": server.config}, fh, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("serve")
    q.add_argument("--socket", required=True)
    q.add_argument("--rollout", default=R5)
    q.add_argument("--value", default=VALUE)
    q.add_argument("--repo", default=None, help="base directory of relative checkpoint paths")
    q.add_argument("--branches", default="alt*,samp:1.0x2")
    q.add_argument("--window", type=int, default=25)
    q.add_argument("--leaf", type=int, default=125)
    q.add_argument("--device", default="cuda")
    q.add_argument("--no-cuda-graphs", action="store_true")
    q.add_argument("--max-roots", type=int, default=32)
    q.add_argument("--max-wait-ms", type=float, default=3.0)
    q.add_argument("--threads", type=int, default=4, help="torch + numba threads")
    q.add_argument("--linger-s", type=float, default=5.0)
    q.add_argument("--startup-s", type=float, default=600.0)
    q.add_argument("--parent-pid", type=int, default=None)
    q.add_argument("--stats", default=None, help="write batch statistics (json) here at exit")
    args = p.parse_args(argv)
    _serve(args)


if __name__ == "__main__":
    main()
