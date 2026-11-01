"""Heuristic opponents in worker processes, overlapped with the GPU forwards (GRPO).

``HeuristicPolicy`` runs one Python heuristic instance per (game, team) on the main thread: at GRPO's branch
scale (768 games, ~35k heuristic decisions per iteration, ~1.2 ms each) that is the largest single cost of an
iteration, while the GPU idles.  ``ParallelHeuristic`` keeps the same per-(game, team) instances -- same
``make_opponent(descriptor, seed + 2 b + team)`` construction, same call sequence per instance, hence the same
moves -- but hosts them in ``HeuristicWorkers`` processes (game ``b`` always goes to worker ``b % n``) and splits
``moves`` into ``start`` (ship the raw states, return at once) and ``finish`` (collect), so the caller can run the
network forwards in between.

Workers are ``spawn``ed with ``CUDA_VISIBLE_DEVICES=""`` and single-threaded numba/OMP/MKL.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from blackout_env.sim.batch import HeuristicPolicy


def _worker(conn) -> None:
    from blackout_env.sim.state import obs_from_raw
    from blackout_env.train.distill.opponents import make_opponent

    inst: dict = {}   # (policy key, game, team) -> heuristic instance
    while True:
        msg = conn.recv()
        op = msg[0]
        if op == "stop":
            return
        if op == "reset":       # drop one policy's instances
            for k in [k for k in inst if k[0] == msg[1]]:
                del inst[k]
        elif op == "forget":    # drop given games of one policy
            games = set(msg[2])
            for k in [k for k in inst if k[0] == msg[1] and k[1] in games]:
                del inst[k]
        elif op == "ninst":
            conn.send(sum(1 for k in inst if k[0] == msg[1]))
        elif op == "act":
            _, key, desc, seed, bs, tms, rs, rm, ticket = msg
            out = np.zeros((len(bs), 5, 2), np.float32)
            obs_cache: dict = {}
            for j in range(len(bs)):
                b, tm = int(bs[j]), int(tms[j])
                if (key, b, tm) not in inst:
                    inst[(key, b, tm)] = make_opponent(desc, seed + 2 * b + tm)
                if b not in obs_cache:
                    obs_cache[b] = obs_from_raw(rs[j], rm[j])
                obs = obs_cache[b]
                names = [f"unit_{i}" for i in range(5 * tm, 5 * tm + 5)]
                acts = inst[(key, b, tm)].act({n: obs[n] for n in names})
                for n_, a in acts.items():
                    out[j, int(n_.split("_")[1]) - 5 * tm] = np.clip(np.asarray(a, np.float32), -1, 1)
            conn.send((ticket, out))


class HeuristicWorkers:
    def __init__(self, n: int):
        ctx = mp.get_context("spawn")
        keep = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update({"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        try:
            self.n, self.conns, self.procs = n, [], []
            for _ in range(n):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b,), daemon=True)
                p.start()
                b.close()
                self.conns.append(a)
                self.procs.append(p)
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._keys = 0
        self._tickets = 0
        self._stash = [dict() for _ in range(n)]   # per worker: replies that arrived while waiting for another ticket

    def send_act(self, k: int, *args) -> int:
        """Send an ``act`` request to worker ``k``; returns its ticket (replies may be collected in any order)."""
        self._tickets += 1
        self.conns[k].send(("act", *args, self._tickets))
        return self._tickets

    def recv_act(self, k: int, ticket: int):
        st = self._stash[k]
        while ticket not in st:
            t, out = self.conns[k].recv()
            st[t] = out
        return st.pop(ticket)

    def new_key(self) -> int:
        self._keys += 1
        return self._keys

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


class ParallelHeuristic(HeuristicPolicy):
    """``HeuristicPolicy`` over ``HeuristicWorkers`` (see module doc).  ``instances`` is not kept locally."""

    def __init__(self, descriptor: str, seed: int, workers: HeuristicWorkers):
        super().__init__(descriptor, seed)
        self.w, self.key = workers, workers.new_key()

    def reset(self, n: int) -> None:
        for c in self.w.conns:
            c.send(("reset", self.key))

    def forget_games(self, games) -> None:
        games = [int(b) for b in games]
        if games:
            for c in self.w.conns:
                c.send(("forget", self.key, games))

    def n_instances(self) -> int:
        for c in self.w.conns:
            c.send(("ninst", self.key))
        return sum(c.recv() for c in self.w.conns)

    def start(self, bsim, idx, tms):
        """Ship games ``idx`` (team ``tms[j]`` for row j) to the workers; returns a handle for ``finish``."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        tms = np.asarray(tms)
        rs, rm = bsim.raw_batch(idx)
        owner = idx % self.w.n
        sent = []
        for k, c in enumerate(self.w.conns):
            rows = np.nonzero(owner == k)[0]
            if len(rows):
                sent.append((k, rows, self.w.send_act(k, self.key, self.descriptor, self.seed, idx[rows], tms[rows], rs[rows], rm[rows])))
        return len(idx), sent

    def finish(self, handle) -> np.ndarray:
        n, sent = handle
        out = np.zeros((n, 5, 2), np.float32)
        for k, rows, ticket in sent:
            out[rows] = self.w.recv_act(k, ticket)
        return out

    def moves(self, bsim, idx, teams, inputs):
        idx = np.asarray(idx)
        n, nt = len(idx), len(teams)
        rows = np.tile(idx, nt)
        tms = np.repeat(np.asarray(teams), n)
        out = self.finish(self.start(bsim, rows, tms)).reshape(nt, n, 5, 2)
        return {tm: out[i] for i, tm in enumerate(teams)}
