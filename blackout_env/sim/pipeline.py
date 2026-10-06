"""B1: pipelined search rollouts -- ``BatchRollout`` for ``Searcher`` with the GPU and the CPU overlapped
(docs/policy_v2_design_ko.md "B1").

``BatchRollout`` + ``ModNetPolicy`` (+ ``ExternalBasePolicy``) do, per decision, render every live game ->
one forward (``NetPolicy.forward`` splits it into ``max_batch``-view chunks, each followed by a blocking
``.cpu()``) -> numpy branch modifications and ``decode_moves`` over every view -> step.  With one CPU
thread the GPU idles while the CPU renders / decodes / steps, and the CPU idles while each chunk runs.

``PipelinedRollout`` plays the same games with the same arithmetic, reordered:

* the maps are rendered packed (int16, ``render_packed``) into pinned host memory and copied to the
  device asynchronously; the device unpacks them (``unpack_graphic``: the same float32 values);
* the live games are cut into the *same* chunks the old forward used (``max_batch`` views = ``max_batch
  // 2`` games, both teams per game), and chunk c's forward is queued before chunk c + 1 is rendered,
  so the GPU works on chunk c while the CPU renders c + 1 and steps c - 1;
* the branch modifications (alt / pair masks, sampling noise) and the arg max / offset pick of
  ``decode_moves`` run on the device (float64 where numpy computed in float64), only the chosen bins and
  their offsets come back; the trigonometry, ``stop`` / ``rot`` and the counters stay on the CPU with
  numpy's expressions;
* an external base policy (§32 overlay) is fed the same rows as before (every window game of a team,
  gathered on the device from the chunks' inputs) in the same ``max_batch`` chunks.

Every forward therefore sees exactly the batch it saw in ``BatchRollout`` -- same rows, same order, same
size -- so the results are bit-identical (tests/test_sim_pipeline.py: CPU and, when present, CUDA).
``chunk_games`` < ``max_batch // 2`` cuts finer chunks (more overlap when few roots are live) at the
price of other forward batch sizes (GPU kernels may round differently: float-noise, not bit-exact).

Unsupported (falls back to ``BatchRollout``, packed): extra opponent nets in use (``onet`` > 0), a base
policy that is not a ``NetPolicy``, CUDA-graph or ``per_sample`` nets, action repeat.
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import core
from .batch import (F32, N_AGENT_COLS, BatchRollout, BatchSim, NetPolicy, RolloutResult, mirror_order,
                    unpack_graphic)
from .mapdata import HUNTER, H, W
from .search import (K_ALT, K_ALTM, K_BASE, K_ROT, K_SAMP, K_STOP, NOISE_SEEDS, NOISE_STEPS, ExternalBasePolicy,
                     ModNetPolicy, _gumbel_table, overlay_base)


def moves_from_bins(j: np.ndarray, o: np.ndarray | None, width: int) -> np.ndarray:
    """``decode_moves``' tail: j int [m, 5] world-order arg max (``k`` = STAND for an odd width), o float32
    [m, 5] the chosen bin's world offset (None: bin centres) -> float32 [m, 5, 2] moves."""
    k = width - (width % 2)
    stand = j == k
    j = np.where(stand, 0, j)
    if o is None:
        from blackout_env.model.direction_bins import bin_vectors

        vec = bin_vectors(k)[j]
    else:
        ang = 2 * math.pi * j / k + o.astype(np.float64)
        vec = np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)
    vec = np.where(stand[..., None], F32(0), vec)
    return np.clip(vec, F32(-1), F32(1)).astype(F32)


class _Dev:
    """Device helpers of one torch device: pinned host buffers (CUDA), index tensors, the decode."""

    def __init__(self, device):
        import torch

        self.torch, self.device = torch, torch.device(device)
        self.cuda = self.device.type == "cuda"
        self._orders: dict = {}

    def host(self, shape, dtype):
        """A host buffer (pinned on CUDA) as (torch tensor, numpy view)."""
        t = self.torch.empty(shape, dtype=dtype, pin_memory=self.cuda)
        return t, t.numpy()

    def put(self, x: np.ndarray):
        """Host array -> device tensor.  On CUDA through a (cached) pinned block: a copy from pageable memory
        waits for the stream's queued work (the forward in flight) before it starts."""
        t = self.torch.from_numpy(np.ascontiguousarray(x))
        if self.cuda:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def order(self, width: int):
        if width not in self._orders:
            k = width - (width % 2)
            o = mirror_order(k)
            if width != k:
                o = np.r_[o, k]
            self._orders[width] = (self.put(o.astype(np.int64)), self.put(o[:k].astype(np.int64)))
        return self._orders[width]

    def decode(self, q, off, team_b):
        """Device ``decode_moves`` up to the trigonometry: q [m, 5, W], off [m, 5, K] | None, team_b bool [m]
        (device) -> j [m, 5] int64, o [m, 5] float32 | None (the chosen bin's world-frame offset)."""
        torch = self.torch
        width = q.shape[-1]
        k = width - (width % 2)
        order, order_k = self.order(width)
        tb = team_b[:, None, None]
        qw = torch.where(tb, q[..., order], q)
        j = qw.argmax(-1)              # first max, as numpy's
        if off is None:
            return j, None
        j0 = torch.where(j == k, torch.zeros_like(j), j)
        offw = torch.where(tb, -off[..., order_k], off)
        return j, offw.gather(-1, j0[..., None])[..., 0]


class _Pending:
    """Device results of one forward chunk on their way to the host (pinned, non-blocking) + an event."""

    def __init__(self, dev: _Dev, j, o):
        torch = dev.torch
        self.dev = dev
        if dev.cuda:
            self.hj = torch.empty(j.shape, dtype=j.dtype, pin_memory=True)
            self.hj.copy_(j, non_blocking=True)
            self.ho = None
            if o is not None:
                self.ho = torch.empty(o.shape, dtype=o.dtype, pin_memory=True)
                self.ho.copy_(o, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
        else:
            self.hj, self.ho, self.event = j.cpu(), (None if o is None else o.cpu()), None

    def result(self):
        if self.event is not None:
            self.event.synchronize()
        return self.hj.numpy(), (None if self.ho is None else self.ho.numpy())


def _net_forward(net: NetPolicy, g, t, a):
    """``NetPolicy.forward``'s device part (one chunk, device tensors): float32 logits, offsets | None."""
    torch = net.torch
    with torch.inference_mode():
        if net.fp16:
            with torch.autocast(device_type=net.device.type, dtype=torch.float16):
                out = net._own(g, t, a)
        else:
            out = net._own(g, t, a)
        q, off = out if net.angle else (out, None)
        return q.float(), (None if off is None else off.float())


class PipelinedRollout:
    """Drop-in for ``BatchRollout(policy, policy, repeat=1)`` inside ``Searcher`` (``policy`` a
    ``ModNetPolicy`` or an ``ExternalBasePolicy`` over one): the same ``run`` / ``timing`` interface."""

    def __init__(self, policy, chunk_games: int | None = None, parallel: bool | None = None, rolling: bool = True,
                 graphs: bool = False):
        """``rolling``: queue a chunk's next decision as soon as it has stepped (across the step boundary).
        ``graphs`` (CUDA): the rollout net's forward of a full chunk as a CUDA graph (fewer launches on the
        CPU; the same kernels on the same batch)."""
        if isinstance(policy, ExternalBasePolicy):
            self.mod, self.ext = policy.mod, policy
        else:
            assert isinstance(policy, ModNetPolicy)
            self.mod, self.ext = policy, None
        self.policy = policy
        self.net: NetPolicy = self.mod.net
        self.parallel = parallel
        self.rolling = bool(rolling)
        self.graphs = bool(graphs) and str(self.net.device).startswith("cuda")
        self._graphs: dict = {}
        self.chunk_games = int(chunk_games) if chunk_games else max(1, self.net.max_batch // 2)
        self.fallback = BatchRollout(policy, policy, repeat=1, packed=True, parallel=parallel)
        self.dev = _Dev(self.net.device)
        self._noise = None
        self._buf = None
        self.timing = {}

    # ----------------------------------------------------------------------------- support
    def supported(self) -> bool:
        net = self.net
        if net.cuda_graphs or net.max_batch < 2:
            return False
        if self.mod.opp_nets and len(self.mod.onet) and (self.mod.onet > 0).any():
            return False
        if self.ext is not None:
            b = self.ext.base
            if type(b) is not NetPolicy or b.cuda_graphs or b.max_batch < 1 or b.device != net.device:
                return False
        return True

    def _buffers(self, n):
        if self._buf is None or self._buf[0][1].shape[0] < n:
            torch, d = self.dev.torch, self.dev
            self._buf = (d.host((n, 2, H, W), torch.int16), d.host((n, 2, 4), torch.float32),
                         d.host((n, 2, 10, N_AGENT_COLS), torch.float32),
                         torch.empty((n, 2, H, W), dtype=torch.int16, device=d.device),
                         torch.empty((n, 2, 4), dtype=torch.float32, device=d.device),
                         torch.empty((n, 2, 10, N_AGENT_COLS), dtype=torch.float32, device=d.device))
        return self._buf

    def _noise_dev(self, k):
        if self._noise is None or self._noise.shape[-1] != k:
            mod = self.mod
            if mod.noise is None or mod.noise.shape[-1] != k:
                mod.noise = _gumbel_table(k)
            self._noise = self.dev.put(mod.noise)
        return self._noise

    # ----------------------------------------------------------------------------- one chunk
    def _rows(self, idx):
        """``ModNetPolicy.moves``' per-(game, team) branch selection for games ``idx`` (both teams), CPU."""
        mod = self.mod
        n = len(idx)
        game_team = np.asarray([0, 1])[None, :].repeat(n, 0)
        tm = mod.team[idx][:, None]
        own = tm == game_team
        opp = (tm >= 0) & ~own
        act_own = own & (mod.left[idx] > 0)[:, None]
        act_opp = opp & (mod.oleft[idx] > 0)[:, None]
        kind = np.where(act_own, mod.kind[idx][:, None], np.where(act_opp, mod.okind[idx][:, None], K_BASE))
        pick = lambda a, b: np.where(own, a[idx][:, None], b[idx][:, None])  # noqa: E731
        return kind, pick(mod.unit, mod.ounit), pick(mod.param, mod.oparam), pick(mod.seed, mod.oseed)

    def _modify(self, q, idx, kind, unit, param, seed):
        """``ModNetPolicy.moves``' logit modifications on the device: q [n, 2, 5, W] float32 (in place)."""
        torch, d = self.dev.torch, self.dev
        k = q.shape[-1]
        rows_n, rows_s = np.nonzero(kind == K_ALT)
        if len(rows_n):
            u = unit[rows_n, rows_s]
            rank = np.rint(param[rows_n, rows_s]).astype(np.int64)
            for r in range(int(rank.max()) + 1):
                m = rank >= r
                rn, rs, ru = d.put(rows_n[m]), d.put(rows_s[m]), d.put(u[m])
                top = q[rn, rs, ru].argmax(-1)
                for dl in (-1, 0, 1):
                    q[rn, rs, ru, (top + dl) % k] = -math.inf
        rows_n, rows_s = np.nonzero(kind == K_ALTM)
        if len(rows_n):
            mask = unit[rows_n, rows_s]
            for u in range(5):
                m = ((mask >> u) & 1) == 1
                if m.any():
                    rn, rs = d.put(rows_n[m]), d.put(rows_s[m])
                    top = q[rn, rs, u].argmax(-1)
                    for dl in (-1, 0, 1):
                        q[rn, rs, u, (top + dl) % k] = -math.inf
        rows_n, rows_s = np.nonzero(kind == K_SAMP)
        if len(rows_n):
            noise = self._noise_dev(k)
            b = idx[rows_n]
            temp = d.put(np.maximum(param[rows_n, rows_s], 1e-3))[:, None, None]
            si = d.put(seed[rows_n, rows_s] % NOISE_SEEDS)
            ti = d.put(self.mod.t[b] % NOISE_STEPS)
            rn, rs = d.put(rows_n), d.put(rows_s)
            # numpy: float32 q / float64 temp + float32 noise -> float64, stored back as float32
            q[rn, rs] = (q[rn, rs].double() / temp + noise[si, ti].double()).float()
        return q

    def _submit_chunk(self, bsim, idx, lo, bufs):
        """Render games ``idx`` into rows lo.. of the buffers, queue their forward + device decode; CPU-side
        branch info for the finish.  Updates the ``ModNetPolicy`` counters (as ``moves`` does)."""
        torch, d = self.dev.torch, self.dev
        (hv_t, hv), (ht_t, ht), (ha_t, ha), dv, dt, da = bufs
        n = len(idx)
        hi = lo + n
        bsim.render_packed(idx, hv[lo:hi], ht[lo:hi], ha[lo:hi], parallel=self.parallel)
        dv[lo:hi].copy_(hv_t[lo:hi], non_blocking=d.cuda)
        dt[lo:hi].copy_(ht_t[lo:hi], non_blocking=d.cuda)
        da[lo:hi].copy_(ha_t[lo:hi], non_blocking=d.cuda)
        g = dv[lo:hi].reshape(n * 2, H, W)
        t = dt[lo:hi].reshape(n * 2, 4)
        a = da[lo:hi].reshape(n * 2, 10, N_AGENT_COLS)
        if self.graphs and n == self.chunk_games:
            q, off = self._graph_forward(g, t, a)
        else:
            q, off = _net_forward(self.net, g, t, a)
        kind, unit, param, seed = self._rows(idx)
        with torch.inference_mode():
            q = self._modify(q.reshape(n, 2, 5, -1), idx, kind, unit, param, seed).reshape(n * 2, 5, -1)
            team_b = self._team_b(n)
            j, o = d.decode(q, off, team_b)
        mod = self.mod
        mod.left[idx] = np.maximum(mod.left[idx] - 1, 0)
        mod.oleft[idx] = np.maximum(mod.oleft[idx] - 1, 0)
        mod.t[idx] += 1
        return _Pending(d, j, o), (kind, unit, param, q.shape[-1])

    def _graph_forward(self, g, t, a):
        """The rollout net on a full chunk through a CUDA graph captured for that size; copies of the outputs
        (the branch modifications write into q, the next replay overwrites the graph's outputs)."""
        torch = self.dev.torch
        key = tuple(g.shape)
        ent = self._graphs.get(key)
        if ent is None:
            sg, st, sa = g.clone(), t.clone(), a.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    _net_forward(self.net, sg, st, sa)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, off = _net_forward(self.net, sg, st, sa)
            ent = self._graphs[key] = (graph, sg, st, sa, q, off)
        graph, sg, st, sa, q, off = ent
        sg.copy_(g)
        st.copy_(t)
        sa.copy_(a)
        graph.replay()
        return q.clone(), (None if off is None else off.clone())

    def _team_b(self, n):
        key = ("tb", n)
        tb = self.dev._orders.get(key)
        if tb is None:
            tb = self.dev._orders[key] = self.dev.put(np.tile(np.array([False, True]), n))
        return tb

    def _finish_chunk(self, pending, info):
        """[n, 2, 5, 2] moves of a chunk (``ModNetPolicy.moves``' tail: trigonometry, stop, rot)."""
        kind, unit, param, width = info
        w0 = time.perf_counter()
        j, o = pending.result()
        self.timing["wait"] += time.perf_counter() - w0
        n = len(kind)
        mv = moves_from_bins(j, o, width).reshape(n, 2, 5, 2)
        rows_n, rows_s = np.nonzero(kind == K_STOP)
        if len(rows_n):
            mv[rows_n, rows_s, unit[rows_n, rows_s]] = 0.0
        rows_n, rows_s = np.nonzero(kind == K_ROT)
        if len(rows_n):
            u = unit[rows_n, rows_s]
            ang = param[rows_n, rows_s]
            v = mv[rows_n, rows_s, u].astype(np.float64)
            c, s = np.cos(ang), np.sin(ang)
            rot = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)
            mv[rows_n, rows_s, u] = np.clip(rot, -1, 1).astype(F32)
        return mv

    def _submit_base(self, rows, tm, bufs):
        """The external base net on views (game rows ``rows`` of this step's act order, team ``tm``), in
        ``max_batch`` chunks as ``NetPolicy.forward``; device gather from the chunks' inputs."""
        torch, d = self.dev.torch, self.dev
        base: NetPolicy = self.ext.base
        dv, dt, da = bufs[3:]
        ri = d.put(rows)
        out = []
        for s in range(0, len(rows), base.max_batch):
            r = ri[s:s + base.max_batch]
            q, off = _net_forward(base, dv[r, tm], dt[r, tm], da[r, tm])
            with torch.inference_mode():
                tb = torch.full((len(r),), tm == 1, dtype=torch.bool, device=d.device)
                j, o = d.decode(q, off, tb)
            out.append((_Pending(d, j, o), q.shape[-1]))
        return out

    @staticmethod
    def _finish_base(parts):
        mv = [moves_from_bins(*p.result(), width) for p, width in parts]
        return np.concatenate(mv) if len(mv) > 1 else mv[0]

    # ----------------------------------------------------------------------------- the loop
    def run(self, roots, horizon: int, reseeds=None, record: bool = False, bsim: BatchSim | None = None,
            stop_when_exhausted: bool = False, trace_steps=()) -> RolloutResult:
        if record or stop_when_exhausted or trace_steps or not self.supported():
            res = self.fallback.run(roots, horizon, reseeds=reseeds, record=record, bsim=bsim,
                                    stop_when_exhausted=stop_when_exhausted, trace_steps=trace_steps)
            self.timing = self.fallback.timing
            return res
        if bsim is None:
            bsim = BatchSim(roots)
        elif roots is not None:
            bsim.load(roots)
        if reseeds is not None:
            bsim.reseed(reseeds)
        n = bsim.n
        self.policy.reset(n)
        bufs = self._buffers(n)
        moves = np.zeros((n, 10, 2), F32)
        steps = np.zeros(n, np.int64)
        first = np.full((n, 2), -1, np.int64)
        for team in (0, 1):
            first[(bsim.ui[:, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1), team] = 0
        clock = time.perf_counter
        tm_ = self.timing = {"render": 0.0, "policy": 0.0, "step": 0.0, "calls": 0, "game_decisions": 0, "wait": 0.0,
                            "wait_base": 0.0, "window_s": 0.0}
        mod, cg = self.mod, self.chunk_games
        pre = []  # chunks of the coming step already queued (rolled over from the last one)
        for t in range(horizon):
            live = np.nonzero(~bsim.over)[0]
            if not len(live):
                break
            c0 = clock()
            m = len(live)
            if self.ext is not None:  # ``ExternalBasePolicy.moves``' snapshot before the counters move
                win = (mod.team[live] >= 0) & (mod.left[live] > 0)
                e_kind, e_unit, e_team = mod.kind[live].copy(), mod.unit[live].copy(), mod.team[live].copy()
            chunks = pre
            done = sum(len(c[1]) for c in chunks)
            assert all(np.array_equal(idx, live[lo:lo + len(idx)]) for lo, idx, _, _ in chunks)
            for lo in range(done, m, cg):
                idx = live[lo:lo + cg]
                chunks.append((lo, idx, *self._submit_chunk(bsim, idx, lo, bufs)))
            pre = []
            base_parts = []
            if self.ext is not None:
                for team in (0, 1):
                    rows = np.nonzero(win & (e_team == team))[0]
                    if len(rows):
                        assert done == 0, "a rolled-over step has no window rows"
                        base_parts.append((team, rows, self._submit_base(rows, team, bufs)))
            c1 = clock()
            tm_["render"] += c1 - c0  # render + queueing the device work
            if not base_parts:
                # roll over: once chunk c has stepped, queue its next decision right away -- the same games
                # make chunk c of the next step when no game of chunks 0..c ended (games are kept in index
                # order) and no window rows remain (the base overlay needs every chunk's inputs at once)
                roll = (self.rolling and t + 1 < horizon
                        and (self.ext is None or not ((mod.team[live] >= 0) & (mod.left[live] > 0)).any()))
                for lo, idx, pend, info in chunks:
                    w0 = clock()
                    mv = self._finish_chunk(pend, info)
                    w1 = clock()
                    moves[idx, :5], moves[idx, 5:] = mv[:, 0], mv[:, 1]
                    bsim.step(moves, idx, parallel=self.parallel)
                    w2 = clock()
                    tm_["policy"] += w1 - w0
                    tm_["step"] += w2 - w1
                    if roll and not bsim.over[idx].any():
                        pre.append((lo, idx, *self._submit_chunk(bsim, idx, lo, bufs)))
                        tm_["render"] += clock() - w2
                    else:
                        roll = False
            else:
                w0 = clock()
                out = np.zeros((m, 2, 5, 2), F32)
                for lo, idx, pend, info in chunks:
                    out[lo:lo + len(idx)] = self._finish_chunk(pend, info)
                for team, rows, parts in base_parts:
                    b0 = clock()
                    ext = self._finish_base(parts)
                    tm_["wait_base"] += clock() - b0
                    out[rows, team] = overlay_base(out[rows, team], ext, e_kind[rows], e_unit[rows], self.ext.overlay)
                moves[live, :5], moves[live, 5:] = out[:, 0], out[:, 1]
                w1 = clock()
                bsim.step(moves, live, parallel=self.parallel)
                tm_["policy"] += w1 - w0
                tm_["step"] += clock() - w1
                tm_["window_s"] += clock() - c0
            tm_["calls"] += m
            tm_["game_decisions"] += m
            steps[live] += 1
            for team in (0, 1):
                hit = (first[live, team] < 0) & (bsim.ui[live, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1)
                first[live[hit], team] = t + 1
        assert not pre
        over = bsim.over.copy()
        winner = np.where(over, bsim.g[:, core.G_WINNER], -1)
        return RolloutResult(bsim.scores, over, winner, steps, bsim.ticks, first, None, bsim)
