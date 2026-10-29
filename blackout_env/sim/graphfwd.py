"""CUDA-graph replay of a MyModel student's forward (fp16 autocast) for small, latency-bound batches.

GRPO's rollouts call ~10 different small batches (tens to hundreds of views) per decision step; on a 2080 Ti the
24x24-map net takes ~0.3 ms of GPU time but ~5-10 ms of Python/kernel-launch time per call.  A graph captured per
*exact* batch size replays the very same kernels the eager call would launch (same shapes -> same cuBLAS/cuDNN
algorithms), so the outputs are bit-identical to ``NetPolicy``'s eager fp16 path while the launch cost is gone.

* inputs: packed int16 map ``[m, H, W]``, team ``[m, 4]``, agents ``[m, 10, 12]`` (numpy) -- one host->device copy
  (a single pinned byte buffer), unpack + autocast forward + ``.float()`` inside the graph;
* output: one float32 device tensor ``[m, 5, K(+K')]`` (logits, then offsets when the net has an angle head);
* graphs share one memory pool (they are replayed strictly in sequence on the current stream), so hundreds of
  sizes cost little memory; sizes above ``max_batch`` or beyond ``max_graphs`` fall back to the eager forward;
* fine direction head (rl/finehead.py, opt-in): ``tap`` (a ``FeatTap`` on the net's angle head) + ``fine`` (a ``FineHead``)
  fold the greedy fine decode INTO the graph -- the offsets of the argmax coarse bin become the argmax fine centre, the
  output keeps its shape (net opponents / greedy eval, no extra memory); ``tap`` + ``expose_feat`` instead keeps the
  angle head's input features as a second graph output, read as ``self.feat`` after each call (the learner, which
  samples the fine index outside the graph);
* ``bucket`` > 0 pads a batch up to a multiple of ``bucket`` (fewer graphs, but the padded GEMM shape may round
  differently from the eager one -- off by default).
"""

from __future__ import annotations

import numpy as np
import torch

from .batch import unpack_graphic


class GraphForward:
    def __init__(self, own, device, angle: bool = True, max_batch: int = 2048, max_graphs: int = 400, bucket: int = 0,
                 autocast: bool = True, tap=None, fine=None, expose_feat: bool = False):
        self.tap, self.fine, self.expose_feat = tap, fine, bool(expose_feat)
        self.feat = None   # expose_feat: the last call's features [m, 5, H] (device tensor, valid until the next call of that size)
        self._side = None
        self.own, self.device, self.angle = own, torch.device(device), angle
        self.max_batch, self.max_graphs, self.bucket, self.autocast = max_batch, max_graphs, bucket, autocast
        self._ents: dict = {}
        self._pool = None
        self.n_eager = 0
        self.kq = None

    # ---- the traced function
    def _fn(self, g, t, a):
        g = unpack_graphic(g) if g.dim() == 3 else g
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.autocast):
            out = self.own(g, t, a)
        if self.angle:
            self.kq = out[0].shape[-1]   # logits width; the offsets follow
            q, off = out[0].float(), out[1].float()
            if self.expose_feat:
                self._side = self.tap.feat
            elif self.fine is not None:
                off = self.fine.greedy_offsets(q, off, self.tap.feat.float())
            return torch.cat([q, off], -1)
        return (out[0] if isinstance(out, tuple) else out).float()

    def eager(self, g, t, a) -> torch.Tensor:
        """The un-graphed path (also what a graph is checked against)."""
        with torch.inference_mode():
            return self._fn(torch.from_numpy(np.ascontiguousarray(g)).to(self.device),
                            torch.from_numpy(np.ascontiguousarray(t)).to(self.device),
                            torch.from_numpy(np.ascontiguousarray(a)).to(self.device))

    def _eager_feat(self, g, t, a) -> torch.Tensor:
        out = self.eager(g, t, a)
        if self.expose_feat:
            self.feat = self._side
        return out

    def _size(self, m: int) -> int:
        return m if self.bucket <= 0 else -(-m // self.bucket) * self.bucket

    def _capture(self, g, t, a, size: int):
        parts = [np.ascontiguousarray(x[:1]) for x in (g, t, a)]
        nbytes = [p.reshape(1, -1).view(np.uint8).shape[1] for p in parts]
        cols = np.cumsum([0] + nbytes)
        host_in = torch.empty((size, int(cols[-1])), dtype=torch.uint8, pin_memory=True)
        host_np = host_in.numpy()
        for k, p in enumerate(parts):
            host_np[:, cols[k]:cols[k + 1]] = p.reshape(1, -1).view(np.uint8)   # pad rows: a real input
        dev_in = host_in.to(self.device)

        def views():
            return [dev_in[:, cols[k]:cols[k + 1]].view(torch.from_numpy(x[:0]).dtype).reshape(size, *x.shape[1:]).contiguous()
                    for k, x in enumerate((g, t, a))]

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):   # warm-up on a side stream (cuBLAS workspaces, lazy tables) before capture
                self._fn(*views())
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: another thread (PPO async_collect's learner update) may launch work on other streams/devices during capture
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            out = self._fn(*views())
        done = torch.cuda.Event()
        return {"graph": graph, "host": host_in, "host_np": host_np, "dev": dev_in, "out": out, "cols": cols, "ev": done,
                "feat": self._side}

    def __call__(self, g, t, a) -> torch.Tensor:
        """numpy ``g`` (packed ``[m, H, W]``), ``t``, ``a`` -> float32 device tensor of the m rows (valid until
        the next call with the same size: consume or copy it before)."""
        m = len(g)
        size = self._size(m)
        if self.device.type != "cuda" or size > self.max_batch:
            self.n_eager += 1
            return self._eager_feat(g, t, a)
        key = (g.ndim, size)
        ent = self._ents.get(key)
        if ent is None:
            if len(self._ents) >= self.max_graphs:
                self.n_eager += 1
                return self._eager_feat(g, t, a)
            ent = self._ents[key] = self._capture(g, t, a, size)
        ent["ev"].synchronize()   # the previous replay of this entry has consumed the pinned buffer
        hn, cols = ent["host_np"], ent["cols"]
        for k, x in enumerate((g, t, a)):
            hn[:m, cols[k]:cols[k + 1]] = np.ascontiguousarray(x).reshape(m, -1).view(np.uint8)
        ent["dev"][:m].copy_(ent["host"][:m], non_blocking=True)
        ent["graph"].replay()
        ent["ev"].record()
        if self.expose_feat:
            self.feat = ent["feat"][:m]
        return ent["out"][:m]
