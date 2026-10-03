"""Batched simulator rollouts for the search teacher (S3): N independent games in lockstep.

    from blackout_env.sim.batch import BatchRollout, NetPolicy
    p11 = NetPolicy.from_checkpoint("checkpoints/distill/frozen/p11_v22.pt", device="mps", fp16=True)
    roll = BatchRollout(p11, p11, repeat=1)
    res = roll.run(roots, horizon=125, reseeds=seeds)     # roots: list[SimState] (copied, not changed)
    res.lead(team=0), res.scores, res.over, res.states()

Pieces (docs/policy_v2_design_ko.md "S3 준비 — 배치 롤아웃"):

* ``BatchSim`` -- every ``SimState`` array stacked on a leading game axis (layouts too: every map
  has the same region count), stepped by ``core.step_kernel`` on per-game views inside one numba
  call (``parallel=True`` = prange over games).  Bit-exact with stepping each ``SimState``.
* ``render_inputs`` -- the policy network's input tensors straight from the state, both teams'
  canonical frames (team_frame.py: team B anti-transposed, units rolled, x/y swapped), into
  preallocated float32 arrays.  Bit-exact with ``sim.obs()`` -> ``MyPolicy``'s tensors
  (tests/test_sim_batch.py), without the env preprocessor or dicts.
* ``decode_moves`` -- a logits(+angle) head's output -> world-frame moves, exactly as
  ``MyPolicy.act`` (argmax over the world-ordered row, bin centre + offset, team B mirrored).
* policies: ``NetPolicy`` (any stateless MyModel student, e.g. P11; CPU/MPS, fp32/fp16),
  ``SmallNetPolicy`` (the small rollout net, model in rollout_net.py) and ``HeuristicPolicy``
  (per-game instances on ``obs_from_raw``: the slow reference/fallback).
* ``BatchRollout`` -- the loop: render, both teams' policies (one merged forward when it is the
  same net), optional action repeat (policy every ``repeat`` decisions once the episode is past
  ``opening_ticks``; the opening race needs every tick), step; games that end drop out.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from . import core
from .mapdata import HUNTER, H, W
from .state import SimState, obs_from_raw

F32 = np.float32
N_CH = 13          # MyObsPreprocessor graphic channels (8 base + battery + 4 items)
N_AGENT_COLS = 12  # pos 2 + team 1 + item one-hot 6 + class one-hot 3
OPENING_TICKS = 500  # 10 s of world updates (50 per second): every decision there calls the policy
_SIM_ARRAYS = SimState._ARRAYS


# ---------------------------------------------------------------------------------- kernels

@njit(cache=True, inline='always')
def _game_S(Sb, b):
    """Game b's state tuple ``S`` as views into the stacked arrays."""
    return (Sb[0][b], Sb[1][b], Sb[2][b], Sb[3][b], Sb[4][b], Sb[5][b], Sb[6][b], Sb[7][b], Sb[8][b],
            Sb[9][b], Sb[10][b], Sb[11][b], Sb[12][b])


@njit(cache=True, inline='always')
def _game(Sb, Mb, b):
    S = _game_S(Sb, b)
    M = (Mb[0][b], Mb[1][b], Mb[2][b], Mb[3][b], Mb[4][b], Mb[5][b], Mb[6][b], Mb[7][b], Mb[8][b],
         Mb[9][b], Mb[10][b])
    return S, M


@njit(cache=True)
def batch_step_seq(Sb, Mb, moves, order, idx):
    for j in range(idx.shape[0]):
        b = idx[j]
        S, M = _game(Sb, Mb, b)
        core.step_kernel(S, M, moves[b], order)


@njit(cache=True, parallel=True)
def batch_step_par(Sb, Mb, moves, order, idx):
    for j in prange(idx.shape[0]):
        b = idx[j]
        S, M = _game(Sb, Mb, b)
        core.step_kernel(S, M, moves[b], order)


@njit(cache=True, inline='always')
def _render_one(S, bg, graphic, team, agents, raw, packed):
    """One game's network inputs for both teams: graphic [2, 13, H, W], team [2, 4],
    agents [2, 10, 12] (index 0 = team A's view, 1 = team B's canonical view)."""
    core.render_raw_state(S, raw)
    core.render_raw_map(S, bg, packed)
    graphic[:, :, :, :] = 0.0
    f15 = F32(15.0)
    for r in range(H):
        for c in range(W):
            p = np.int64(packed[r, c])
            base = p & 7
            bat = (p >> 3) & 15
            item = (p >> 7) & 7
            graphic[0, base, r, c] = 1.0
            # team B: ally/enemy channels swapped, then the anti-transpose (r, c) -> (W-1-c, H-1-r)
            bb = base
            if base == 4:
                bb = 5
            elif base == 5:
                bb = 4
            elif base == 6:
                bb = 7
            elif base == 7:
                bb = 6
            rb, cb = W - 1 - c, H - 1 - r
            graphic[1, bb, rb, cb] = 1.0
            v = F32(bat) / f15
            graphic[0, 8, r, c] = v
            graphic[1, 8, rb, cb] = v
            if 1 <= item <= 4:
                graphic[0, 8 + item, r, c] = 1.0
                graphic[1, 8 + item, rb, cb] = 1.0
    _render_units(raw, team, agents)


@njit(cache=True, inline='always')
def _render_units(raw, team, agents):
    """agents [2, 10, 12] and team [2, 4] of both teams' views from the raw broadcast."""
    f15 = F32(15.0)
    agents[:, :, :] = 0.0
    for u in range(10):
        x, y, held, cls = raw[4 * u], raw[4 * u + 1], raw[4 * u + 2], raw[4 * u + 3]
        sign = F32(1.0) if u < 5 else F32(-1.0)
        kb = (u + 5) % 10  # team B canonical row of unit u
        agents[0, u, 0], agents[0, u, 1], agents[0, u, 2] = x, y, sign
        agents[1, kb, 0], agents[1, kb, 1], agents[1, kb, 2] = y, x, -sign
        if held < 0:
            v = (-held) / f15
            agents[0, u, 4] = v
            agents[1, kb, 4] = v
        else:
            slot = int(np.rint(held))
            if 0 <= slot < 6:
                agents[0, u, 3 + slot] = 1.0
                agents[1, kb, 3 + slot] = 1.0
        k = int(np.rint(cls))
        if 0 <= k < 3:
            agents[0, u, 9 + k] = 1.0
            agents[1, kb, 9 + k] = 1.0
    team[0, 0], team[0, 1], team[0, 2], team[0, 3] = raw[40], raw[41], raw[42], raw[43]
    team[1, 0], team[1, 1], team[1, 2], team[1, 3] = raw[41], raw[40], raw[42], raw[43]


@njit(cache=True, inline='always')
def _render_packed_one(S, bg, views, team, agents, raw, packed):
    """As ``_render_one`` with the map left packed (int16 [2, H, W], ``unpack_graphic`` layout:
    base | battery << 3 | item << 7, team B's view ally/enemy-swapped and anti-transposed)."""
    core.render_raw_state(S, raw)
    core.render_raw_map(S, bg, packed)
    for r in range(H):
        for c in range(W):
            p = np.int64(packed[r, c])
            base = p & 7
            bb = base
            if base == 4:
                bb = 5
            elif base == 5:
                bb = 4
            elif base == 6:
                bb = 7
            elif base == 7:
                bb = 6
            views[0, r, c] = p
            views[1, W - 1 - c, H - 1 - r] = (p - base) + bb
    _render_units(raw, team, agents)


@njit(cache=True, parallel=True)
def batch_render_packed_par(Sb, bg, idx, views, team, agents):
    for j in prange(idx.shape[0]):
        b = idx[j]
        S = _game_S(Sb, b)
        raw = np.empty(44, dtype=np.float32)
        packed = np.empty((H, W), dtype=np.uint16)
        _render_packed_one(S, bg[b], views[j], team[j], agents[j], raw, packed)


@njit(cache=True)
def batch_render_packed_seq(Sb, bg, idx, views, team, agents):
    raw = np.empty(44, dtype=np.float32)
    packed = np.empty((H, W), dtype=np.uint16)
    for j in range(idx.shape[0]):
        b = idx[j]
        S = _game_S(Sb, b)
        _render_packed_one(S, bg[b], views[j], team[j], agents[j], raw, packed)


def unpack_graphic(p):
    """torch int [m, H, W] packed views -> float32 [m, 13, H, W], exactly ``_render_one``'s graphic."""
    import torch

    p = p.long()
    base, bat, item = p & 7, (p >> 3) & 15, (p >> 7) & 7
    ar = torch.arange(13, device=p.device).view(1, 13, 1, 1)
    g = (base.unsqueeze(1) == ar).float()                       # channels 0-7 (8-12 never match base)
    g = g + ((item.unsqueeze(1) + 8 == ar) & (ar >= 9)).float()  # items 1-4 -> channels 9-12
    g[:, 8] = bat.float() / 15.0
    return g


@njit(cache=True, parallel=True)
def batch_render_par(Sb, bg, idx, graphic, team, agents):
    """Inputs of games ``idx`` into rows 0..len(idx)-1 of graphic [n, 2, 13, H, W], team [n, 2, 4],
    agents [n, 2, 10, 12]."""
    for j in prange(idx.shape[0]):
        b = idx[j]
        S = _game_S(Sb, b)
        raw = np.empty(44, dtype=np.float32)
        packed = np.empty((H, W), dtype=np.uint16)
        _render_one(S, bg[b], graphic[j], team[j], agents[j], raw, packed)


@njit(cache=True)
def batch_render_seq(Sb, bg, idx, graphic, team, agents):
    raw = np.empty(44, dtype=np.float32)
    packed = np.empty((H, W), dtype=np.uint16)
    for j in range(idx.shape[0]):
        b = idx[j]
        S = _game_S(Sb, b)
        _render_one(S, bg[b], graphic[j], team[j], agents[j], raw, packed)


PARALLEL_MIN = 16  # below this many games numba's thread pool costs more than it saves


# ---------------------------------------------------------------------------------- batched state

class BatchSim:
    """N games' ``SimState`` arrays stacked on axis 0 (copies of the given states)."""

    def __init__(self, sims: list[SimState]):
        self.n = len(sims)
        self.order = np.asarray(sims[0].order, dtype=np.int64)
        for s in sims:
            if not np.array_equal(s.order, self.order):
                raise ValueError("every state must share the AgentAct order")
        for name in _SIM_ARRAYS:
            setattr(self, name, np.ascontiguousarray(np.stack([getattr(s, name) for s in sims])))
        self.Mb = tuple(np.ascontiguousarray(np.stack([s.M[k] for s in sims])) for k in range(len(sims[0].M)))
        self.bg = np.ascontiguousarray(np.stack([s.bg for s in sims]))
        self.layouts = [s.layout for s in sims]
        self.all = np.arange(self.n, dtype=np.int64)

    @property
    def Sb(self):
        return tuple(getattr(self, name) for name in _SIM_ARRAYS)

    def load(self, sims: list[SimState]) -> None:
        """Overwrite every game with ``sims`` (same count), no allocation."""
        assert len(sims) == self.n
        for b, s in enumerate(sims):
            for name in _SIM_ARRAYS:
                getattr(self, name)[b] = getattr(s, name)
            for k in range(len(self.Mb)):
                self.Mb[k][b] = s.M[k]
            self.bg[b] = s.bg
            self.layouts[b] = s.layout

    def reseed(self, seeds) -> None:
        """Fresh RNG per game for sampled special spawns (``SimState.reseed``)."""
        from .unity_random import init_state

        for b, k in enumerate(seeds):
            self.rng[b] = init_state(int(k))
            self.g[b, core.G_SPAWN_MODE] = core.SPAWN_RNG

    def state(self, b: int) -> SimState:
        """Game ``b`` as a standalone ``SimState`` (a copy)."""
        lay = self.layouts[b]
        sim = object.__new__(SimState)
        sim.layout, sim.order, sim.bg = lay, self.order, self.bg[b].copy()
        sim.M = (lay.tile, lay.region, lay.reg_kind, lay.reg_team, lay.reg_in, lay.reg_out,
                 lay.reg_locked, lay.reg_tiles, lay.reg_ntiles, lay.walk, lay.half)
        for name in _SIM_ARRAYS:
            setattr(sim, name, getattr(self, name)[b].copy())
        return sim

    @property
    def over(self) -> np.ndarray:
        return self.g[:, core.G_OVER] != 0

    @property
    def scores(self) -> np.ndarray:
        return self.g[:, [core.G_SCORE_A, core.G_SCORE_B]].copy()

    @property
    def ticks(self) -> np.ndarray:
        return self.g[:, core.G_TICK].copy()

    def step(self, moves: np.ndarray, idx: np.ndarray | None = None, parallel: bool | None = None) -> None:
        """One decision for games ``idx`` (default all); ``moves`` float32 [N, 10, 2] indexed by game.
        ``parallel`` None = prange over games from PARALLEL_MIN games on."""
        idx = self.all if idx is None else np.ascontiguousarray(idx, dtype=np.int64)
        if parallel is None:
            parallel = len(idx) >= PARALLEL_MIN
        fn = batch_step_par if parallel else batch_step_seq
        fn(self.Sb, self.Mb, np.ascontiguousarray(moves, dtype=F32), self.order, idx)

    def render(self, idx, graphic, team, agents, parallel: bool | None = None) -> None:
        """Network inputs of games ``idx`` into rows 0..len(idx)-1 (see ``_render_one``)."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        if parallel is None:
            parallel = len(idx) >= PARALLEL_MIN
        fn = batch_render_par if parallel else batch_render_seq
        fn(self.Sb, self.bg, idx, graphic, team, agents)

    def render_packed(self, idx, views, team, agents, parallel: bool | None = None) -> None:
        """As ``render`` with the map packed: views int16 [n, 2, H, W] (``unpack_graphic``)."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        if parallel is None:
            parallel = len(idx) >= PARALLEL_MIN
        fn = batch_render_packed_par if parallel else batch_render_packed_seq
        fn(self.Sb, self.bg, idx, views, team, agents)

    def raw(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Game ``b``'s raw broadcast (44 floats) and packed map, as ``SimState.raw_state/raw_map``."""
        S = tuple(getattr(self, name)[b] for name in _SIM_ARRAYS)
        rs = np.zeros(44, dtype=np.float32)
        core.render_raw_state(S, rs)
        rm = np.zeros((H, W), dtype=np.uint16)
        core.render_raw_map(S, self.bg[b], rm)
        return rs, rm


def batteries_in_play(bsim: BatchSim, idx: np.ndarray) -> np.ndarray:
    """[len(idx)] battery points on the map (floor and storages) plus carried, as
    ``dead_segments.batteries_in_play`` reads them off an observation (team A's view)."""
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    g = np.zeros((len(idx), 2, N_CH, H, W), F32)
    t = np.zeros((len(idx), 2, 4), F32)
    a = np.zeros((len(idx), 2, 10, N_AGENT_COLS), F32)
    batch_render_seq(bsim.Sb, bsim.bg, idx, g, t, a)
    on_map = np.rint(g[:, 0, 8] * 15).reshape(len(idx), -1).sum(1)
    carried = np.rint(np.maximum(a[:, 0, :, 4], 0) * 15).sum(1)
    return on_map + carried


# ---------------------------------------------------------------------------------- decoding

def mirror_order(k: int) -> np.ndarray:
    """World bin j of a team-B row = canonical bin order[j] (direction_bins.mirror_bin)."""
    return (k // 4 - np.arange(k)) % k


def decode_moves(q: np.ndarray, off: np.ndarray | None, team_b: np.ndarray) -> np.ndarray:
    """[n, 5, K] logits (+ offsets, radians, canonical frame) of n team views -> [n, 5, 2] float32
    world moves, the way ``MyPolicy.act`` and ``joint_moves``' clip turn them into moves.
    ``team_b`` [n] bool.  K bins, no STAND (a STAND-width head is not supported here)."""
    n, units, k = q.shape
    if k % 2:
        raise ValueError("decode_moves: STAND heads are not supported")
    order = mirror_order(k)
    qw = np.where(team_b[:, None, None], q[..., order], q)
    j = qw.argmax(axis=-1)  # first max in world order, like np.argmax on MyPolicy's world row
    if off is None:  # bin centres, exactly direction_bins.bin_vectors
        from blackout_env.model.direction_bins import bin_vectors

        vec = bin_vectors(k)[j]
    else:
        offw = np.where(team_b[:, None, None], -off[..., order], off)
        o = np.take_along_axis(offw, j[..., None], axis=-1)[..., 0].astype(np.float64)
        ang = 2 * math.pi * j / k + o  # direction_bins.offset_vectors' expression
        vec = np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)
    return np.clip(vec, F32(-1), F32(1))


# ---------------------------------------------------------------------------------- policies

class BatchPolicy:
    """Protocol: ``moves(bsim, idx, teams, inputs)`` -> {team: float32 [len(idx), 5, 2]} world moves
    for the own units of each team in ``teams`` (unit order 0-4 / 5-9).  ``inputs`` =
    (graphic [n, 2, 13, H, W], team [n, 2, 4], agents [n, 2, 10, 12]) already rendered for ``idx``
    (row j = game idx[j]).  ``reset(n)``: a new rollout of n games (stateful policies)."""

    needs_inputs = True

    def reset(self, n: int) -> None:
        pass

    def moves(self, bsim: BatchSim, idx: np.ndarray, teams: tuple, inputs) -> dict:
        raise NotImplementedError


class NetPolicy(BatchPolicy):
    """A stateless MyModel student (logits head, K bins, optional angle head: P10/P11) run on a
    whole batch: one forward over every requested (game, team) view.  ``fp16`` = autocast
    (MPS/CUDA); ``max_batch`` splits huge batches (MPS memory).

    ``per_sample=True`` runs every view as its own batch-1 forward -- slow, but bit-exact with
    ``MyPolicy`` (whose net always sees batch 1): torch's CPU ``Linear`` on a 2-D input uses a
    matrix-vector kernel at batch 1 and blocked GEMMs otherwise, which round differently
    (~1e-7 rad in the offsets, ~1e-5 in the logits), so a batched forward is not bit-exact with
    the single-game path even on CPU (tests/test_sim_batch.py, docs "S3 준비")."""

    def __init__(self, net, device: str = "cpu", fp16: bool = False, max_batch: int = 2048,
                 per_sample: bool = False, cuda_graphs: bool = False, graph_buckets=None):
        import torch

        from blackout_env.train.distill.inference import _OwnQ

        self.torch = torch
        self.device = torch.device(device)
        from .rollout_net import RolloutNet, RolloutOwnQ

        wrap = RolloutOwnQ if isinstance(net, RolloutNet) else _OwnQ
        self.own = wrap(net.eval()).to(self.device)
        self.angle = bool(getattr(net, "angle_head", False))
        self.fp16 = fp16
        self.max_batch = 1 if per_sample else max_batch
        # CUDA graphs per batch size (small-batch search: kernel launches dominate); same kernels -> same values
        # ``graph_buckets`` (sorted sizes): one graph per bucket, a batch is padded up to the next bucket
        # (the pad rows hold earlier real inputs; their outputs are dropped) -- a search server's batch
        # size changes every call, exact-size graphs would capture hundreds of graphs.  Larger -> eager.
        self.cuda_graphs = bool(cuda_graphs) and self.device.type == "cuda"
        self.graph_buckets = sorted(int(b) for b in graph_buckets) if graph_buckets else None
        self._graphs: dict = {}

    def _bucket(self, m: int):
        if self.graph_buckets is None:
            return m
        for b in self.graph_buckets:
            if b >= m:
                return b
        return None

    accepts_packed = True  # ``moves``/``forward`` take packed int16 maps too (BatchRollout(packed=True))

    def _own(self, g, t, a):
        """The net on device tensors; a 3-D ``g`` is a packed map, unpacked on the device."""
        return self.own(unpack_graphic(g) if g.dim() == 3 else g, t, a)

    def _graph_forward(self, g, t, a):
        torch = self.torch
        m = g.shape[0]
        size = self._bucket(m)
        key = (g.ndim, size)
        ent = self._graphs.get(key)
        if ent is None:
            pad = lambda x: np.concatenate([x, np.repeat(x[:1], size - m, 0)]) if size > m else x.copy()  # noqa: E731
            sg, st, sa = (torch.from_numpy(pad(x)).to(self.device) for x in (g, t, a))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):  # warm-up on a side stream before capture
                    self._own(sg, st, sa)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                out = self._own(sg, st, sa)
                out = torch.cat(out, -1) if self.angle else out  # one device -> host copy
            ent = self._graphs[key] = (graph, sg, st, sa, out)
        graph, sg, st, sa, out = ent
        sg[:m].copy_(torch.from_numpy(g))
        st[:m].copy_(torch.from_numpy(t))
        sa[:m].copy_(torch.from_numpy(a))
        graph.replay()
        o = out[:m].float().cpu().numpy()
        if not self.angle:
            return o, None
        k = o.shape[-1] // 2
        return o[..., :k], o[..., k:]

    @classmethod
    def from_checkpoint(cls, path: str, **kw) -> "NetPolicy":
        import torch

        from blackout_env.train.distill.inference import load_net

        return cls(load_net(torch.load(path, map_location="cpu", weights_only=True)["policy_state"]), **kw)

    def forward(self, graphic, team, agents) -> tuple[np.ndarray, np.ndarray | None]:
        """[m, 13, H, W] (or packed int16 [m, H, W]), [m, 4], [m, 10, 12] float32 numpy -> logits,
        offsets [m, 5, K]."""
        torch = self.torch
        if (self.cuda_graphs and len(graphic) <= self.max_batch and not self.fp16
                and self._bucket(len(graphic)) is not None):
            return self._graph_forward(np.ascontiguousarray(graphic), np.ascontiguousarray(team),
                                       np.ascontiguousarray(agents))
        qs, offs = [], []
        for s in range(0, len(graphic), self.max_batch):
            g = torch.from_numpy(graphic[s:s + self.max_batch]).to(self.device)
            t = torch.from_numpy(team[s:s + self.max_batch]).to(self.device)
            a = torch.from_numpy(agents[s:s + self.max_batch]).to(self.device)
            with torch.inference_mode():
                if self.fp16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        out = self._own(g, t, a)
                else:
                    out = self._own(g, t, a)
            q, off = out if self.angle else (out, None)
            qs.append(q.float().cpu().numpy())
            offs.append(off.float().cpu().numpy() if off is not None else None)
        q = np.concatenate(qs)
        off = np.concatenate(offs) if self.angle else None
        return q, off

    def moves(self, bsim, idx, teams, inputs):
        graphic, team, agents = inputs
        n = len(idx)
        sel = list(teams)
        cut = (slice(None) if sel == [0, 1] else sel)  # both teams: a view, no copy of the batch
        g = graphic[:n][:, cut].reshape(n * len(sel), *graphic.shape[2:])
        t = team[:n][:, cut].reshape(n * len(sel), 4)
        a = agents[:n][:, cut].reshape(n * len(sel), 10, N_AGENT_COLS)
        q, off = self.forward(g, t, a)
        team_b = np.tile(np.asarray(sel) == 1, n)
        mv = decode_moves(q, off, team_b).reshape(n, len(sel), 5, 2)
        return {tm: mv[:, i] for i, tm in enumerate(sel)}


class HeuristicPolicy(BatchPolicy):
    """Any ``make_opponent`` descriptor, one instance per game *and team* (a heuristic instance binds
    to the team it first acts for: a V23 shared by both teams returns zero moves for the second;
    history-dependent heuristics keep their own state), fed ``obs_from_raw`` dicts: the slow
    reference path and a fallback."""

    needs_inputs = False

    def __init__(self, descriptor: str, seed: int = 0):
        self.descriptor, self.seed = descriptor, seed
        self.instances: dict = {}

    def reset(self, n: int) -> None:
        self.instances = {}

    def _instance(self, b: int, team: int):
        key = (b, team)
        if key not in self.instances:
            from blackout_env.train.distill.opponents import make_opponent

            self.instances[key] = make_opponent(self.descriptor, self.seed + 2 * b + team)
        return self.instances[key]

    def moves(self, bsim, idx, teams, inputs):
        out = {tm: np.zeros((len(idx), 5, 2), np.float32) for tm in teams}
        for j, b in enumerate(idx):
            obs = obs_from_raw(*bsim.raw(int(b)))
            for tm in teams:
                names = [f"unit_{i}" for i in range(5 * tm, 5 * tm + 5)]
                acts = self._instance(int(b), tm).act({n: obs[n] for n in names})
                for n_, a in acts.items():
                    out[tm][j, int(n_.split("_")[1]) - 5 * tm] = np.clip(np.asarray(a, np.float32), -1, 1)
        return out


class HybridPolicy(BatchPolicy):
    """``first`` while a game's episode clock is before ``switch_ticks`` (the opening race needs
    P11's precision), ``then`` afterwards (e.g. the small rollout net)."""

    def __init__(self, first: BatchPolicy, then: BatchPolicy, switch_ticks: int = OPENING_TICKS):
        self.first, self.then, self.switch_ticks = first, then, int(switch_ticks)
        self.needs_inputs = first.needs_inputs or then.needs_inputs

    def reset(self, n: int) -> None:
        self.first.reset(n)
        self.then.reset(n)

    def moves(self, bsim, idx, teams, inputs):
        early = bsim.g[idx, core.G_TICK] < self.switch_ticks
        if early.all() or not early.any():
            return (self.first if early.all() else self.then).moves(bsim, idx, teams, inputs)
        out = {tm: np.zeros((len(idx), 5, 2), np.float32) for tm in teams}
        for pol, rows in ((self.first, early), (self.then, ~early)):
            sub = tuple(x[: len(idx)][rows] for x in inputs)
            for tm, mv in pol.moves(bsim, idx[rows], teams, sub).items():
                out[tm][rows] = mv
        return out


# ---------------------------------------------------------------------------------- rollouts

@dataclass
class RolloutResult:
    scores: np.ndarray        # int64 [N, 2] final (A, B)
    over: np.ndarray          # bool [N] the match ended within the horizon
    winner: np.ndarray        # int [N] 0 / 1 / -1 draw or not over
    steps: np.ndarray         # int [N] decisions taken
    ticks: np.ndarray         # int [N] episode ticks at the end
    first_hunter: np.ndarray  # int [N, 2] first decision index (from the root) a team had a Hunter, -1 never
    positions: np.ndarray | None  # float32 [T, N, 10, 2] per-step positions (record=True)
    bsim: BatchSim
    exhausted: np.ndarray | None = None   # bool [N] stopped: no battery left after an absorption (stop_when_exhausted)
    lead_trace: np.ndarray | None = None  # int [len(trace_steps), N] team A's lead after each trace step (the final lead past the end)

    def lead(self, team: int = 0) -> np.ndarray:
        return self.scores[:, team] - self.scores[:, 1 - team]

    def states(self) -> list[SimState]:
        return [self.bsim.state(b) for b in range(self.bsim.n)]


class BatchRollout:
    """Plays N games in lockstep: ``policy_a`` for team A, ``policy_b`` for team B, ``horizon``
    decisions (or until each game ends).  ``repeat`` k: once a game's episode clock passes
    ``opening_ticks`` the policies are only called every k decisions (from the first decision past
    the opening), the moves in between repeat the last ones."""

    def __init__(self, policy_a: BatchPolicy, policy_b: BatchPolicy, repeat: int = 1,
                 opening_ticks: int = OPENING_TICKS, parallel: bool | None = None, repeat_b: int | None = None,
                 packed: bool = False):
        self.pa, self.pb = policy_a, policy_b
        # packed: render the map as int16 [n, 2, H, W] (13x fewer bytes to the device; the policies
        # unpack on the device) -- only for policies that accept it
        self.packed = bool(packed) and all(getattr(p, "accepts_packed", False) or not p.needs_inputs
                                           for p in (policy_a, policy_b))
        self.repeat = (int(repeat), int(repeat if repeat_b is None else repeat_b))  # per team
        self.opening_ticks, self.parallel = int(opening_ticks), parallel
        self._buf = None

    def _buffers(self, n):
        if self._buf is None or self._buf[0].shape[0] < n:
            g = np.zeros((n, 2, H, W), np.int16) if self.packed else np.zeros((n, 2, N_CH, H, W), F32)
            self._buf = (g, np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
        return self._buf

    def run(self, roots: list[SimState], horizon: int, reseeds=None, record: bool = False,
            bsim: BatchSim | None = None, stop_when_exhausted: bool = False, trace_steps=()) -> RolloutResult:
        """``stop_when_exhausted``: a game also ends at an absorption that leaves no battery in play
        (floor, storages, carried) -- the Unity evaluations' ``MatchTask.stop_when_exhausted``: nothing
        is left to score.  ``trace_steps``: decision counts after which team A's lead is recorded."""
        if bsim is None:
            bsim = BatchSim(roots)
        else:
            bsim.load(roots)
        if reseeds is not None:
            bsim.reseed(reseeds)
        n = bsim.n
        self.pa.reset(n)
        if self.pb is not self.pa:
            self.pb.reset(n)
        bufs = self._buffers(n)
        moves = np.zeros((n, 10, 2), F32)
        steps = np.zeros(n, np.int64)
        since = np.zeros((2, n), np.int64)  # past the opening: decisions since a team's last policy call, mod repeat
        first = np.full((n, 2), -1, np.int64)
        for team in (0, 1):
            first[(bsim.ui[:, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1), team] = 0
        pos = np.zeros((horizon, n, 10, 2), F32) if record else None
        need_inputs = self.pa.needs_inputs or self.pb.needs_inputs
        clock = time.perf_counter
        tm_ = self.timing = {"render": 0.0, "policy": 0.0, "step": 0.0, "calls": 0, "game_decisions": 0}
        done = 0
        stopped = np.zeros(n, bool)
        trace_steps = [int(x) for x in trace_steps]
        trace = np.zeros((len(trace_steps), n), np.int64)
        for t in range(horizon):
            live = np.nonzero(~bsim.over & ~stopped)[0]
            if not len(live):
                break
            past = bsim.g[live, core.G_TICK] >= self.opening_ticks
            calls = [(~past | (since[team, live] == 0)) if self.repeat[team] > 1 else np.ones(len(live), bool)
                     for team in (0, 1)]
            union = calls[0] | calls[1]
            act = live[union]
            if len(act):
                c0 = clock()
                if need_inputs:
                    (bsim.render_packed if self.packed else bsim.render)(act, *bufs, parallel=self.parallel)
                c1 = clock()
                ca, cb = calls[0][union], calls[1][union]  # rows of act each team acts on
                if self.pa is self.pb and ca.all() and cb.all():
                    out = self.pa.moves(bsim, act, (0, 1), bufs)
                    moves[act, :5], moves[act, 5:] = out[0], out[1]
                else:
                    for team, pol, rows in ((0, self.pa, ca), (1, self.pb, cb)):
                        if not rows.any():
                            continue
                        sub = bufs if rows.all() else tuple(x[: len(act)][rows] for x in bufs)
                        moves[act[rows], 5 * team:5 * team + 5] = pol.moves(bsim, act[rows], (team,), sub)[team]
                tm_["render"] += c1 - c0
                tm_["policy"] += clock() - c1
                tm_["calls"] += len(act)
            for team in (0, 1):
                if self.repeat[team] > 1:
                    since[team, live] = np.where(past, (since[team, live] + 1) % self.repeat[team], 0)
            c0 = clock()
            abs_before = bsim.gf[live, core.GF_ABS].copy() if stop_when_exhausted else None
            bsim.step(moves, live, parallel=self.parallel)
            tm_["step"] += clock() - c0
            if stop_when_exhausted:
                fired = live[(bsim.gf[live, core.GF_ABS] < abs_before) & ~bsim.over[live]]
                if len(fired):
                    stopped[fired[batteries_in_play(bsim, fired) == 0]] = True
            tm_["game_decisions"] += len(live)
            steps[live] += 1
            for team in (0, 1):
                hit = (first[live, team] < 0) & (bsim.ui[live, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1)
                first[live[hit], team] = t + 1
            if record:
                pos[t] = bsim.uf
            for i, at in enumerate(trace_steps):
                if at == t + 1:
                    trace[i] = bsim.scores[:, 0] - bsim.scores[:, 1]
            done = t + 1
        final = bsim.scores[:, 0] - bsim.scores[:, 1]
        for i, at in enumerate(trace_steps):
            ended = steps < at  # the game ended before this step: its final lead
            trace[i, ended] = final[ended]
        over = bsim.over.copy()
        winner = np.where(over, bsim.g[:, core.G_WINNER], -1)
        winner = np.where(stopped, np.where(final > 0, 0, np.where(final < 0, 1, -1)), winner)
        return RolloutResult(bsim.scores, over, winner, steps, bsim.ticks, first,
                             pos[:done] if record else None, bsim, stopped if stop_when_exhausted else None,
                             trace if trace_steps else None)
