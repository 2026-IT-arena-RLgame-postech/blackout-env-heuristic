"""GRPO on the simulator: branch groups + value bootstrap (docs/rl_finetune_plan_ko.md, run R3).

One iteration:

1. **Trunk games** (``G`` of them, official rules: ``SimState.from_seed``, 420 s) advance ``trunk_step``
   decisions with the learner playing greedily against its opponent; their states become the roots.
   A finished trunk game restarts from a fresh seed with a new opponent and side.
2. **Branch groups**: every root is copied ``K`` times; all K copies get the same sim reseed (common
   random numbers) and the same opponent.  The learner *samples* its first ``sample_steps`` decisions
   (the only decisions that get a gradient) and then plays greedily for ``leaf_steps`` more; the
   opponent plays its own policy throughout.
3. **Reward** at the leaf (exact if the match ended): ``R = w + lam * tanh(lead / scale)``, ``w`` the
   value net's mirror-symmetric P(win) - P(loss) (sign of the lead once over), ``lead`` its
   symmetric final-lead estimate (points).
4. **Advantage** ``A = R - mean_group(R)``, divided by the batch std of those deviations (one scale
   for the whole batch, so a group whose branches all end the same has A ~ 0 instead of amplified
   noise).
5. **Update**: PPO-clipped ratio per unit action (the 5 units' bins are independent categoricals),
   + ``beta`` * exact KL(pi || pi_ref) over the bins (pi_ref = the starting student).  The angle
   offsets stay deterministic (no policy gradient through them).

Opponents are drawn per trunk game from ``opponents`` with weights ``opp_weights`` (default uniform),
reweighted by PFSP when ``pfsp`` is set: each opponent keeps an EMA of the learner's score against it
(branch leaf winness mapped to [0, 1]); ``hard`` multiplies its base weight by ``floor + (1 - s)**pfsp_p``
(play more against what beats us), ``even`` by ``floor + 4 s (1 - s)``.  Heuristic opponents in the
branches are separate instances, reset every iteration (a branch never sees another game's state).

Value-net retraining (``value_train``; iterative GRPO, Shao et al. 2024 Alg. 1, where the reward model is
retrained on the policy's own samples with 10 % historical replay): the trunk is advanced in
``value_snaps`` chunks and both teams' views of every live trunk game are kept per chunk; when a trunk
game ends its rows get the final lead and result (in that view's frame) and enter a ring buffer.  Every
``value_every`` iterations the leaf value net takes ``value_steps`` Adam steps on the final + result heads
(``value_loss`` with the horizon deltas masked), each batch ``1 - value_replay`` from the newest
``value_recent`` rows and ``value_replay`` from older ones.

Only stateless MyModel students with a logits head (P24c) are supported.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from blackout_env.sim import core
from blackout_env.sim.batch import (
    F32,
    BatchPolicy,
    BatchRollout,
    BatchSim,
    HeuristicPolicy,
    NetPolicy,
    decode_moves,
    unpack_graphic,
)
from blackout_env.sim.state import SimState
from blackout_env.train.distill.inference import _OwnQ, load_net


# ---------------------------------------------------------------------------------- policies

class LearnerPolicy(BatchPolicy):
    """The trainable student.  Per game: the first ``sample_steps`` of its own decisions (counted from
    ``reset``) are sampled from softmax(logits / temperature) and recorded; later ones are greedy
    (exactly ``NetPolicy``'s argmax decode).  ``sample_steps = 0``: greedy everywhere."""

    needs_inputs = True
    accepts_packed = True

    def __init__(self, own: _OwnQ, device, sample_steps: int = 0, temperature: float = 1.0,
                 fp16: bool = True, seed: int = 0, graphs=None):
        self.own, self.device = own, torch.device(device)
        self.sample_steps, self.temperature, self.fp16 = int(sample_steps), float(temperature), fp16
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        self.t = np.zeros(0, np.int64)
        self.records: list = []
        self.graphs = graphs   # a sim.graphfwd.GraphForward over ``own`` (fp16 CUDA only), else the eager path

    def reset(self, n: int) -> None:
        self.t = np.zeros(n, np.int64)
        self.records = []

    @torch.no_grad()
    def _forward(self, g, t, a):
        if self.graphs is not None and g.ndim == 3:
            out = self.graphs(g, t, a)
            return out[..., :self.graphs.kq], out[..., self.graphs.kq:]
        g = torch.from_numpy(np.ascontiguousarray(g)).to(self.device)
        g = unpack_graphic(g) if g.dim() == 3 else g
        t = torch.from_numpy(np.ascontiguousarray(t)).to(self.device)
        a = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16 and self.device.type == "cuda"):
            q, off = self.own(g, t, a)
        return q.float(), off.float()

    def moves(self, bsim, idx, teams, inputs):
        graphic, team, agents = inputs
        n = len(idx)
        out = {}
        for i, tm in enumerate(teams):
            g, ts, ag = graphic[:n][:, tm], team[:n][:, tm], agents[:n][:, tm]
            q, off = self._forward(g, ts, ag)
            samp = self.t[idx] < self.sample_steps
            chosen = q.argmax(-1)
            r = None
            if samp.any():
                r = np.nonzero(samp)[0]
                rows = torch.from_numpy(r).to(self.device)
                p = torch.softmax(q[rows] / self.temperature, -1)
                pick = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=self.rng).view(p.shape[:-1])
                chosen[rows] = pick
            chosen_np = chosen.cpu().numpy()
            if r is not None:
                self.records.append((idx[r].copy(), g[r].copy(), ts[r].copy(), ag[r].copy(),
                                     chosen_np[r].astype(np.int8)))
            onehot = np.zeros(q.shape, np.float32)
            np.put_along_axis(onehot, chosen_np[..., None], 1.0, -1)
            team_b = np.full(n, tm == 1)
            out[tm] = decode_moves(onehot, off.cpu().numpy(), team_b)
            self.t[idx] += 1
        return out


def _moves_views(self, idx, g, ts, ag, team, chunks):
    """The learner on one view per row (row j = game ``idx[j]`` seen as team ``team[j]``) in a single forward.
    ``chunks`` = [(lo, hi), ...]: contiguous row ranges that the sequential per-team calls would have been; sampling
    (multinomial draws, ``records``) is done chunk by chunk in that order, so the RNG stream and the record order are
    those of the per-team ``moves`` calls (only the forward's batch composition differs)."""
    q, off = self._forward(g, ts, ag)
    chosen = q.argmax(-1)
    sampled = []
    for lo, hi in chunks:
        if hi <= lo:
            continue
        samp = self.t[idx[lo:hi]] < self.sample_steps
        if samp.any():
            r = np.nonzero(samp)[0] + lo
            rows = torch.from_numpy(r).to(self.device)
            p = torch.softmax(q[rows] / self.temperature, -1)
            pick = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=self.rng).view(p.shape[:-1])
            chosen[rows] = pick
            sampled.append(r)
    chosen_np = chosen.cpu().numpy()
    for r in sampled:
        self.records.append((idx[r].copy(), g[r].copy(), ts[r].copy(), ag[r].copy(), chosen_np[r].astype(np.int8)))
    self.t[idx] += 1
    onehot = np.zeros(q.shape, np.float32)
    np.put_along_axis(onehot, chosen_np[..., None], 1.0, -1)
    return decode_moves(onehot, off.cpu().numpy(), team == 1)


LearnerPolicy.moves_views = torch.no_grad()(_moves_views)


class JointSeats(BatchPolicy):
    """``Seats`` for both teams at once (one policy object for team A and B: ``BatchRollout`` makes a single call per
    decision step with both teams' views).  Each policy (the learner, every net opponent) then runs one forward over
    the views of all its games instead of one per team, and heuristic opponents are started on their worker processes
    before the forwards and collected after them."""

    needs_inputs = True
    accepts_packed = True

    def __init__(self, learner: LearnerPolicy, opps: list[BatchPolicy], opp_of: np.ndarray, learner_team: np.ndarray,
                 split: bool = False):
        self.learner, self.opps, self.opp_of, self.learner_team = learner, opps, opp_of, learner_team
        self.split = split   # keep every network forward's batch composition of the per-team ``Seats`` calls (bit-identical)

    def reset(self, n: int) -> None:
        self.learner.reset(n)

    def moves(self, bsim, idx, teams, inputs):
        assert tuple(teams) == (0, 1)
        if self.split:
            return self._moves_split(bsim, idx, inputs)
        n = len(idx)
        graphic, team, agents = (x[:n] for x in inputs)
        lt = self.learner_team[idx]
        otm = 1 - lt
        ok = self.opp_of[idx]
        out = np.zeros((2, n, 5, 2), F32)
        later = []
        pend = []
        for k, p in enumerate(self.opps):
            rows = np.nonzero(ok == k)[0]
            if not len(rows):
                continue
            if hasattr(p, "start"):
                pend.append((rows, p, p.start(bsim, idx[rows], otm[rows])))
            else:
                later.append((rows, p))
        # learner: team-A-seat games first, then team-B-seat games (the order of the per-team calls)
        r0, r1 = np.nonzero(lt == 0)[0], np.nonzero(lt == 1)[0]
        lr = np.concatenate([r0, r1])
        if len(lr):
            mv = self.learner.moves_views(idx[lr], graphic[lr, lt[lr]], team[lr, lt[lr]], agents[lr, lt[lr]],
                                          lt[lr], [(0, len(r0)), (len(r0), len(lr))])
            out[lt[lr], lr] = mv
        for rows, p in later:
            tm_rows = otm[rows]
            if hasattr(p, "forward"):   # NetPolicy: one forward over the views of all its games
                o = otm[rows]
                q, off = p.forward(graphic[rows, o], team[rows, o], agents[rows, o])
                out[o, rows] = decode_moves(q, off, o == 1)
            else:
                for tm in (0, 1):
                    sel = rows[tm_rows == tm]
                    if len(sel):
                        sub = tuple(x[sel] for x in (graphic, team, agents))
                        out[tm, sel] = p.moves(bsim, idx[sel], (tm,), sub)[tm]
        for rows, p, h in pend:
            out[otm[rows], rows] = p.finish(h)
        return {0: out[0], 1: out[1]}


    def _moves_split(self, bsim, idx, inputs):
        """What the two per-team ``Seats`` calls (team 0, then team 1) compute, with the heuristic opponents of both
        teams started on the workers before any network forward and collected after them.  Every learner / net
        forward sees exactly the rows it would see under ``Seats`` (same batch sizes and order: bit-identical moves,
        records and RNG stream); heuristic instances are keyed per (game, team), so shipping both teams' rows in one
        message changes nothing."""
        n = len(idx)
        sub_all = tuple(x[:n] for x in inputs)
        lt = self.learner_team[idx]
        otm = 1 - lt
        ok = self.opp_of[idx]
        out = np.zeros((2, n, 5, 2), F32)
        pend = []
        for k, p in enumerate(self.opps):
            if hasattr(p, "start"):
                rows = np.nonzero(ok == k)[0]
                if len(rows):
                    pend.append((rows, p, p.start(bsim, idx[rows], otm[rows])))
        for tm in (0, 1):
            mine = lt == tm
            if mine.any():
                sub = tuple(x[mine] for x in sub_all)
                out[tm, mine] = self.learner.moves(bsim, idx[mine], (tm,), sub)[tm]
            for k, p in enumerate(self.opps):
                if hasattr(p, "start"):
                    continue
                rows = np.nonzero(~mine & (ok == k))[0]
                if len(rows):
                    sub = tuple(x[rows] for x in sub_all)
                    out[tm, rows] = p.moves(bsim, idx[rows], (tm,), sub)[tm]
        for rows, p, h in pend:
            out[otm[rows], rows] = p.finish(h)
        return {0: out[0], 1: out[1]}


class Seats(BatchPolicy):
    """Plays one team of every game: the learner in the games where ``learner_team[b] == team``,
    otherwise opponent ``opps[opp_of[b]]``.  ``reset`` resets the learner but not the opponents (a trunk game's
    heuristic opponent keeps its state across trunk segments; call ``forget`` for restarted games)."""

    needs_inputs = True
    accepts_packed = True

    def __init__(self, team: int, learner: LearnerPolicy, opps: list[BatchPolicy], opp_of: np.ndarray,
                 learner_team: np.ndarray):
        self.team, self.learner, self.opps = team, learner, opps
        self.opp_of, self.learner_team = opp_of, learner_team

    def reset(self, n: int) -> None:
        self.learner.reset(n)  # both seats reset the shared learner at the start of a rollout: harmless

    def moves(self, bsim, idx, teams, inputs):
        (tm,) = teams
        out = np.zeros((len(idx), 5, 2), F32)
        mine = self.learner_team[idx] == tm
        groups = [(self.learner, mine)] + [(p, ~mine & (self.opp_of[idx] == k)) for k, p in enumerate(self.opps)]
        for pol, rows in groups:
            if rows.any():
                sub = tuple(x[: len(idx)][rows] for x in inputs)
                out[rows] = pol.moves(bsim, idx[rows], (tm,), sub)[tm]
        return {tm: out}


def _heuristic_aliases() -> dict:
    from blackout_env.heuristics.params import ALIASES

    return dict(ALIASES)


def make_opponent_policy(spec: str, device: str, seed: int = 0, graphs: bool = False, workers=None) -> BatchPolicy:
    """``net:<checkpoint>`` -> batched ``NetPolicy`` (fp16 on CUDA); anything else -> ``HeuristicPolicy``
    (a ``make_opponent`` descriptor, or an alias ``V32`` / ``V32.1``; slow, per game in Python)."""
    if spec.startswith("net:"):
        return NetPolicy.from_checkpoint(spec[4:], device=device, fp16=device.startswith("cuda"), graphs=graphs)
    desc = _heuristic_aliases().get(spec, spec)
    if workers is not None:
        from blackout_env.rl.hpool import ParallelHeuristic

        return ParallelHeuristic(desc, seed, workers)
    return HeuristicPolicy(desc, seed)


def forget(pol: BatchPolicy, games) -> None:
    """Drop a heuristic opponent's per-game state for restarted games."""
    if hasattr(pol, "forget_games"):
        pol.forget_games(games)
    elif isinstance(pol, HeuristicPolicy):
        for b in games:
            for tm in (0, 1):
                pol.instances.pop((int(b), tm), None)


# ---------------------------------------------------------------------------------- reward

class LeafValue:
    """Mirror-symmetric value-net reading at a leaf: (winness, lead) for ``teams[b]``; exact when over."""

    def __init__(self, path: str, device: str, batch: int = 4096):
        from blackout_env.value.model import load_value_net

        self.device = torch.device(device)
        self.net = load_value_net(path, device=self.device).eval()
        self.batch = batch

    @torch.no_grad()
    def __call__(self, bsim: BatchSim, teams: np.ndarray):
        from blackout_env.sim.batch import N_AGENT_COLS, N_CH, H, W
        from blackout_env.value.model import decode_outputs

        lead = (bsim.scores[:, 0] - bsim.scores[:, 1]).astype(np.float64)
        lead = np.where(teams == 0, lead, -lead)
        win = np.sign(lead)
        live = np.nonzero(~bsim.over)[0]
        if len(live):
            gr = np.zeros((len(live), 2, N_CH, H, W), F32)
            tm = np.zeros((len(live), 2, 4), F32)
            ag = np.zeros((len(live), 2, 10, N_AGENT_COLS), F32)
            bsim.render(live, gr, tm, ag)
            gr, tm, ag = gr.reshape(-1, N_CH, H, W), tm.reshape(-1, 4), ag.reshape(-1, 10, N_AGENT_COLS)
            fin, wl = [], []
            for s in range(0, len(gr), self.batch):
                g = torch.from_numpy(gr[s:s + self.batch]).to(self.device)
                t = torch.from_numpy(tm[s:s + self.batch]).to(self.device)
                a = torch.from_numpy(ag[s:s + self.batch]).to(self.device)
                d = decode_outputs(self.net(g, t, a).float(), t[:, 0] - t[:, 1])
                fin.append(d["final"] * 100.0)
                wl.append(d["p"][:, 2] - d["p"][:, 0])
            fin = np.concatenate(fin).reshape(-1, 2)
            wl = np.concatenate(wl).reshape(-1, 2)
            own = (teams[live] == 1).astype(int)
            r = np.arange(len(live))
            lead[live] = (fin[r, own] - fin[r, 1 - own]) / 2.0
            win[live] = (wl[r, own] - wl[r, 1 - own]) / 2.0
        return win, lead


# ---------------------------------------------------------------------------------- trainer

@dataclass
class GRPOConfig:
    init: str = "checkpoints/distill/frozen/p24c_v321.pt"
    value: str = "checkpoints/value/s2v_f3xd2/iter2.pt"
    opponents: list = field(default_factory=lambda: ["net:checkpoints/distill/frozen/p24c_v321.pt"])
    opp_weights: list | None = None
    pfsp: str = "none"         # none | hard | even
    pfsp_p: float = 2.0
    pfsp_floor: float = 0.1
    pfsp_ema: float = 0.05     # per-iteration EMA rate of the per-opponent score
    drop_flat: float = 0.0     # drop groups whose reward std is below this (DAPO-style dynamic sampling)
    ref_reset_every: int = 0   # >0: pi_ref <- current policy every N iterations (iterative GRPO, Shao et al. Alg. 1)
    value_train: bool = False  # retrain the leaf value net on finished trunk games
    value_snaps: int = 5       # trunk chunks (= view snapshots) per iteration
    value_every: int = 10
    value_steps: int = 50
    value_batch: int = 256
    value_lr: float = 1e-5
    value_recent: int = 20000  # rows counted as "new"
    value_replay: float = 0.1  # share of each batch from older rows
    value_min_rows: int = 5000
    value_buffer: int = 400000
    groups: int = 64           # G roots per iteration (= trunk games)
    k: int = 8                 # branches per root
    trunk_step: int = 250      # decisions the trunk games advance between roots (10 s)
    sample_steps: int = 25     # sampled learner decisions per branch (1 s)
    leaf_steps: int = 125      # greedy decisions after them before the value read (5 s)
    temperature: float = 1.0
    lam: float = 0.2           # margin weight
    scale: float = 30.0        # margin tanh scale, points
    lr: float = 3e-6
    clip: float = 0.2
    beta: float = 0.02         # KL(pi || ref)
    epochs: int = 2
    minibatch: int = 1024
    max_grad_norm: float = 1.0
    device: str = "cuda"
    seed: int = 0
    joint: bool = False        # (opt-in: not bit-identical, fp16 batch-composition noise) one policy call per decision step for both teams (JointSeats): one forward per policy, not per team
    heur_workers: int = -1     # heuristic opponents in this many worker processes (-1: 4 on CUDA, else none; 0: in-process)
    update_fp16: bool = False  # autocast fp16 in update()'s forward/backward (~2x faster; changes numerics slightly, off by default)
    overlap_heur: bool = False  # heuristic opponents of both teams run on the workers during the network forwards (bit-identical to the per-team Seats calls; ignored with joint)
    graphs: bool = True        # CUDA-graph replay of the learner / net-opponent forwards (bit-identical, sim/graphfwd.py)


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        ck = torch.load(cfg.init, map_location="cpu", weights_only=True)
        net = load_net(ck["policy_state"])
        assert net.policy_head == "logits", "GRPO needs a logits-head student"
        self.net = net.to(self.dev).eval()   # no dropout: eval mode also while training
        self.own = _OwnQ(self.net)
        self.ref = _OwnQ(copy.deepcopy(self.net).eval().requires_grad_(False))
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.leaf = LeafValue(cfg.value, cfg.device)
        gr = bool(cfg.graphs) and self.dev.type == "cuda"
        self.learner_gf = None
        if gr:
            from blackout_env.sim.graphfwd import GraphForward

            self.learner_gf = GraphForward(self.own, self.dev, angle=True, max_batch=2048)
        nw = cfg.heur_workers if cfg.heur_workers >= 0 else (4 if self.dev.type == "cuda" else 0)
        self.hworkers = None
        if nw > 0 and any(not s.startswith("net:") for s in cfg.opponents):
            from blackout_env.rl.hpool import HeuristicWorkers

            self.hworkers = HeuristicWorkers(nw)
        self.opps = [make_opponent_policy(s, cfg.device, cfg.seed + 7 * i, graphs=gr, workers=self.hworkers)
                     for i, s in enumerate(cfg.opponents)]
        # branches get their own heuristic instances (nets are stateless and shared)
        mk = (lambda p: p.__class__(p.descriptor, p.seed + 1, self.hworkers) if self.hworkers is not None
              else HeuristicPolicy(p.descriptor, p.seed + 1))
        self.branch_opps = [mk(p) if isinstance(p, HeuristicPolicy) else p for p in self.opps]
        w = np.asarray(cfg.opp_weights if cfg.opp_weights else [1.0] * len(self.opps), np.float64)
        assert len(w) == len(self.opps), "opp_weights must match opponents"
        self.base_w = w / w.sum()
        self.opp_score = np.full(len(self.opps), 0.5)
        self.opp_w = self._pfsp_weights()
        g = cfg.groups
        self.trunk = [self._fresh_seed() for _ in range(g)]
        self.trunk_opp = self.rng.choice(len(self.opps), g, p=self.opp_w)
        self.trunk_side = self.rng.integers(0, 2, g)
        self.iter = 0
        self.vbuf = ValueBuffer(cfg.value_buffer) if cfg.value_train else None
        if cfg.value_train:
            self.vopt = torch.optim.AdamW(self.leaf.net.parameters(), lr=cfg.value_lr, weight_decay=0.0)
            self.pending = {b: [] for b in range(g)}

    def _pfsp_weights(self) -> np.ndarray:
        c, s = self.cfg, self.opp_score
        if c.pfsp == "hard":
            f = c.pfsp_floor + (1.0 - s) ** c.pfsp_p
        elif c.pfsp == "even":
            f = c.pfsp_floor + 4.0 * s * (1.0 - s)
        else:
            assert c.pfsp == "none", c.pfsp
            f = np.ones_like(s)
        w = self.base_w * f
        return w / w.sum()

    def _fresh_seed(self) -> SimState:
        return SimState.from_seed(int(self.rng.integers(0, 2**31 - 1)))

    def _rollout(self, roots, learner, opp_of, side, horizon, reseeds=None, opps=None):
        opps = self.opps if opps is None else opps
        if self.cfg.joint or self.cfg.overlap_heur:
            pj = JointSeats(learner, opps, opp_of, side, split=not self.cfg.joint)
            return BatchRollout(pj, pj, packed=True).run(roots, horizon, reseeds=reseeds)
        pa = Seats(0, learner, opps, opp_of, side)
        pb = Seats(1, learner, opps, opp_of, side)
        return BatchRollout(pa, pb, packed=True).run(roots, horizon, reseeds=reseeds)

    # -------- collection
    def collect(self):
        c = self.cfg
        t0 = time.perf_counter()
        greedy = LearnerPolicy(self.own, self.dev, 0, graphs=self.learner_gf)
        n_chunks = c.value_snaps if c.value_train else 1
        roots, steps = self.trunk, 0
        for j in range(n_chunks):
            h = c.trunk_step // n_chunks + (1 if j < c.trunk_step % n_chunks else 0)
            res = self._rollout(roots, greedy, self.trunk_opp, self.trunk_side, h)
            roots, steps = res.states(), steps + int(res.steps.sum())
            if c.value_train:
                self._snapshot(res)
        # restart finished trunk games (fresh seed, opponent, side); their heuristic state is dropped
        ended = np.nonzero(res.over)[0]
        if c.value_train:
            self._label(ended, res.bsim.scores)
        for b in ended:
            roots[b] = self._fresh_seed()
            self.trunk_opp[b] = self.rng.choice(len(self.opps), p=self.opp_w)
            self.trunk_side[b] = self.rng.integers(0, 2)
        for p in self.opps:
            forget(p, ended)
        self.trunk = roots
        t1 = time.perf_counter()
        # branch groups from the live roots (a just-restarted game is a fine root too: the opening)
        g_idx = np.arange(len(roots))
        k = c.k
        branch_roots = [roots[b] for b in g_idx for _ in range(k)]
        opp_of = np.repeat(self.trunk_opp[g_idx], k)
        side = np.repeat(self.trunk_side[g_idx], k)
        reseeds = np.repeat(self.rng.integers(0, 2**31 - 1, len(g_idx)), k)
        learner = LearnerPolicy(self.own, self.dev, c.sample_steps, c.temperature, seed=int(self.rng.integers(2**31)),
                                graphs=self.learner_gf)
        for p in self.branch_opps:
            if isinstance(p, HeuristicPolicy):
                p.reset(0)
        br = self._rollout(branch_roots, learner, opp_of, side, c.sample_steps + c.leaf_steps, reseeds,
                           opps=self.branch_opps)
        t2 = time.perf_counter()
        win, lead = self.leaf(br.bsim, side)
        # PFSP: per-opponent EMA of the learner's leaf score, then reweight the restart draw
        score = (win + 1.0) / 2.0
        opp_stats = {}
        for i in range(len(self.opps)):
            m = opp_of == i
            if m.any():
                self.opp_score[i] += c.pfsp_ema * (float(score[m].mean()) - self.opp_score[i])
            opp_stats[i] = int(m.sum() // k)
        self.opp_w = self._pfsp_weights()
        R = win + c.lam * np.tanh(lead / c.scale)
        Rg = R.reshape(-1, k)
        dev = Rg - Rg.mean(1, keepdims=True)
        scale = dev.std() + 1e-8
        adv = (dev / scale).reshape(-1)
        recs = learner.records
        game = np.concatenate([r[0] for r in recs])
        keep = np.repeat(Rg.std(1) >= c.drop_flat, k)[game]   # all True when drop_flat = 0
        cat = lambda j: np.concatenate([r[j] for r in recs])[keep]
        batch = {"g": cat(1), "t": cat(2), "a": cat(3), "act": cat(4), "adv": adv[game][keep].astype(np.float32)}
        t3 = time.perf_counter()
        stats = {
            "trunk_s": t1 - t0, "branch_s": t2 - t1, "leaf_s": t3 - t2, "views": len(game),
            "trunk_restarts": len(ended), "R_mean": float(R.mean()), "R_group_std": float(Rg.std(1).mean()),
            "kept_frac": float(keep.mean()),
            "R_dev_std": float(scale), "win_mean": float(win.mean()), "lead_mean": float(lead.mean()),
            "branch_over": float(br.over.mean()), "decisions": int(steps + br.steps.sum()),
            "opp_score": [round(float(x), 4) for x in self.opp_score],
            "opp_w": [round(float(x), 4) for x in self.opp_w],
            "opp_groups": [opp_stats[i] for i in range(len(self.opps))],
        }
        return batch, stats

    # -------- value-net retraining
    def _snapshot(self, res) -> None:
        """Both teams' packed views of every live trunk game, kept until the game ends."""
        from blackout_env.sim.batch import N_AGENT_COLS, H, W

        live = np.nonzero(~res.over)[0]
        if not len(live):
            return
        pk = np.zeros((len(live), 2, H, W), np.int16)   # per team: one packed plane
        tm = np.zeros((len(live), 2, 4), F32)
        am = np.zeros((len(live), 2, 10, N_AGENT_COLS), F32)
        res.bsim.render_packed(live, pk, tm, am)
        for j, b in enumerate(live):
            self.pending[int(b)].append((pk[j], tm[j], am[j]))

    def _label(self, ended, scores) -> None:
        for b in ended:
            rows = self.pending[int(b)]
            self.pending[int(b)] = []
            if not rows:
                continue
            lead0 = float(scores[b, 0] - scores[b, 1]) / 100.0
            for tm in (0, 1):
                lead = lead0 if tm == 0 else -lead0
                self.vbuf.add(np.stack([r[0][tm] for r in rows]), np.stack([r[1][tm] for r in rows]),
                              np.stack([r[2][tm] for r in rows]), lead, int(np.sign(lead)))

    def train_value(self) -> dict:
        from blackout_env.value.model import value_loss

        c, buf, net = self.cfg, self.vbuf, self.leaf.net
        if buf.size < c.value_min_rows:
            return {"v_rows": buf.size}
        net.train()
        tot = {"v_loss": 0.0, "v_final": 0.0, "v_ce": 0.0}
        for _ in range(c.value_steps):
            idx = buf.sample(self.rng, c.value_batch, c.value_recent, c.value_replay)
            to = lambda x: torch.from_numpy(x[idx]).to(self.dev)
            g, t, a = unpack_graphic(to(buf.g)), to(buf.t), to(buf.a)
            out = net(g, t, a)
            hh = out.shape[1] - 4
            tg = {"delta": torch.zeros(len(idx), hh, device=self.dev), "delta_mask": torch.zeros(len(idx), hh, device=self.dev),
                  "final": to(buf.final), "result": to(buf.result)}
            loss, st = value_loss(out, tg, t[:, 0] - t[:, 1])
            self.vopt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.max_grad_norm)
            self.vopt.step()
            tot["v_loss"] += loss.item(); tot["v_final"] += st["final"].item(); tot["v_ce"] += st["ce"].item()
        net.eval()
        return {**{k: v / c.value_steps for k, v in tot.items()}, "v_rows": buf.size}

    # -------- update
    def _logits(self, own, g, t, a):
        g = unpack_graphic(g) if g.dim() == 3 else g
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.cfg.update_fp16 and self.dev.type == "cuda"):
            q, _ = own(g, t, a)
        return q.float() / self.cfg.temperature

    def update(self, batch):
        c = self.cfg
        n = len(batch["act"])
        if n == 0:   # every group dropped as flat
            return {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0, "grad_norm": 0.0}
        to = lambda x: torch.from_numpy(x).to(self.dev)
        g, t, a = to(batch["g"]), to(batch["t"]), to(batch["a"])
        act, adv = to(batch["act"]).long(), to(batch["adv"])
        with torch.no_grad():  # old log-probs on the training path (fp32), ref logits
            old_lp, ref_lp = [], []
            for s in range(0, n, c.minibatch):
                sl = slice(s, s + c.minibatch)
                lp = F.log_softmax(self._logits(self.own, g[sl], t[sl], a[sl]), -1)
                old_lp.append(lp.gather(-1, act[sl].unsqueeze(-1)).squeeze(-1))
                ref_lp.append(F.log_softmax(self._logits(self.ref, g[sl], t[sl], a[sl]), -1))
            old_lp, ref_lp = torch.cat(old_lp), torch.cat(ref_lp)
        st = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0, "grad_norm": 0.0, "n_mb": 0}
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, c.minibatch):
                mb = perm[s:s + c.minibatch]
                lp_all = F.log_softmax(self._logits(self.own, g[mb], t[mb], a[mb]), -1)   # [m, 5, W]
                lp = lp_all.gather(-1, act[mb].unsqueeze(-1)).squeeze(-1)                 # [m, 5]
                ratio = torch.exp(lp - old_lp[mb])
                A = adv[mb].unsqueeze(-1)
                pg = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                p = lp_all.exp()
                kl = (p * (lp_all - ref_lp[mb])).sum(-1).mean()
                loss = pg + c.beta * kl
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    st["loss"] += loss.item(); st["pg"] += pg.item(); st["kl"] += kl.item()
                    st["ent"] += -(p * lp_all).sum(-1).mean().item()
                    st["clipfrac"] += ((ratio - 1).abs() > c.clip).float().mean().item()
                    st["grad_norm"] += gn.item(); st["n_mb"] += 1
        m = max(st.pop("n_mb"), 1)
        return {k: v / m for k, v in st.items()}

    def step(self):
        batch, cs = self.collect()
        t0 = time.perf_counter()
        us = self.update(batch)
        us["update_s"] = time.perf_counter() - t0
        self.iter += 1
        if self.cfg.ref_reset_every and self.iter % self.cfg.ref_reset_every == 0:
            self.ref.net.load_state_dict(self.net.state_dict())
        if self.cfg.value_train and self.iter % self.cfg.value_every == 0:
            t0 = time.perf_counter()
            us.update(self.train_value())
            us["value_s"] = time.perf_counter() - t0
        return {"iter": self.iter, **cs, **us}

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"policy_state": self.net.state_dict(),
                    "grpo": {**vars(self.cfg), "iter": self.iter, "opp_score": self.opp_score.tolist(), **(extra or {})}}, path)
        if self.cfg.value_train:
            from blackout_env.value.model import save_value_net

            save_value_net(path.replace(".pt", "_value.pt"), self.leaf.net,
                           {"init": self.cfg.value, "grpo_iter": self.iter, "rows": self.vbuf.size})


class ValueBuffer:
    """Ring buffer of labelled trunk views (packed graphic, team_state, agents, final lead /100, result)."""

    def __init__(self, cap: int):
        from blackout_env.sim.batch import N_AGENT_COLS, H, W

        self.cap, self.n, self.size = cap, 0, 0   # n = rows ever written
        self.g = np.zeros((cap, H, W), np.int16)
        self.t = np.zeros((cap, 4), F32)
        self.a = np.zeros((cap, 10, N_AGENT_COLS), F32)
        self.final = np.zeros(cap, F32)
        self.result = np.zeros(cap, np.int64)

    def add(self, g, t, a, final: float, result: int) -> None:
        for j in range(len(g)):
            i = self.n % self.cap
            self.g[i], self.t[i], self.a[i], self.final[i], self.result[i] = g[j], t[j], a[j], final, result
            self.n += 1
        self.size = min(self.n, self.cap)

    def sample(self, rng, k: int, recent: int, replay: float) -> np.ndarray:
        """Row indices: ``1 - replay`` of them from the newest ``recent`` rows, the rest from older ones."""
        new = min(recent, self.size)
        n_old = int(round(k * replay)) if self.size > new else 0
        back = rng.integers(0, new, k - n_old)                        # 0 = newest
        if n_old:
            back = np.concatenate([back, rng.integers(new, self.size, n_old)])
        return (self.n - 1 - back) % self.cap


# ---------------------------------------------------------------------------------- evaluation

def eval_matches(net_or_path, opp_spec: str, seeds, device: str, horizon: int = 11000):
    """Full official-rules matches, greedy learner vs ``opp_spec``, every seed from both sides.
    Returns (win rate with draws as 1/2, mean margin, se of margin)."""
    pol = (NetPolicy.from_checkpoint(net_or_path, device=device) if isinstance(net_or_path, str)
           else NetPolicy(net_or_path, device=device))
    opp = make_opponent_policy(opp_spec, device)
    seeds = list(seeds)
    roots = [SimState.from_seed(s) for s in seeds]
    margins = []
    for side in (0, 1):
        pa, pb = (pol, opp) if side == 0 else (opp, pol)
        res = BatchRollout(pa, pb, packed=True).run(roots, horizon)
        lead = res.lead(side)
        margins.append(lead)
    m = np.concatenate(margins).astype(np.float64)
    wr = float(((m > 0) + 0.5 * (m == 0)).mean())
    return wr, float(m.mean()), float(m.std(ddof=1) / math.sqrt(len(m)))
