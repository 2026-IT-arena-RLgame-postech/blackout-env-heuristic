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
                 fp16: bool = True, seed: int = 0):
        self.own, self.device = own, torch.device(device)
        self.sample_steps, self.temperature, self.fp16 = int(sample_steps), float(temperature), fp16
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        self.t = np.zeros(0, np.int64)
        self.records: list = []

    def reset(self, n: int) -> None:
        self.t = np.zeros(n, np.int64)
        self.records = []

    @torch.no_grad()
    def _forward(self, g, t, a):
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
            if samp.any():
                rows = torch.from_numpy(np.nonzero(samp)[0]).to(self.device)
                p = torch.softmax(q[rows] / self.temperature, -1)
                pick = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=self.rng).view(p.shape[:-1])
                chosen[rows] = pick
                r = rows.cpu().numpy()
                self.records.append((idx[r].copy(), g[r].copy(), ts[r].copy(), ag[r].copy(),
                                     pick.cpu().numpy().astype(np.int8)))
            onehot = F.one_hot(chosen, q.shape[-1]).float().cpu().numpy()
            team_b = np.full(n, tm == 1)
            out[tm] = decode_moves(onehot, off.cpu().numpy(), team_b)
            self.t[idx] += 1
        return out


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


def make_opponent_policy(spec: str, device: str, seed: int = 0) -> BatchPolicy:
    """``net:<checkpoint>`` -> batched ``NetPolicy`` (fp16 on CUDA); anything else -> ``HeuristicPolicy``
    (a ``make_opponent`` descriptor; slow, per game in Python)."""
    if spec.startswith("net:"):
        return NetPolicy.from_checkpoint(spec[4:], device=device, fp16=device.startswith("cuda"))
    return HeuristicPolicy(spec, seed)


def forget(pol: BatchPolicy, games) -> None:
    """Drop a heuristic opponent's per-game state for restarted games."""
    if isinstance(pol, HeuristicPolicy):
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
        self.opps = [make_opponent_policy(s, cfg.device, cfg.seed + 7 * i) for i, s in enumerate(cfg.opponents)]
        w = np.asarray(cfg.opp_weights if cfg.opp_weights else [1.0] * len(self.opps), np.float64)
        self.opp_w = w / w.sum()
        g = cfg.groups
        self.trunk = [self._fresh_seed() for _ in range(g)]
        self.trunk_opp = self.rng.choice(len(self.opps), g, p=self.opp_w)
        self.trunk_side = self.rng.integers(0, 2, g)
        self.iter = 0

    def _fresh_seed(self) -> SimState:
        return SimState.from_seed(int(self.rng.integers(0, 2**31 - 1)))

    def _rollout(self, roots, learner, opp_of, side, horizon, reseeds=None):
        pa = Seats(0, learner, self.opps, opp_of, side)
        pb = Seats(1, learner, self.opps, opp_of, side)
        return BatchRollout(pa, pb, packed=True).run(roots, horizon, reseeds=reseeds)

    # -------- collection
    def collect(self):
        c = self.cfg
        t0 = time.perf_counter()
        greedy = LearnerPolicy(self.own, self.dev, 0)
        res = self._rollout(self.trunk, greedy, self.trunk_opp, self.trunk_side, c.trunk_step)
        roots = res.states()
        # restart finished trunk games (fresh seed, opponent, side); their heuristic state is dropped
        ended = np.nonzero(res.over)[0]
        live = np.nonzero(~res.over)[0]
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
        learner = LearnerPolicy(self.own, self.dev, c.sample_steps, c.temperature, seed=int(self.rng.integers(2**31)))
        br = self._rollout(branch_roots, learner, opp_of, side, c.sample_steps + c.leaf_steps, reseeds)
        t2 = time.perf_counter()
        win, lead = self.leaf(br.bsim, side)
        R = win + c.lam * np.tanh(lead / c.scale)
        Rg = R.reshape(-1, k)
        dev = Rg - Rg.mean(1, keepdims=True)
        scale = dev.std() + 1e-8
        adv = (dev / scale).reshape(-1)
        recs = learner.records
        game = np.concatenate([r[0] for r in recs])
        batch = {
            "g": np.concatenate([r[1] for r in recs]), "t": np.concatenate([r[2] for r in recs]),
            "a": np.concatenate([r[3] for r in recs]), "act": np.concatenate([r[4] for r in recs]),
            "adv": adv[game].astype(np.float32),
        }
        t3 = time.perf_counter()
        stats = {
            "trunk_s": t1 - t0, "branch_s": t2 - t1, "leaf_s": t3 - t2, "views": len(game),
            "trunk_restarts": len(ended), "R_mean": float(R.mean()), "R_group_std": float(Rg.std(1).mean()),
            "R_dev_std": float(scale), "win_mean": float(win.mean()), "lead_mean": float(lead.mean()),
            "branch_over": float(br.over.mean()), "decisions": int(res.steps.sum() + br.steps.sum()),
        }
        return batch, stats

    # -------- update
    def _logits(self, own, g, t, a):
        g = unpack_graphic(g) if g.dim() == 3 else g
        q, _ = own(g, t, a)
        return q.float() / self.cfg.temperature

    def update(self, batch):
        c = self.cfg
        n = len(batch["act"])
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
        return {"iter": self.iter, **cs, **us}

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"policy_state": self.net.state_dict(), "grpo": {**vars(self.cfg), "iter": self.iter, **(extra or {})}}, path)


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
