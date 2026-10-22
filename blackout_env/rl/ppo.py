"""PPO baseline for the GRPO A/B (docs/rl_finetune_plan_ko.md, run R1).

Same simulator, opponents, reward, KL anchor and update rule as ``grpo.GRPOTrainer``; what differs is
where the advantage comes from:

* ``games`` trunk games run on-policy: the learner *samples every* decision, ``segment`` decisions per
  iteration (a finished game restarts from a fresh seed / opponent / side, like GRPO's trunk);
* the reward is sparse: ``sign(lead) + lam * tanh(lead / scale)`` on the step the match ends, 0 before;
* a separate critic (a copy of the S1/S2 value net) reads ``V(s) = (P(win) - P(loss)) + lam *
  tanh(final / scale)`` from its heads (the same quantity GRPO's leaf reads), bootstraps the segment
  end, and is trained on the GAE(gamma = 1, ``gae_lambda``) returns (Huber), the first
  ``critic_warmup`` iterations critic only.  ``critic_head='hlgauss'``: a categorical pmf over the return (HL-Gauss CE),
  V = its mean (see reports/rl/ppo_r2_setup.md).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from blackout_env.sim.batch import N_AGENT_COLS, BatchRollout, H, W, unpack_graphic
from blackout_env.value.model import REG_SCALE, load_value_net

from .grpo import GRPOConfig, GRPOTrainer, LearnerPolicy, Seats, forget


@dataclass
class PPOConfig(GRPOConfig):
    games: int = 256           # on-policy trunk games
    segment: int = 128         # decisions per game per iteration
    gae_lambda: float = 0.95
    critic_lr: float = 1e-5
    critic_warmup: int = 5     # iterations with critic updates only
    vf_huber: float = 0.1
    critic_fp16: bool = False  # (opt-in, changes numerics) fp16 autocast + GradScaler for the critic (update epochs and value reads)
    overlap_heur: bool = True  # (bit-identical) heuristic opponents overlap the network forwards
    # distributional critic ("Stop Regressing", Farebrother et al. 2024): categorical pmf over the *return*, HL-Gauss CE loss
    critic_head: str = "reg"   # "reg" (Huber on the S2 heads' V) | "hlgauss"
    critic_bins: int = 101
    critic_vmin: float = -1.4  # returns lie in about [-(1 + lam), 1 + lam] = [-1.2, 1.2]
    critic_vmax: float = 1.4
    critic_sigma: float = 0.75  # HL-Gauss sigma in bin widths
    critic_prefit_epochs: int = 4   # one-off fit of the new head to the S2 heads' V on the first batch (hlgauss only)
    critic_prefit_lr: float = 1e-3


class _HLLogits(torch.nn.Module):
    """(autocast +) critic forward returning the fp32 bin logits; autocast inside, like grpo._AutocastFwd."""

    def __init__(self, mod, fp16: bool, device_type: str):
        super().__init__()
        self.mod, self.fp16, self.dt = mod, fp16, device_type

    def forward(self, g, t, a):
        with torch.autocast(self.dt, dtype=torch.float16, enabled=self.fp16):
            lg = self.mod.forward_full(g, t, a)[1]
        return lg.float()


class PPOTrainer(GRPOTrainer):
    def __init__(self, cfg: PPOConfig):
        cfg.groups = cfg.games
        super().__init__(cfg)
        self.critic = load_value_net(cfg.value, device=self.dev)
        self.critic.train(False)
        self.hl = cfg.critic_head == "hlgauss"
        if cfg.critic_head not in ("reg", "hlgauss"):
            raise ValueError(cfg.critic_head)
        self._prefit_done = not self.hl
        self._hl_ent = float("nan")
        if self.hl:
            # reuse ValueNet's optional bin head (a Linear on the head's hidden features, fed by forward_full); the centres
            # buffer is ours (asymmetric ranges allowed).  The new head starts untrained -> _prefit_head in the first collect.
            cr = self.critic
            cr.n_bins = cfg.critic_bins
            cr.bin_out = torch.nn.Linear(cr.head[-1].in_features, cfg.critic_bins).to(self.dev)
            e = torch.linspace(cfg.critic_vmin, cfg.critic_vmax, cfg.critic_bins + 1, device=self.dev)
            cr.register_buffer("bin_centers", (e[:-1] + e[1:]) / 2, persistent=False)
            self.hl_edges = e
            self.hl_sigma = cfg.critic_sigma * (cfg.critic_vmax - cfg.critic_vmin) / cfg.critic_bins
        self.copt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
        self.cscaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10**9, enabled=bool(cfg.critic_fp16) and self.dev.type == "cuda")

    def _hl_target(self, x):
        """HL-Gauss target pmf [B, bins] of returns x [B]: N(x, sigma^2) mass per bin, renormalised over the support."""
        x = x.float().clamp(self.cfg.critic_vmin, self.cfg.critic_vmax)
        cdf = torch.special.erf((self.hl_edges[None] - x[:, None]) / (self.hl_sigma * 2 ** 0.5))
        mass = (cdf[:, 1:] - cdf[:, :-1]) / 2
        return mass / mass.sum(1, keepdim=True).clamp_min(1e-12)

    def _hl_logits(self, g, t, a):
        """fp32 bin logits [B, bins] (compiled in training for full-size minibatches, like the regression critic)."""
        c = self.cfg
        if self.critic.training and c.update_compile and self.dev.type == "cuda" and g.shape[0] == c.minibatch:
            key = (id(self.critic), c.critic_fp16, "hl")
            if key not in self._compiled:
                self._compiled[key] = torch.compile(_HLLogits(self.critic, c.critic_fp16, self.dev.type), dynamic=False)
            return self._compiled[key](g, t, a)
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=c.critic_fp16 and self.dev.type == "cuda"):
            lg = self.critic.forward_full(g, t, a)[1]
        return lg.float()

    def _value(self, g, t, a, orig=False):
        """Critic V for canonical own-team views (differentiable).  hlgauss: E[pmf] over the return bins (``orig=True``:
        the S2 heads' V instead, used by the pre-fit)."""
        if self.hl and not orig:
            g = unpack_graphic(g) if g.dim() == 3 else g
            lg = self._hl_logits(g, t, a)
            return (torch.softmax(lg, -1) * self.critic.bin_centers).sum(-1), lg
        v = self._value_reg(g, t, a)
        return (v, None) if self.hl else v

    def _value_reg(self, g, t, a):
        g = unpack_graphic(g) if g.dim() == 3 else g
        if self.critic.training and self.cfg.update_compile and self.dev.type == "cuda" and g.shape[0] == self.cfg.minibatch:
            out = self._compiled_fwd(self.critic, self.cfg.critic_fp16, logits=False)(g, t, a)
        else:
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.cfg.critic_fp16 and self.dev.type == "cuda"):
                out = self.critic(g, t, a)
            out = out.float()
        h = out.shape[1] - 4
        lead = (t[:, 0] - t[:, 1]).float() * 100.0          # team_state scores are /100
        final = lead + out[:, h] / REG_SCALE * 100.0
        p = torch.softmax(out[:, h + 1:], -1)
        return (p[:, 2] - p[:, 0]) + self.cfg.lam * torch.tanh(final / self.cfg.scale)

    @torch.no_grad()
    def _values_np(self, g, t, a, bs=4096, orig=False):
        out, ent = [], []
        for s in range(0, len(g), bs):
            to = lambda x: torch.from_numpy(np.ascontiguousarray(x[s:s + bs])).to(self.dev)
            v = self._value(to(g), to(t), to(a), orig=orig)
            if self.hl:
                v, lg = v
                if lg is not None:
                    lp = torch.log_softmax(lg, -1)
                    ent.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
            out.append(v.cpu().numpy())
        self._hl_ent_last = float(np.concatenate(ent).mean()) if ent else float("nan")
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _prefit_head(self, g, t, a):
        """One-off: fit the fresh bin head (critic body frozen) with HL-Gauss CE to the S2 heads' V of the first batch's
        views, so the GAE of that very batch (and the policy update after the warm-up) starts from a calibrated critic."""
        c = self.cfg
        orig = self._values_np(g, t, a, orig=True)
        tgt_v = torch.from_numpy(orig).to(self.dev)
        gt, tt, at = (torch.from_numpy(np.ascontiguousarray(x)).to(self.dev) for x in (g, t, a))
        params = list(self.critic.bin_out.parameters())
        for q in self.critic.parameters():
            q.requires_grad_(False)
        for q in params:
            q.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=c.critic_prefit_lr)
        n, ce_hist = len(tgt_v), []
        for _ in range(c.critic_prefit_epochs):
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, c.minibatch):
                mb = perm[s:s + c.minibatch]
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=c.critic_fp16 and self.dev.type == "cuda"):
                    lg = self.critic.forward_full(unpack_graphic(gt[mb]) if gt.dim() == 3 else gt[mb], tt[mb], at[mb])[1]
                loss = -(self._hl_target(tgt_v[mb]) * torch.log_softmax(lg.float(), -1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ce_hist.append(loss.item())
        for q in self.critic.parameters():
            q.requires_grad_(True)
        self._prefit_done = True
        pv = self._values_np(g, t, a)
        self.prefit_stats = {"prefit_ce_first": ce_hist[0], "prefit_ce_last": ce_hist[-1],
                             "prefit_v_rmse": float(np.sqrt(np.mean((pv - orig) ** 2))),
                             "prefit_v_std": float(orig.std())}

    def collect(self):
        c = self.cfg
        t0 = time.perf_counter()
        n = len(self.trunk)
        learner = LearnerPolicy(self.own, self.dev, 10**9, c.temperature, seed=int(self.rng.integers(2**31)),
                                graphs=self.learner_gf)
        res = self._rollout(self.trunk, learner, self.trunk_opp, self.trunk_side, c.segment)
        t1 = time.perf_counter()
        recs = learner.records
        game = np.concatenate([r[0] for r in recs])
        g = np.concatenate([r[1] for r in recs]); ts = np.concatenate([r[2] for r in recs])
        ag = np.concatenate([r[3] for r in recs]); act = np.concatenate([r[4] for r in recs])
        if not self._prefit_done:
            self._prefit_head(g, ts, ag)
        v = self._values_np(g, ts, ag)
        v_ent = self._hl_ent_last
        # bootstrap: the learner's view of every unfinished game's final state
        side = self.trunk_side
        bs = res.bsim
        live = np.nonzero(~bs.over)[0]
        boot = np.zeros(n, np.float64)
        if len(live):
            pk = np.zeros((len(live), 2, H, W), np.int16)
            tm = np.zeros((len(live), 2, 4), np.float32)
            am = np.zeros((len(live), 2, 10, N_AGENT_COLS), np.float32)
            bs.render_packed(live, pk, tm, am)
            r = np.arange(len(live))
            s_ = side[live]
            boot[live] = self._values_np(pk[r, s_], tm[r, s_], am[r, s_])
        lead = (bs.scores[:, 0] - bs.scores[:, 1]).astype(np.float64)
        lead = np.where(side == 0, lead, -lead)
        final_r = np.sign(lead) + c.lam * np.tanh(lead / c.scale)
        # GAE per game (records of a game are in time order: one learner call per decision)
        adv = np.zeros(len(game), np.float64)
        ret = np.zeros(len(game), np.float64)
        for b in range(n):
            rows = np.nonzero(game == b)[0]
            if not len(rows):
                continue
            vb = v[rows].astype(np.float64)
            nxt = final_r[b] if bs.over[b] else boot[b]   # value after the last step (terminal: reward, V=0)
            gae = 0.0
            for j in range(len(rows) - 1, -1, -1):
                v_next = nxt if j == len(rows) - 1 else vb[j + 1]
                delta = v_next - vb[j]                      # r = 0 except folded into the terminal v_next
                gae = delta + c.gae_lambda * gae
                adv[rows[j]] = gae
                ret[rows[j]] = gae + vb[j]
        ended = np.nonzero(bs.over)[0]
        # PFSP: per-opponent EMA of the learner's result on the games that ended (no branch leaves here)
        score = (np.sign(lead) + 1.0) / 2.0
        for i in range(len(self.opps)):
            m = ended[self.trunk_opp[ended] == i]
            if len(m):
                self.opp_score[i] += c.pfsp_ema * (float(score[m].mean()) - self.opp_score[i])
        self.opp_w = self._pfsp_weights()
        roots = res.states()
        for b in ended:
            roots[b] = self._fresh_seed()
            self.trunk_opp[b] = self.rng.choice(len(self.opps), p=self.opp_w)
            self.trunk_side[b] = self.rng.integers(0, 2)
        for p in self.opps:
            forget(p, ended)
        self.trunk = roots
        a_std = adv.std() + 1e-8
        ev = 1.0 - float(np.var(ret - v) / (np.var(ret) + 1e-12))
        batch = {"g": g, "t": ts, "a": ag, "act": act, "adv": (adv / a_std).astype(np.float32),
                 "ret": ret.astype(np.float32)}
        stats = {"trunk_s": t1 - t0, "leaf_s": time.perf_counter() - t1, "views": len(game),
                 "trunk_restarts": len(ended), "decisions": int(res.steps.sum()), "v_mean": float(v.mean()), "adv_std": float(a_std),
                 "ret_mean": float(ret.mean()), "ret_std": float(ret.std()), "expl_var": ev, "abs_adv": float(np.abs(adv).mean()),
                 "pmf_entropy": v_ent, **getattr(self, "prefit_stats", {}),
                 "ended_win": float((final_r[ended] > 0).mean()) if len(ended) else float("nan"),
                 "ended_lead": float(lead[ended].mean()) if len(ended) else float("nan"),
                 "opp_score": [round(float(x), 4) for x in self.opp_score],
                 "opp_w": [round(float(x), 4) for x in self.opp_w]}
        return batch, stats

    def update(self, batch):
        c = self.cfg
        n = len(batch["act"])
        to = lambda x: torch.from_numpy(x).to(self.dev)
        g, t, a, ret = to(batch["g"]), to(batch["t"]), to(batch["a"]), to(batch["ret"])
        vl, nv = 0.0, 0
        self.critic.train()
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, c.minibatch):
                mb = perm[s:s + c.minibatch]
                if self.hl:
                    _, lg = self._value(g[mb], t[mb], a[mb])
                    loss = -(self._hl_target(ret[mb]) * torch.log_softmax(lg, -1)).sum(-1).mean()   # fp32 log_softmax
                else:
                    loss = F.smooth_l1_loss(self._value(g[mb], t[mb], a[mb]), ret[mb], beta=c.vf_huber)
                self.copt.zero_grad(set_to_none=True)
                if self.cscaler.is_enabled():   # loss scaling: fp16 activation gradients of a ~1e-2 Huber loss underflow
                    self.cscaler.scale(loss).backward()
                    self.cscaler.unscale_(self.copt)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.cscaler.step(self.copt)
                    self.cscaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.copt.step()
                vl += loss.item(); nv += 1
        self.critic.eval()
        st = {"vf_loss": vl / max(nv, 1)}
        if self.iter < c.critic_warmup:
            return st
        return {**st, **super().update(batch)}

    def save(self, path: str, extra: dict | None = None) -> None:
        super().save(path, extra)
        torch.save(self.critic.state_dict(), path.replace(".pt", "_critic.pt"))
