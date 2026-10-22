"""PPO baseline for the GRPO A/B (docs/rl_finetune_plan_ko.md, run R1).

Same simulator, opponents, reward, KL anchor and update rule as ``grpo.GRPOTrainer``; what differs is
where the advantage comes from:

* ``games`` trunk games run on-policy: the learner *samples every* decision, ``segment`` decisions per
  iteration (a finished game restarts from a fresh seed / opponent / side, like GRPO's trunk);
* the reward is sparse: ``sign(lead) + lam * tanh(lead / scale)`` on the step the match ends, 0 before;
* a separate critic (a copy of the S1/S2 value net) reads ``V(s) = (P(win) - P(loss)) + lam *
  tanh(final / scale)`` from its heads (the same quantity GRPO's leaf reads), bootstraps the segment
  end, and is trained on the GAE(gamma = 1, ``gae_lambda``) returns (Huber), the first
  ``critic_warmup`` iterations critic only.
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


class PPOTrainer(GRPOTrainer):
    def __init__(self, cfg: PPOConfig):
        cfg.groups = cfg.games
        super().__init__(cfg)
        self.critic = load_value_net(cfg.value, device=self.dev)
        self.critic.train(False)
        self.copt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
        self.cscaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10**9, enabled=bool(cfg.critic_fp16) and self.dev.type == "cuda")

    def _value(self, g, t, a):
        """Critic V for canonical own-team views (differentiable)."""
        g = unpack_graphic(g) if g.dim() == 3 else g
        if self.critic.training and self.cfg.update_compile and self.dev.type == "cuda":
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
    def _values_np(self, g, t, a, bs=4096):
        out = []
        for s in range(0, len(g), bs):
            to = lambda x: torch.from_numpy(np.ascontiguousarray(x[s:s + bs])).to(self.dev)
            out.append(self._value(to(g), to(t), to(a)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

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
        v = self._values_np(g, ts, ag)
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
        batch = {"g": g, "t": ts, "a": ag, "act": act, "adv": (adv / a_std).astype(np.float32),
                 "ret": ret.astype(np.float32)}
        stats = {"trunk_s": t1 - t0, "leaf_s": time.perf_counter() - t1, "views": len(game),
                 "trunk_restarts": len(ended), "decisions": int(res.steps.sum()), "v_mean": float(v.mean()), "adv_std": float(a_std),
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
