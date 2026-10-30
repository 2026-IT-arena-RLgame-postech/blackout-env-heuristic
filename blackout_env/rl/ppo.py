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
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from blackout_env.sim.batch import N_AGENT_COLS, BatchRollout, H, W, unpack_graphic
from blackout_env.value.model import REG_SCALE, load_value_net

from . import league as lg
from .grpo import GRPOConfig, GRPOTrainer, LearnerPolicy, Seats, forget, make_opponent_policy


@dataclass
class PPOConfig(GRPOConfig):
    games: int = 256           # on-policy trunk games
    segment: int = 128         # decisions per game per iteration
    gae_lambda: float = 0.95
    critic_lr: float = 1e-5
    critic_warmup: int = 5     # iterations with critic updates only
    critic_init: str = ""      # warm-start the critic from a PPO run's <ckpt>_critic.pt (state dict) instead of the bare value net
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
    critic_read_batch: int = 4096      # rows per eager critic forward when reading V (4096 peaks at ~2.4 GB of activations; fp16 batch-composition noise only)
    cuda_mem_fraction: float = 0.0     # >0: torch.cuda.set_per_process_memory_fraction (soft cap: the allocator frees cached blocks before it would exceed it)
    graph_shared_pool: bool = False    # (memory only, outputs identical) all CUDA-graph forwards (learner, net opponents, snapshots) share one graph memory pool
    # ---- league arm B (all off by default = old behaviour, bit-identical); see reports/rl/ppo_league_b_setup.md
    adv_center_opp: bool = False       # subtract each opponent's batch-mean advantage (then the global-std scaling)
    adv_center_min_games: int = 10     # ... only for opponents with >= this many (finished or active) games in the batch
    league: bool = False               # gated self-play snapshots as extra net opponents
    league_dir: str = ""               # snapshot checkpoints go here (rl_train: the run dir)
    league_gate: float = 0.85          # PFSP-weighted mean EMA score of the fixed-pool opponents must be >= this
    league_min_gap: int = 100          # iterations since the last snapshot
    league_min_start: int = 150        # iterations since the start
    league_max: int = 4                # max snapshots
    league_steps: list = field(default_factory=lambda: [0.05, 0.10, 0.20])   # share ramp per snapshot (of trunk games)
    league_ramp_every: int = 20        # min iterations between ramp steps of one snapshot
    league_ramp_min_games: int = 100   # finished games (lifetime) of the snapshot before it may ramp
    league_adv_ratio: float = 1.5      # max (adv std of the snapshot's rows) / (adv std of the fixed-pool rows), over league_window its
    league_ev_min: float = 0.4         # min EV_mc over the snapshot's finished games
    league_window: int = 20            # iterations of the adv-std window
    league_anchor_min: float = 0.4     # fixed opponents keep >= this combined share
    league_warmup: int = 2             # critic-only iterations after each snapshot
    league_snap_graphs: bool = True    # CUDA-graph forwards for snapshots (each costs ~0.5 GB of graph outputs; False: eager, ~1 s/it each)
    league_ref_reset: bool = False     # KL anchor pi_ref <- the learner at each new snapshot (default: pi_ref stays the init policy)
    # ---- async actor-learner (off by default): the rollout of batch k+1 runs in a thread (on ``actor_device``, or a second CUDA
    # stream of ``device``) while batch k updates the learner.  1-iteration policy lag: batch k+1 is played by the weights after
    # update k-1; the PPO ratio uses those behaviour weights (``_old_own``), values/GAE come from the actor's critic copy.
    async_collect: bool = False
    actor_device: str = ""             # "" = same device as the learner (overlap via a second CUDA stream)
    async_decoupled: bool = True       # decoupled PPO: clip vs the proximal policy (weights at update start) x truncated IS pi_prox/pi_beh
    async_is_cap: float = 2.0          # truncation of that importance weight


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
        if cfg.cuda_mem_fraction > 0 and torch.device(cfg.device).type == "cuda":
            torch.cuda.set_per_process_memory_fraction(cfg.cuda_mem_fraction)
        self.snaps: list[dict] = []        # league snapshots: {path, added, level, last_step, win: deque of (n, sum, sumsq)}
        self.n_fixed = len(cfg.opponents)  # the first n_fixed entries of self.opps are the fixed pool (anchors)
        super().__init__(cfg)
        n_o = len(self.opps)
        self.fin = np.zeros((n_o, 6))      # per opponent, finished games: [games, rows, sum R, sum R^2, sum e, sum e^2], e = R - V
        self.gv = np.zeros((cfg.games, 3)) # per trunk game since its (re)start: [rows, sum V, sum V^2]
        self.fixed_win: deque = deque(maxlen=max(cfg.league_window, 1))   # per iteration (n, sum, sumsq) of the fixed pool's used adv
        self.last_snap_iter = 0
        self.warm_until = 0                # policy updates are skipped while iter < max(critic_warmup, warm_until)
        self._events: list = []
        self._gpool = None
        if cfg.graph_shared_pool and self.dev.type == "cuda" and cfg.graphs:
            self._gpool = torch.cuda.graph_pool_handle()
            for gf in [self.learner_gf] + [getattr(p, "_gf", None) for p in self.opps]:
                if gf is not None:
                    gf._pool = self._gpool
        self.critic = load_value_net(cfg.value, device=self.dev)
        if cfg.critic_init:
            self.critic.load_state_dict(torch.load(cfg.critic_init, map_location=self.dev, weights_only=True))
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
        self._async = bool(cfg.async_collect)
        self.adev = torch.device(cfg.actor_device or cfg.device)
        if self._async:
            self._init_actor()
        self.cscaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10**9, enabled=bool(cfg.critic_fp16) and self.dev.type == "cuda")

    # ------------------------------------------------------------------ async actor-learner
    def _init_actor(self) -> None:
        """Actor-side copies (policy + its CUDA graphs, critic) on ``adev`` and a behaviour copy of the policy on ``dev``."""
        c = self.cfg
        assert not self.hl and c.fine_bins == 0 and not c.offset_rl and not c.value_train and not c.joint, \
            "async_collect: reg critic, plain head, no value_train / joint"
        from blackout_env.train.distill.inference import _OwnQ

        self._act_net = copy.deepcopy(self.net).to(self.adev).eval().requires_grad_(False)
        self._act_own = _OwnQ(self._act_net)
        self._act_critic = copy.deepcopy(self.critic).to(self.adev).eval().requires_grad_(False)
        self.beh = _OwnQ(copy.deepcopy(self.net).eval().requires_grad_(False))
        self._act_gf, self._astream = None, None
        if self.adev.type == "cuda":
            from blackout_env.sim.graphfwd import GraphForward

            self._astream = torch.cuda.Stream(device=self.adev)
            self._lstream = torch.cuda.Stream(device=self.dev) if self.dev.type == "cuda" else None
            if c.graphs:
                self._act_gf = GraphForward(self._act_own, self.adev, angle=True, max_batch=2048)
                with torch.cuda.device(self.adev):
                    apool = torch.cuda.graph_pool_handle()
                for gf in [self._act_gf] + [getattr(p, "_gf", None) for p in self.opps]:
                    if gf is not None:
                        gf._pool = apool
                self._apool = apool
        self._pending = None   # (batch, collect stats, behaviour state dict) of the batch collected during the last update

    def _sync_actor(self) -> dict:
        """Actor copies <- the learner's current weights (in place: captured CUDA graphs stay valid); returns those weights
        (a copy on ``dev``) = the behaviour policy of the batch the actor collects next."""
        sd = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self._act_net.load_state_dict(sd)
        self._act_critic.load_state_dict(self.critic.state_dict())
        if self.adev.type == "cuda":
            torch.cuda.synchronize(self.adev)
        return sd

    def _collect_actor(self):
        if self._astream is None:
            return self.collect()
        with torch.cuda.device(self.adev), torch.cuda.stream(self._astream):
            out = self.collect()
        self._astream.synchronize()
        return out

    def _step_async(self) -> dict:
        import threading

        if self._pending is None:   # first iteration: collect synchronously
            beh = self._sync_actor()
            batch, cs = self._collect_actor()
        else:
            batch, cs, beh = self._pending
        beh_next = self._sync_actor()   # the weights after the last update play the next batch
        box: dict = {}

        def run():
            try:
                box["out"] = self._collect_actor()
            except BaseException as e:   # re-raised in the main thread
                box["err"] = e

        th = threading.Thread(target=run, name="ppo-actor", daemon=True)
        th.start()
        self.beh.net.load_state_dict(beh)
        self._old_own = self.beh
        t0 = time.perf_counter()
        try:
            if getattr(self, "_lstream", None) is not None:
                with torch.cuda.stream(self._lstream):
                    us = self.update(batch)
                self._lstream.synchronize()
            else:
                us = self.update(batch)
        finally:
            self._old_own = None
        t1 = time.perf_counter()
        th.join()
        if "err" in box:
            raise box["err"]
        us["update_s"] = t1 - t0
        us["actor_wait_s"] = time.perf_counter() - t1   # time the learner waited for the actor after its update
        nb, ncs = box["out"]
        self._pending = (nb, ncs, beh_next)
        self.iter += 1
        return {"iter": self.iter, **cs, **us}

    # ------------------------------------------------------------------ league
    def _pfsp_weights(self, n: int | None = None) -> np.ndarray:
        if n is not None or not self.snaps:
            return super()._pfsp_weights(n)
        shares = lg.snap_shares([s["level"] for s in self.snaps], self.cfg.league_steps, self.cfg.league_anchor_min)
        return lg.league_weights(super()._pfsp_weights(self.n_fixed), shares)

    def gate_score(self) -> float:
        """PFSP-weighted mean over the fixed-pool opponents of their EMA score (``opp_score``, learner win share with draws
        as 1/2, updated per iteration from the games that ended), weights = base weight x PFSP factor renormalised over the
        fixed pool (= how the fixed pool is being sampled)."""
        return lg.fixed_gate(self.opp_score[:self.n_fixed], super()._pfsp_weights(self.n_fixed))

    def _add_snapshot(self) -> dict:
        """Freeze the current learner as a new net opponent (checkpoint file in ``league_dir``), share level 0."""
        c = self.cfg
        d = c.league_dir or "."
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"snap{len(self.snaps) + 1}_it{self.iter}.pt")
        ck = {"policy_state": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
              "league": {"iter": self.iter, "index": len(self.snaps) + 1}}
        if self.fine is not None:   # the snapshot plays with the fine head (NetPolicy.from_checkpoint reads it)
            ck["fine_head_state"] = {k: v.detach().cpu() for k, v in self.fine.state_dict().items()}
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)
        return self._attach_snapshot(path, level=0, added=self.iter, last_step=self.iter)

    def _attach_snapshot(self, path: str, level: int, added: int, last_step: int) -> dict:
        c = self.cfg
        pol = make_opponent_policy("net:" + path, c.actor_device or c.device,
                                   graphs=bool(c.graphs) and self.dev.type == "cuda" and c.league_snap_graphs)
        pool = getattr(self, "_apool", None) or self._gpool
        if pool is not None and getattr(pol, "_gf", None) is not None:
            pol._gf._pool = pool
        self.opps.append(pol)
        self.branch_opps.append(pol)
        self.opp_score = np.append(self.opp_score, 0.5)
        self.fin = np.vstack([self.fin, np.zeros((1, 6))])
        sn = {"path": path, "added": added, "level": level, "last_step": last_step, "win": deque(maxlen=max(c.league_window, 1))}
        self.snaps.append(sn)
        self.opp_w = self._pfsp_weights()
        return sn

    def _league_post(self, out: dict) -> None:
        """After a step (iter already incremented): maybe advance ramp steps, maybe add a snapshot; logs into ``out``."""
        c = self.cfg
        gs = self.gate_score()
        out["gate_score"] = gs
        for j, sn in enumerate(self.snaps):
            ok, info = lg.ramp_ok(fin=self.fin[self.n_fixed + j], snap_adv_win=sn["win"], fixed_adv_win=self.fixed_win,
                                  it=self.iter, last_step_iter=sn["last_step"], level=sn["level"], n_levels=len(c.league_steps),
                                  every=c.league_ramp_every, min_games=c.league_ramp_min_games, max_ratio=c.league_adv_ratio,
                                  min_ev=c.league_ev_min)
            if ok:
                sn["level"] += 1
                sn["last_step"] = self.iter
                self._events.append({"ramp_step": j + 1, "level": sn["level"], "share": c.league_steps[sn["level"]], **info})
        if lg.gate_open(gs, self.iter, self.last_snap_iter, len(self.snaps), thr=c.league_gate, min_gap=c.league_min_gap,
                        min_start=c.league_min_start, max_snaps=c.league_max):
            sn = self._add_snapshot()
            self.last_snap_iter = self.iter
            self.warm_until = self.iter + c.league_warmup
            if c.league_ref_reset:
                self._sync_ref()
            self._events.append({"snapshot": len(self.snaps), "path": os.path.basename(sn["path"]), "gate_score": gs})
        self.opp_w = self._pfsp_weights()
        out["league_events"] = self._events
        out["n_snaps"] = len(self.snaps)
        out["snap_levels"] = [s["level"] for s in self.snaps]
        out["snap_ev_mc"] = [round(lg.ev_mc(self.fin[self.n_fixed + j]), 4) for j in range(len(self.snaps))]
        out["snap_games"] = [int(self.fin[self.n_fixed + j, 0]) for j in range(len(self.snaps))]
        out["opp_w_next"] = [round(float(x), 4) for x in self.opp_w]

    def step(self):
        self._events = []
        out = self._step_async() if self._async else super().step()
        if self.cfg.league:
            self._league_post(out)
        if self.dev.type == "cuda":
            out["cuda_collect_peak_mb"] = getattr(self, "_collect_peak", float("nan"))                       # allocated, rollouts + critic reads
            out["cuda_update_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20  # allocated, updates (+ league bookkeeping)
            out["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 2**20
            out["cuda_between_peak_mb"] = getattr(self, "_between_peak", float("nan"))
            torch.cuda.reset_peak_memory_stats()
        return out

    # ------------------------------------------------------------------ resume
    def save_resume(self, path: str, extra: dict | None = None) -> None:
        """Full trainer state (policy, critic, optimisers, scalers, iteration, PFSP scores, rng, league/snapshot state).
        Not saved: the trunk games in flight (they restart from fresh seeds) and the heuristic opponents' state.
        Snapshot weights live in their own files (``league_dir``)."""
        st = {"policy_state": self.net.state_dict(), "critic_state": self.critic.state_dict(),
              "opt": self.opt.state_dict(), "copt": self.copt.state_dict(),
              "scaler": self.scaler.state_dict(), "cscaler": self.cscaler.state_dict(),
              "iter": self.iter, "opp_score": self.opp_score.copy(), "fin": self.fin.copy(), "rng": self.rng.bit_generator.state,
              "last_snap_iter": self.last_snap_iter, "warm_until": self.warm_until, "prefit_done": self._prefit_done,
              "fixed_win": list(self.fixed_win),
              "snaps": [{"file": os.path.basename(s["path"]), "added": s["added"], "level": s["level"], "last_step": s["last_step"],
                         "win": list(s["win"])} for s in self.snaps],
              "extra": extra or {}}
        if self.cfg.league_ref_reset:
            st["ref_state"] = self.ref.net.state_dict()
        if self.fine is not None:
            st["fine_state"], st["fine_ref_state"] = self.fine.state_dict(), self.fine_ref.state_dict()
        if self.log_sigma is not None:
            st["log_sigma"] = float(self.log_sigma)
        torch.save(st, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_resume(self, path: str) -> dict:
        c = self.cfg
        st = torch.load(path, map_location=self.dev, weights_only=False)
        self.net.load_state_dict(st["policy_state"])
        if "ref_state" in st:
            self.ref.net.load_state_dict(st["ref_state"])
        if self.fine is not None:
            self.fine.load_state_dict(st["fine_state"])
            self.fine_ref.load_state_dict(st["fine_ref_state"])
        if self.log_sigma is not None and "log_sigma" in st:
            with torch.no_grad():
                self.log_sigma.fill_(st["log_sigma"])
        self.critic.load_state_dict(st["critic_state"])
        self.opt.load_state_dict(st["opt"]); self.copt.load_state_dict(st["copt"])
        self.scaler.load_state_dict(st["scaler"]); self.cscaler.load_state_dict(st["cscaler"])
        self.iter = st["iter"]
        self.rng.bit_generator.state = st["rng"]
        self._prefit_done = st["prefit_done"]
        for s in st["snaps"]:
            sn = self._attach_snapshot(os.path.join(c.league_dir or ".", s["file"]), s["level"], s["added"], s["last_step"])
            sn["win"].extend(tuple(x) for x in s["win"])
        self.opp_score = np.asarray(st["opp_score"], np.float64).copy()
        self.fin = np.asarray(st["fin"], np.float64).copy()
        assert len(self.opp_score) == len(self.opps) == len(self.fin)
        self.fixed_win.extend(tuple(x) for x in st["fixed_win"])
        self.last_snap_iter, self.warm_until = st["last_snap_iter"], st["warm_until"]
        self.opp_w = self._pfsp_weights()
        self.trunk = [self._fresh_seed() for _ in range(c.games)]   # in-flight games restart
        self.trunk_opp = self.rng.choice(len(self.opps), c.games, p=self.opp_w)
        self.trunk_side = self.rng.integers(0, 2, c.games)
        self.gv[:] = 0
        return st["extra"]

    def _hl_target(self, x):
        """HL-Gauss target pmf [B, bins] of returns x [B]: N(x, sigma^2) mass per bin, renormalised over the support."""
        x = x.float().clamp(self.cfg.critic_vmin, self.cfg.critic_vmax)
        cdf = torch.special.erf((self.hl_edges[None] - x[:, None]) / (self.hl_sigma * 2 ** 0.5))
        mass = (cdf[:, 1:] - cdf[:, :-1]) / 2
        return mass / mass.sum(1, keepdim=True).clamp_min(1e-12)

    def _hl_logits(self, g, t, a):
        """fp32 bin logits [B, bins] (compiled in training for full-size minibatches, like the regression critic)."""
        c = self.cfg
        if self.critic.training and c.update_compile and self.dev.type == "cuda" and g.shape[0] in (c.minibatch, c.update_micro):
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

    def _value_actor(self, g, t, a):
        """``_value_reg`` with the actor's critic copy on ``adev`` (eager)."""
        g = unpack_graphic(g) if g.dim() == 3 else g
        with torch.autocast(self.adev.type, dtype=torch.float16, enabled=self.cfg.critic_fp16 and self.adev.type == "cuda"):
            out = self._act_critic(g, t, a)
        out = out.float()
        h = out.shape[1] - 4
        lead = (t[:, 0] - t[:, 1]).float() * 100.0
        final = lead + out[:, h] / REG_SCALE * 100.0
        p = torch.softmax(out[:, h + 1:], -1)
        return (p[:, 2] - p[:, 0]) + self.cfg.lam * torch.tanh(final / self.cfg.scale)

    def _value_reg(self, g, t, a):
        g = unpack_graphic(g) if g.dim() == 3 else g
        if self.critic.training and self.cfg.update_compile and self.dev.type == "cuda" and g.shape[0] in (self.cfg.minibatch, self.cfg.update_micro):
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
    def _values_np(self, g, t, a, bs=None, orig=False, actor=False):
        bs = bs or self.cfg.critic_read_batch
        out, ent = [], []
        dev = self.adev if actor else self.dev
        for s in range(0, len(g), bs):
            to = lambda x: torch.from_numpy(np.ascontiguousarray(x[s:s + bs])).to(dev)
            v = self._value_actor(to(g), to(t), to(a)) if actor else self._value(to(g), to(t), to(a), orig=orig)
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
        if self.dev.type == "cuda" and not self._async:
            self._between_peak = torch.cuda.max_memory_allocated() / 2**20   # since the end of the last step (checkpoint writes, evals)
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n = len(self.trunk)
        if self._async:   # actor copies (the learner's own net is being updated concurrently)
            learner = LearnerPolicy(self._act_own, self.adev, 10**9, c.temperature, seed=int(self.rng.integers(2**31)),
                                    graphs=self._act_gf)
        else:
            learner = LearnerPolicy(self.own, self.dev, 10**9, c.temperature, seed=int(self.rng.integers(2**31)),
                                    graphs=self.learner_gf, **self._learner_kw())
        res = self._rollout(self.trunk, learner, self.trunk_opp, self.trunk_side, c.segment)
        t1 = time.perf_counter()
        recs = learner.records
        game = np.concatenate([r[0] for r in recs])
        g = np.concatenate([r[1] for r in recs]); ts = np.concatenate([r[2] for r in recs])
        ag = np.concatenate([r[3] for r in recs]); act = np.concatenate([r[4] for r in recs])
        off = np.concatenate([r[5] for r in recs]) if c.offset_rl else None
        fine = np.concatenate([r[5] for r in recs]) if c.fine_bins > 0 else None
        if not self._prefit_done:
            self._prefit_head(g, ts, ag)
        v = self._values_np(g, ts, ag, actor=self._async)
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
            boot[live] = self._values_np(pk[r, s_], tm[r, s_], am[r, s_], actor=self._async)
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
            self.gv[b] += (len(rows), vb.sum(), (vb * vb).sum())
            nxt = final_r[b] if bs.over[b] else boot[b]   # value after the last step (terminal: reward, V=0)
            gae = 0.0
            for j in range(len(rows) - 1, -1, -1):
                v_next = nxt if j == len(rows) - 1 else vb[j + 1]
                delta = v_next - vb[j]                      # r = 0 except folded into the terminal v_next
                gae = delta + c.gae_lambda * gae
                adv[rows[j]] = gae
                ret[rows[j]] = gae + vb[j]
        ended = np.nonzero(bs.over)[0]
        # per-opponent advantage statistics / optional centering (``adv`` stays the raw GAE; ``adv_used`` feeds the policy)
        n_opp = len(self.opps)
        row_opp = self.trunk_opp[game]
        gpo = np.bincount(self.trunk_opp[np.unique(game)], minlength=n_opp)   # games (finished or active) per opponent in the batch
        adv_used, raw_mean = adv, np.full(n_opp, np.nan)
        if c.adv_center_opp:
            adv_used, raw_mean = lg.center_by_opponent(adv, row_opp, gpo, c.adv_center_min_games)
        else:
            for i in range(n_opp):
                m = row_opp == i
                if m.any():
                    raw_mean[i] = adv[m].mean()
        opp_adv_std, win = np.full(n_opp, np.nan), {}
        for i in range(n_opp):
            m = row_opp == i
            if m.any():
                a_i = adv_used[m]
                opp_adv_std[i] = a_i.std()
                win[i] = (float(m.sum()), float(a_i.sum()), float((a_i * a_i).sum()))
        fx = [win[i] for i in range(self.n_fixed) if i in win]
        self.fixed_win.append(tuple(np.sum(fx, 0)) if fx else (0.0, 0.0, 0.0))
        for j, sn in enumerate(self.snaps):
            sn["win"].append(win.get(self.n_fixed + j, (0.0, 0.0, 0.0)))
        hook = getattr(self, "diag_hook", None)   # offline diagnostics (examples/critic_shift_diag.py); no effect when unset
        if hook is not None:
            hook(dict(game=game, g=g, t=ts, a=ag, v=v, adv=adv, ret=ret, opp_of=self.trunk_opp.copy(), side=side.copy(),
                      over=bs.over.copy(), boot=boot, final_r=final_r, lead=lead, bsim=bs))
        # PFSP: per-opponent EMA of the learner's result on the games that ended (no branch leaves here)
        score = (np.sign(lead) + 1.0) / 2.0
        for i in range(len(self.opps)):
            m = ended[self.trunk_opp[ended] == i]
            if len(m):
                self.opp_score[i] += c.pfsp_ema * (float(score[m].mean()) - self.opp_score[i])
        self.opp_w = self._pfsp_weights()
        # finished games: MC explained-variance sums per opponent (R = the game's final reward, e = R - V over all its rows)
        for b in ended:
            n_r, s_v, ss_v = self.gv[b]
            if n_r > 0:
                R = float(final_r[b])
                self.fin[self.trunk_opp[b]] += (1.0, n_r, n_r * R, n_r * R * R, n_r * R - s_v, n_r * R * R - 2 * R * s_v + ss_v)
            self.gv[b] = 0.0
        roots = res.states()
        for b in ended:
            roots[b] = self._fresh_seed()
            self.trunk_opp[b] = self.rng.choice(len(self.opps), p=self.opp_w)
            self.trunk_side[b] = self.rng.integers(0, 2)
        for p in self.opps:
            forget(p, ended)
        self.trunk = roots
        a_std = adv_used.std() + 1e-8
        ev = 1.0 - float(np.var(ret - v) / (np.var(ret) + 1e-12))
        batch = {"g": g, "t": ts, "a": ag, "act": act, "adv": (adv_used / a_std).astype(np.float32),
                 "ret": ret.astype(np.float32)}
        if off is not None:
            batch["off"] = off
        if fine is not None:
            batch["fine"] = fine
        if self.dev.type == "cuda" and not self._async:
            self._collect_peak = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        stats = {"trunk_s": t1 - t0, "leaf_s": time.perf_counter() - t1, "views": len(game),
                 "trunk_restarts": len(ended), "decisions": int(res.steps.sum()), "v_mean": float(v.mean()), "adv_std": float(a_std),
                 "ret_mean": float(ret.mean()), "ret_std": float(ret.std()), "expl_var": ev, "abs_adv": float(np.abs(adv_used).mean()),
                 "adv_std_raw": float(adv.std()),
                 "opp_adv_mean": [round(float(x), 4) for x in raw_mean],      # per-opponent mean of the raw GAE advantage
                 "opp_adv_std": [round(float(x), 4) for x in opp_adv_std],    # per-opponent std of the advantage fed to the policy (pre global scaling)
                 "opp_share": [round(float(x), 4) for x in gpo / max(gpo.sum(), 1)],   # this batch's share of games per opponent
                 "opp_ev_mc": [round(lg.ev_mc(f), 4) for f in self.fin],      # cumulative EV_mc on finished games
                 "opp_fin": [int(f[0]) for f in self.fin],                    # cumulative finished games per opponent
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
                self.copt.zero_grad(set_to_none=True)
                m_all = len(mb)
                sub = c.update_micro if c.update_micro > 0 else m_all   # gradient accumulation over micro-chunks
                for u in range(0, m_all, sub):
                    ids = mb[u:u + sub]
                    frac = len(ids) / m_all
                    if self.hl:
                        _, lg = self._value(g[ids], t[ids], a[ids])
                        loss = -(self._hl_target(ret[ids]) * torch.log_softmax(lg, -1)).sum(-1).mean()   # fp32 log_softmax
                    else:
                        loss = F.smooth_l1_loss(self._value(g[ids], t[ids], a[ids]), ret[ids], beta=c.vf_huber)
                    sl = loss * frac if sub < m_all else loss
                    if self.cscaler.is_enabled():   # loss scaling: fp16 activation gradients of a ~1e-2 Huber loss underflow
                        self.cscaler.scale(sl).backward()
                    else:
                        sl.backward()
                    vl += loss.item() * frac
                if self.cscaler.is_enabled():
                    self.cscaler.unscale_(self.copt)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.cscaler.step(self.copt)
                    self.cscaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.copt.step()
                nv += 1
        self.critic.eval()
        st = {"vf_loss": vl / max(nv, 1)}
        if self.iter < max(c.critic_warmup, self.warm_until):
            return st
        return {**st, **super().update(batch)}

    def save(self, path: str, extra: dict | None = None) -> None:
        super().save(path, extra)
        torch.save(self.critic.state_dict(), path.replace(".pt", "_critic.pt"))
