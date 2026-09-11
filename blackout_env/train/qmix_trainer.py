"""
BBF-style DQN + QMIX self-play trainer.

Both teams are controlled by the SAME shared network (MyModel already normalizes
graphic/team_state/agent_states to "my team's own perspective" — see MyObsPreprocessor — so
one set of weights naturally plays both sides). Every env.step() yields one transition from
each team's perspective, pushed into that team's own SequentialReplayBuffer stream.

Self-play stability: each episode, one physical team (randomized) is the "online" side and
acts through the live `net`, while the other is the "opponent" side and acts through the slow
EMA copy `ema_net` instead — so the opponent doesn't chase the online net's every gradient
step (classic self-play non-stationarity/oscillation). Both sides' transitions are still
pushed to their own replay buffer and trained on; QMIX/DQN are off-policy, so training on
EMA-chosen actions is standard off-policy replay, not a correctness issue.

BBF components combined with QMIX (per project's explicit choices — see conversation):
  - IQN distributional Q-head (MyModel.q_head) instead of scalar Q, mixed team-wide via
    DistributionalQMixer (DFAC-style, Sun et al. 2021 — see modules/qmix_mixer.py), which
    mixes agents' quantile *means* through the ordinary nonlinear QMixer (preserving IGM
    exactly) and their zero-mean "shape" through a separate non-negative linear combination.
  - SPR auxiliary loss on the pooled vision latent only, via an EMA target encoder (`ema_net`)
    and SPRPredictor's open-loop K-step latent rollout (see modules/spr_predictor.py).
  - n-step returns + gamma annealing (BBF's main efficiency lever), truncated at episode
    boundaries by SequentialReplayBuffer's sequential layout (see train/returns.py).
  - Prioritized Experience Replay (train/replay_buffer.py, train/segment_tree.py).
  - Periodic shrink-and-perturb reset, scoped ONLY to graphic_encoder for now (conservative,
    per project decision — the attention trunk's FFN blocks are a separate future step).
  - AdamW (weight decay) instead of plain Adam.

Three model copies, each with a different role/update rule (this is intentional, not
duplication — see BBF/SPR literature for why the Q-learning target and the SPR target need
different staleness properties):
  - `net`        : online, trained by gradient descent every train_step().
  - `target_net` : hard-synced from `net` every `target_update_interval` train steps — the
                   Double-DQN bootstrap target for the Q-learning loss.
  - `ema_net`     : soft/EMA-updated from `net` every train_step() — doubles as the SPR target
                    encoder AND the self-play opponent's action-selection network (see above).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import N_AGENTS, N_TEAM_A, team_a_agents, team_b_agents
from blackout_env.model.modules import DistributionalQMixer, GraphicEncoder, QMixer, SPRPredictor, quantile_huber_loss
from blackout_env.model.my_model import N_DISCRETE_ACTIONS, MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS
from blackout_env.train.ema import ema_update
from blackout_env.train.replay_buffer import SequentialReplayBuffer
from blackout_env.train.reset_utils import shrink_and_perturb
from blackout_env.train.returns import compute_n_step_return, compute_spr_valid_mask
from blackout_env.train.schedules import linear_anneal, log_linear_anneal

TEAM_SIGN_COL = 2  # agent_state row: [pos_x, pos_y, team, *item_onehot, *class_onehot]
N_TEAM = N_TEAM_A  # 5 — both teams are the same size


@dataclass
class QMIXConfig:
    hidden_size: int = 256
    n_items: int = 5
    n_classes: int = 3
    team_state_size: int = 4

    mixer_embed_dim: int = 64
    mixer_hyper_hidden: int = 128

    buffer_capacity: int = 10_000  # per stream (team A / team B each get their own buffer)
    batch_size: int = 64           # split evenly across the two streams
    min_buffer_size: int = 1_000   # per stream, before training starts

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 10.0

    n_quantiles: int = 8  # IQN quantile samples per forward pass during training

    # n-step / gamma annealing (BBF): n_step counts DOWN, gamma counts UP, over the same
    # fraction of total training.
    n_step_start: int = 10
    n_step_end: int = 3
    gamma_start: float = 0.97
    gamma_end: float = 0.997
    anneal_frac: float = 0.1  # fraction of total_env_steps over which annealing completes

    # SPR (vision-only self-predictive auxiliary loss)
    spr_k: int = 5
    spr_loss_weight: float = 1.0
    ema_tau: float = 0.99

    # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_eps: float = 1e-3
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    train_every: int = 4          # env steps between train_step() calls
    grad_steps_per_call: int = 1  # gradient updates per train_step() call (replay-ratio knob)
    target_update_interval: int = 500  # train (gradient) steps between target hard-syncs

    # Periodic CNN-only reset (shrink-and-perturb). reset_interval=0 disables it.
    reset_interval: int = 0
    reset_alpha: float = 0.5

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5_000  # env steps
    device: str = "cpu"


def _own_team_rows(tensor: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
    """
    Gathers each sample's own-team 5 unit rows out of `tensor`'s dim=1 (the unit axis), in
    ascending unit-index order — works for any trailing shape (q_values [B,10,A], or
    quantile_values [B,10,Q,A]).

    "Own team" is always a contiguous physical block (unit 0-4, or unit 5-9 — see
    MyObsPreprocessor.preprocess_agent_states, team is never interleaved), and the sign is
    right there in agent_states' team column, so a single scalar per sample (row 0's sign)
    is enough to tell which block: sign > 0 means row 0 (physical unit 0) is on "my" team,
    i.e. own block is [0:5]; sign < 0 means own block is [5:10].
    """
    B = tensor.shape[0]
    trailing_shape = tensor.shape[2:]
    is_team_a = agent_states[:, 0, TEAM_SIGN_COL] > 0
    own_start = torch.where(is_team_a, 0, N_TEAM_A)
    own_idx = own_start.unsqueeze(1) + torch.arange(N_TEAM, device=tensor.device).unsqueeze(0)  # [B, N_TEAM]
    idx = own_idx.view(B, N_TEAM, *([1] * len(trailing_shape))).expand(B, N_TEAM, *trailing_shape)
    return torch.gather(tensor, 1, idx)


class QMIXTrainer:
    def __init__(self, env: BlackOutEnv, config: QMIXConfig) -> None:
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)

        model_kwargs = dict(
            hidden_size=config.hidden_size,
            n_items=config.n_items,
            n_classes=config.n_classes,
            team_state_size=config.team_state_size,
        )
        self.net = MyModel(**model_kwargs).to(self.device)
        self.target_net = copy.deepcopy(self.net).to(self.device)
        self.target_net.eval()
        self.ema_net = copy.deepcopy(self.net).to(self.device)
        self.ema_net.eval()

        self.mixer = QMixer(
            n_agents=N_TEAM,
            state_dim=config.hidden_size,
            embed_dim=config.mixer_embed_dim,
            hyper_hidden=config.mixer_hyper_hidden,
        ).to(self.device)
        self.dist_mixer = DistributionalQMixer(
            self.mixer, state_dim=config.hidden_size, n_agents=N_TEAM, hyper_hidden=config.mixer_hyper_hidden
        ).to(self.device)
        self.target_dist_mixer = copy.deepcopy(self.dist_mixer).to(self.device)
        self.target_dist_mixer.eval()

        self.spr_predictor = SPRPredictor(
            config.hidden_size, N_DISCRETE_ACTIONS, n_units=N_AGENTS
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            itertools.chain(self.net.parameters(), self.dist_mixer.parameters(), self.spr_predictor.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        graphic_shape = env.observation_space(team_a_agents()[0])["graphic"].shape
        agent_state_size = 2 + 1 + (config.n_items + 1) + config.n_classes
        buffer_kwargs = dict(
            capacity=config.buffer_capacity,
            graphic_shape=graphic_shape,
            team_state_size=config.team_state_size,
            agent_state_size=agent_state_size,
            n_units=N_AGENTS,
            per_alpha=config.per_alpha,
            per_eps=config.per_eps,
        )
        self.buffer_a = SequentialReplayBuffer(**buffer_kwargs)
        self.buffer_b = SequentialReplayBuffer(**buffer_kwargs)

        self.team_a_agents = team_a_agents()
        self.team_b_agents = team_b_agents()

        self.env_step_count = 0
        self.train_step_count = 0
        self._total_env_steps_hint = 1  # set properly in run(); avoids div-by-zero if train_step() is called standalone
        self._online_is_team_a = True  # re-randomized every episode in _reset_env()

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    def epsilon(self) -> float:
        frac = min(1.0, self.env_step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _anneal_frac(self) -> float:
        anneal_steps = max(1, self.cfg.anneal_frac * self._total_env_steps_hint)
        return self.env_step_count / anneal_steps

    def current_n_step(self) -> int:
        return max(1, round(linear_anneal(self.cfg.n_step_start, self.cfg.n_step_end, self._anneal_frac())))

    def current_gamma(self) -> float:
        return log_linear_anneal(self.cfg.gamma_start, self.cfg.gamma_end, self._anneal_frac())

    def current_per_beta(self) -> float:
        frac = min(1.0, self.env_step_count / max(1, self._total_env_steps_hint))
        return linear_anneal(self.cfg.per_beta_start, self.cfg.per_beta_end, frac)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _to_batch(self, obs_a: dict, obs_b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stacks both teams' perspectives into one B=2 batch: index 0 = team A, index 1 = team B."""
        graphic = np.stack([obs_a["graphic"], obs_b["graphic"]])
        team_state = np.stack([obs_a["team_state"], obs_b["team_state"]])
        agent_states = np.stack([obs_a["agent_states"], obs_b["agent_states"]])
        return (
            torch.tensor(graphic, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2),
            torch.tensor(team_state, dtype=torch.float32, device=self.device),
            torch.tensor(agent_states, dtype=torch.float32, device=self.device),
        )

    @torch.no_grad()
    def select_actions(self, obs: dict[str, dict[str, np.ndarray]], epsilon: float):
        """
        Returns (env_actions, full_direction_idx) where env_actions is the dict[agent,(dx,dy)]
        BlackOutEnv.step() expects, and full_direction_idx is [10] (physical unit order, both
        teams) for the replay buffer.

        This episode's "opponent" side (see `self._online_is_team_a`, module docstring) acts
        through `ema_net` instead of `net` for self-play stability.
        """
        obs_a, obs_b = obs[self.team_a_agents[0]], obs[self.team_b_agents[0]]
        graphic, team_state, agent_states = self._to_batch(obs_a, obs_b)

        q_online, *_ = self.net(graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles)  # [2,10,8]
        q_ema, *_ = self.ema_net(graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles)

        opponent_idx = 1 if self._online_is_team_a else 0
        q_values = q_online.clone()
        q_values[opponent_idx] = q_ema[opponent_idx]

        own_q = _own_team_rows(q_values, agent_states)  # [2, N_TEAM, 8]
        greedy = own_q.argmax(dim=-1).cpu().numpy()  # [2, N_TEAM]

        direction_idx = np.where(
            np.random.rand(2, N_TEAM) < epsilon,
            np.random.randint(0, N_DISCRETE_ACTIONS, size=(2, N_TEAM)),
            greedy,
        )

        env_actions = {}
        full_direction_idx = np.zeros(N_AGENTS, dtype=np.int64)
        for team_idx, agents in enumerate((self.team_a_agents, self.team_b_agents)):
            for slot, agent in enumerate(agents):
                d = direction_idx[team_idx, slot]
                env_actions[agent] = DIRECTION_VECTORS[d]
                full_direction_idx[team_idx * N_TEAM + slot] = d

        return env_actions, full_direction_idx

    def _reset_env(self):
        """
        Resets the env and re-randomizes which physical team is this episode's "online" (live
        `net`) side vs the EMA-controlled "opponent" side (see module docstring).
        """
        obs, info = self.env.reset()
        self._online_is_team_a = bool(np.random.rand() < 0.5)
        return obs, info

    def collect_step(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        epsilon = self.epsilon()
        env_actions, full_direction_idx = self.select_actions(obs, epsilon)

        obs_a, obs_b = obs[self.team_a_agents[0]], obs[self.team_b_agents[0]]
        next_obs, rewards, terminations, _, _ = self.env.step(env_actions)
        done = any(terminations.values())

        reward_a = sum(rewards[a] for a in self.team_a_agents)
        reward_b = sum(rewards[a] for a in self.team_b_agents)

        self.buffer_a.push(obs_a["graphic"], obs_a["team_state"], obs_a["agent_states"], full_direction_idx, reward_a, done)
        self.buffer_b.push(obs_b["graphic"], obs_b["team_state"], obs_b["agent_states"], full_direction_idx, reward_b, done)

        self.env_step_count += 1

        if done:
            next_obs, _ = self._reset_env()
        return next_obs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _to_graphic_tensor(self, graphic_np: np.ndarray) -> torch.Tensor:
        """
        [B,H,W,C] numpy -> [B,C,H,W] tensor. No augmentation: unlike Atari pixels, each cell
        of `graphic` is a 1:1 semantic reading of one map tile, so a spatial shift wouldn't be
        "the same scene from a slightly different view" (DrQ's actual premise) — it would
        just replace real boundary tiles with padding and lose information, with nothing
        gained in return.
        """
        return torch.tensor(graphic_np, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

    def _compute_loss(
        self, buffer: SequentialReplayBuffer, batch_size: int, n_step: int, gamma: float, beta: float
    ):
        """Returns (total_loss_per_sample [B], td_error [B], indices [B]) for one stream's batch."""
        window = max(n_step, self.cfg.spr_k)
        batch = buffer.sample(batch_size, window=window, beta=beta)
        indices = batch["indices"]

        graphic = self._to_graphic_tensor(batch["graphic"])
        team_state = torch.tensor(batch["team_state"], dtype=torch.float32, device=self.device)
        agent_states = torch.tensor(batch["agent_states"], dtype=torch.float32, device=self.device)
        actions_full = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)  # [B, 10]
        own_actions = _own_team_rows(actions_full.unsqueeze(-1), agent_states).squeeze(-1)  # [B, N_TEAM]
        is_weights = torch.tensor(batch["is_weights"], dtype=torch.float32, device=self.device)

        # ---- n-step return + bootstrap state (Double DQN target, distributional) ----
        n_step_return, bootstrap_idx, not_done, gamma_eff = compute_n_step_return(
            buffer.reward, buffer.done, indices, buffer.capacity, n_step, gamma
        )
        n_step_return_t = torch.tensor(n_step_return, dtype=torch.float32, device=self.device)
        not_done_t = torch.tensor(not_done, dtype=torch.float32, device=self.device)
        gamma_eff_t = torch.tensor(gamma_eff, dtype=torch.float32, device=self.device)

        boot_graphic = self._to_graphic_tensor(buffer.graphic[bootstrap_idx])
        boot_team_state = torch.tensor(buffer.team_state[bootstrap_idx], dtype=torch.float32, device=self.device)
        boot_agent_states = torch.tensor(buffer.agent_states[bootstrap_idx], dtype=torch.float32, device=self.device)

        # ---- online forward (current state): Q-learning prediction + SPR rollout start ----
        q_values, quantile_values, tau, vision_latent, global_latent = self.net(
            graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles
        )
        own_quantiles = _own_team_rows(quantile_values, agent_states)  # [B, N_TEAM, Q, n_actions]
        chosen_quantiles = torch.gather(
            own_quantiles, 3, own_actions.view(*own_actions.shape, 1, 1).expand(-1, -1, self.cfg.n_quantiles, 1)
        ).squeeze(-1)  # [B, N_TEAM, Q]
        q_tot_online = self.dist_mixer(chosen_quantiles, global_latent)  # [B, Q]

        # ---- Double DQN target: online net picks the bootstrap action, target net evaluates it ----
        with torch.no_grad():
            boot_q_online, *_ = self.net(boot_graphic, boot_team_state, boot_agent_states, n_quantiles=self.cfg.n_quantiles)
            boot_greedy = _own_team_rows(boot_q_online, boot_agent_states).argmax(dim=-1)  # [B, N_TEAM]

            _, boot_quantiles_target, _, _, boot_global_target = self.target_net(
                boot_graphic, boot_team_state, boot_agent_states, n_quantiles=self.cfg.n_quantiles
            )
            boot_own_quantiles_target = _own_team_rows(boot_quantiles_target, boot_agent_states)  # [B, N_TEAM, Q', A]
            boot_chosen_target = torch.gather(
                boot_own_quantiles_target, 3, boot_greedy.view(*boot_greedy.shape, 1, 1).expand(-1, -1, self.cfg.n_quantiles, 1)
            ).squeeze(-1)  # [B, N_TEAM, Q']
            q_tot_target = self.target_dist_mixer(boot_chosen_target, boot_global_target)  # [B, Q']

            target = n_step_return_t.unsqueeze(1) + gamma_eff_t.unsqueeze(1) * not_done_t.unsqueeze(1) * q_tot_target

        iqn_loss = quantile_huber_loss(q_tot_online, tau, target)  # [B]
        with torch.no_grad():
            td_error = (q_tot_online.mean(dim=1) - target.mean(dim=1)).abs()

        # ---- SPR: open-loop K-step latent rollout vs EMA target encoder ----
        pooled_vision = vision_latent.mean(dim=1)  # [B, hidden]
        offsets_future = np.arange(1, self.cfg.spr_k + 1)
        offsets_action = np.arange(0, self.cfg.spr_k)
        future_graphic = buffer.read_window(indices, offsets_future, "graphic")  # [B, K, H, W, C]
        future_team_state = buffer.read_window(indices, offsets_future, "team_state")
        future_agent_states = buffer.read_window(indices, offsets_future, "agent_states")
        action_window = buffer.read_window(indices, offsets_action, "actions")  # [B, K, 10]
        valid_mask = compute_spr_valid_mask(buffer.done, indices, buffer.capacity, self.cfg.spr_k)  # [B, K] bool

        Bsz = graphic.shape[0]
        K = self.cfg.spr_k
        with torch.no_grad():
            flat_graphic = self._to_graphic_tensor(future_graphic.reshape(Bsz * K, *future_graphic.shape[2:]))
            flat_team_state = torch.tensor(
                future_team_state.reshape(Bsz * K, -1), dtype=torch.float32, device=self.device
            )
            flat_agent_states = torch.tensor(
                future_agent_states.reshape(Bsz * K, *future_agent_states.shape[2:]), dtype=torch.float32, device=self.device
            )
            _, _, _, ema_vision_latent, _ = self.ema_net(
                flat_graphic, flat_team_state, flat_agent_states, n_quantiles=1
            )
            target_pooled = ema_vision_latent.mean(dim=1).view(Bsz, K, -1)  # [B, K, hidden]

        action_window_t = torch.tensor(action_window, dtype=torch.long, device=self.device)  # [B, K, 10]
        valid_mask_t = torch.tensor(valid_mask, dtype=torch.float32, device=self.device)  # [B, K]

        z = pooled_vision
        spr_losses = []
        for k in range(K):
            z = self.spr_predictor.step(z, action_window_t[:, k, :])
            spr_losses.append(self.spr_predictor.loss(z, target_pooled[:, k, :]))
        spr_losses = torch.stack(spr_losses, dim=1)  # [B, K]
        spr_loss = (spr_losses * valid_mask_t).sum(dim=1) / (valid_mask_t.sum(dim=1) + 1e-6)  # [B]

        total_loss = is_weights * iqn_loss + self.cfg.spr_loss_weight * spr_loss  # [B]
        return total_loss, td_error.cpu().numpy(), indices

    def train_step(self) -> float | None:
        if len(self.buffer_a) < self.cfg.min_buffer_size or len(self.buffer_b) < self.cfg.min_buffer_size:
            return None

        n_step = self.current_n_step()
        gamma = self.current_gamma()
        beta = self.current_per_beta()
        half = self.cfg.batch_size // 2

        losses = []
        for _ in range(self.cfg.grad_steps_per_call):
            loss_a, td_a, idx_a = self._compute_loss(self.buffer_a, half, n_step, gamma, beta)
            loss_b, td_b, idx_b = self._compute_loss(self.buffer_b, half, n_step, gamma, beta)
            loss = torch.cat([loss_a, loss_b]).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(self.net.parameters(), self.dist_mixer.parameters(), self.spr_predictor.parameters()),
                self.cfg.grad_clip,
            )
            self.optimizer.step()

            self.buffer_a.update_priorities(idx_a, td_a)
            self.buffer_b.update_priorities(idx_b, td_b)

            ema_update(self.ema_net, self.net, self.cfg.ema_tau)
            ema_update(self.spr_predictor.target_projector, self.spr_predictor.projector, self.cfg.ema_tau)

            self.train_step_count += 1
            if self.train_step_count % self.cfg.target_update_interval == 0:
                self.target_net.load_state_dict(self.net.state_dict())
                self.target_dist_mixer.load_state_dict(self.dist_mixer.state_dict())

            losses.append(loss.item())

        return sum(losses) / len(losses)

    def maybe_reset(self) -> None:
        """
        Checked once per env step (see run()) rather than inside train_step() — train_step()
        only actually runs on env steps that are multiples of `train_every`, so gating the
        reset on env_step_count *inside* it would silently drift the effective period to
        lcm(train_every, reset_interval) instead of the configured reset_interval.
        """
        if not (self.cfg.reset_interval > 0 and self.env_step_count % self.cfg.reset_interval == 0):
            return

        shrink_and_perturb(
            self.net.graphic_encoder,
            # .to(self.device): reinit_fn builds on CPU by default: mixing its plain params
            # into a CUDA-resident net's parameters in-place would otherwise fail with a
            # device-mismatch error the first time reset actually fires under CUDA training.
            reinit_fn=lambda: GraphicEncoder(hidden_size=self.cfg.hidden_size).to(self.device),
            alpha=self.cfg.reset_alpha,
        )

        # A partial reinit changes what each CNN parameter's Adam momentum was computed
        # against -- keeping stale momentum would actively fight the newly-perturbed weights,
        # defeating half the point of resetting (Ash & Adams 2020 / BBF both reset optimizer
        # state alongside parameters for exactly this reason).
        for p in self.net.graphic_encoder.parameters():
            self.optimizer.state.pop(p, None)

        # target_net / ema_net would otherwise keep evaluating the OLD (pre-reset) CNN
        # representation for up to target_update_interval steps / at EMA's slow pace, while
        # the online net has already jumped to a partly-random one -- bootstrapping a
        # Q-learning target against such a mismatched representation right after a reset is
        # exactly the kind of thing that destabilizes training, so sync both immediately
        # instead of letting them lag.
        self.target_net.graphic_encoder.load_state_dict(self.net.graphic_encoder.state_dict())
        self.ema_net.graphic_encoder.load_state_dict(self.net.graphic_encoder.state_dict())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, total_env_steps: int) -> None:
        self._total_env_steps_hint = total_env_steps
        obs, _ = self._reset_env()
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        recent_losses: list[float] = []
        while self.env_step_count < total_env_steps:
            obs = self.collect_step(obs)
            self.maybe_reset()

            if self.env_step_count % self.cfg.train_every == 0:
                loss = self.train_step()
                if loss is not None:
                    recent_losses.append(loss)

            if self.env_step_count % 1000 == 0:
                elapsed = time.time() - t0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
                print(
                    f"[step {self.env_step_count}] eps={self.epsilon():.3f} n_step={self.current_n_step()} "
                    f"gamma={self.current_gamma():.4f} avg_loss={avg_loss:.4f} "
                    f"({self.env_step_count / max(elapsed, 1e-9):.1f} steps/s)"
                )
                recent_losses.clear()

            if self.env_step_count % self.cfg.checkpoint_interval == 0:
                self.save(ckpt_dir / f"step_{self.env_step_count}.pt")

        self.save(ckpt_dir / "final.pt")

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_state": self.net.state_dict(),
                # dist_mixer's state already includes the inner QMixer's params (registered
                # as its `mixer` submodule) plus DistributionalQMixer's own shape_weight
                # params -- one key covers both, no need to also save self.mixer separately.
                "dist_mixer_state": self.dist_mixer.state_dict(),
                "spr_state": self.spr_predictor.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "env_step_count": self.env_step_count,
                "train_step_count": self.train_step_count,
            },
            path,
        )
        print(f"[checkpoint] saved {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["policy_state"])
        self.ema_net.load_state_dict(ckpt["policy_state"])
        self.dist_mixer.load_state_dict(ckpt["dist_mixer_state"])
        self.target_dist_mixer.load_state_dict(ckpt["dist_mixer_state"])
        self.spr_predictor.load_state_dict(ckpt["spr_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.env_step_count = ckpt["env_step_count"]
        self.train_step_count = ckpt["train_step_count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, help="Path to the Unity build executable")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total env steps to train for")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale")
    parser.add_argument("--graphics", action="store_true", help="Show the Unity window instead of headless")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = BlackOutEnv(
        env_path=args.build,
        # GraphicEncoder assumes a 24x24 input (GRID_H=GRID_W=6 after two stride-2 convs,
        # see my_model.py) — must match the Unity build's RenderTextureSensor resolution.
        map_w=24,
        map_h=24,
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
    )
    config = QMIXConfig(checkpoint_dir=args.checkpoint_dir, device=args.device)
    trainer = QMIXTrainer(env, config)

    if args.resume:
        trainer.load(Path(args.resume))

    try:
        trainer.run(args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
