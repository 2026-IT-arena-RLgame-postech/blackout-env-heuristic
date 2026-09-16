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

Heuristic bootstrap (offline -> online handoff, see QMIXConfig.heuristic_fill_frac):
phase 1 fills both replay streams purely from HeuristicPolicyMixture rollouts (no net/ema_net
forward pass at all), with train_step() already running once each stream passes
bootstrap_train_start_frac. Phase 2 (permanent, one-way switch — see `_bootstrapping`) then
epsilon-mixes each unit's action between this episode's heuristic policy and the model's own
greedy Q, annealing epsilon from scratch starting at the transition. Because
SequentialReplayBuffer is a plain FIFO ring, the buffer's composition drifts from
all-heuristic toward increasingly model-influenced transitions for free as phase 2 collection
overwrites the oldest entries — no explicit offline/online reweighting needed.

BBF components combined with QMIX (per project's explicit choices — see conversation):
  - IQN distributional Q-head (MyModel.q_head) instead of scalar Q, mixed team-wide via
    DistributionalQMixer (DFAC-style, Sun et al. 2021 — see modules/qmix_mixer.py), which
    mixes agents' quantile *means* through the ordinary nonlinear QMixer (preserving IGM
    exactly) and their zero-mean "shape" through a separate non-negative linear combination.
  - SPR auxiliary loss on the pooled vision latent only, via an EMA target encoder (`ema_net`)
    and SPRPredictor's open-loop K-step latent rollout (see modules/spr_predictor.py).
  - n-step returns + gamma annealing (BBF's main efficiency lever), truncated at "episode"
    boundaries by SequentialReplayBuffer's sequential layout (see train/returns.py) -- "episode"
    here means each absorption interval (~120s), not the full ~600s match: see collect_step's
    absorption_fired handling for why a full match is too long a horizon to bootstrap over.
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import N_AGENTS, N_TEAM_A, team_a_agents, team_b_agents
from blackout_env.env.my_obs_preprocessor import MyObsPreprocessor
from blackout_env.env.obs_preprocessor import load_semantic_config
from blackout_env.env.team_frame import (
    canonical_obs,
    mirror_action_idx,
    mirror_agent_states,
    mirror_direction_idx,
    mirror_graphic,
)
from blackout_env.heuristics import HeuristicPolicyMixture
from blackout_env.train.offline_dataset import load_dataset_into
from blackout_env.model.modules import (
    AttentionLayers,
    DistributionalQMixer,
    GraphicEncoder,
    QMixer,
    SPRPredictor,
    clamp_pressure_stats,
    quantile_huber_loss,
)
from blackout_env.model.my_model import ATTENTION_DEPTH, N_ATTENTION_HEADS, N_DISCRETE_ACTIONS, MyModel
from blackout_env.model.action_mask import masked_greedy
from blackout_env.model.my_policy import DIRECTION_VECTORS, direction_vector_to_idx
from blackout_env.train.ema import ema_update
from blackout_env.train.replay_buffer import (
    SOURCE_DATASET,
    SOURCE_NAMES,
    SOURCE_SELF_PLAY,
    SOURCE_SELF_VS_HEURISTIC,
    SequentialReplayBuffer,
)
from blackout_env.train.reset_utils import shrink_and_perturb
from blackout_env.train.returns import compute_n_step_return, compute_spr_valid_mask
from blackout_env.train.schedules import linear_anneal, log_linear_anneal
from blackout_env.train.tb_logger import TBLogger, default_run_dir

TEAM_SIGN_COL = 2  # agent_state row: [pos_x, pos_y, team, *item_onehot, *class_onehot]
N_TEAM = N_TEAM_A  # 5 — both teams are the same size
ABSORPTION_IDX = 3  # team_state row: [own_score, opp_score, episode_time_left, absorption_time_left]


@dataclass
class QMIXConfig:
    hidden_size: int = 128
    n_items: int = 5
    n_classes: int = 3
    team_state_size: int = 4

    mixer_embed_dim: int = 64
    mixer_hyper_hidden: int = 128

    buffer_capacity: int = 10_000  # per stream (team A / team B each get their own buffer)
    batch_size: int = 64           # split evenly across the two streams

    # ---- Heuristic bootstrap (offline -> online handoff) ----
    # Phase 1 ("heuristic fill"): both teams act purely via HeuristicPolicyMixture -- no
    # net/ema_net forward pass for action selection at all. Once EACH stream (buffer_a AND
    # buffer_b) reaches heuristic_fill_frac * capacity, the trainer permanently switches to
    # phase 2: per-unit epsilon-greedy between this episode's heuristic action (exploration)
    # and the model's own greedy Q argmax (exploitation) -- see select_actions(). Because the
    # replay buffer is a plain FIFO ring (SequentialReplayBuffer), once phase 2 starts pushing
    # increasingly model-influenced transitions, the oldest pure-heuristic ones are naturally
    # evicted over the next `capacity` steps with no extra bookkeeping -- the offline (heuristic)
    # -> online (model) handoff falls out of the ring buffer's own eviction order for free.
    heuristic_fill_frac: float = 1.0
    # Training starts once EACH stream has this fraction of capacity, even while still in
    # phase 1 -- replaces the old fixed min_buffer_size (which was really this same fraction,
    # 1_000/10_000 = 10%; expressing it as a fraction survives a buffer_capacity change without
    # silently changing what fraction of the buffer training waits for).
    bootstrap_train_start_frac: float = 0.2
    # Separate seeds so team A's and team B's HeuristicPolicyMixture don't always sample the
    # identical policy_id/parameters each episode -- more matchup diversity during bootstrap.
    heuristic_seed_a: int = 0
    heuristic_seed_b: int = 1
    # Per-unit probability of a uniformly random compass direction instead of the heuristic's
    # own choice during phase 1 (see select_actions_heuristic). 0 reproduces the old pure
    # heuristic-vs-heuristic bootstrap exactly. Also used verbatim by the standalone
    # collect_heuristic_dataset.py script, which drives this same method to build an offline
    # dataset outside of any online run.
    heuristic_bootstrap_noise_frac: float = 0.0

    # ---- Heuristic-opponent mixing (phase 2 only, on top of the bootstrap above) ----
    # Phase 2's self-play opponent is otherwise always ema_net (see module docstring) -- two
    # nets that only ever adapt to each other can converge to a mutually low-risk equilibrium
    # (e.g. neither side ever completes a risky deposit) that looks converged in every training
    # metric (return, win-rate, loss, weight norms) yet loses to a real heuristic that actually
    # executes the scoring loop. With probability heuristic_opponent_frac, each episode's
    # opponent side plays through its full HeuristicPolicyMixture (heuristic_a/heuristic_b, same
    # instances used for epsilon-exploration) for the WHOLE episode instead of ema_net -- see
    # select_actions(). The online side is unaffected (still net greedy + epsilon-heuristic
    # exploration); only which policy controls the opponent's row changes. 0 reproduces the old
    # pure-self-play behavior exactly.
    heuristic_opponent_frac: float = 0.3

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 10.0

    # Separate AdamW param group for MyModel.graphic_encoder only -- None means "inherit lr/
    # weight_decay above", so this is a no-op unless explicitly set. Added after observing
    # grad_norm/graphic_encoder decay ~7 orders of magnitude over a 200k-step offline pretrain
    # run (0.82 -> ~1e-7) while every other component's grad_norm stayed flat at 1e-2..1e-1 --
    # graphic_encoder started with the weakest grad_norm among all components even at step 0,
    # and a single global weight_decay applied via one AdamW over all params erodes a
    # persistently-weaker branch faster than its own (also weak) gradient can rebuild it, a
    # self-reinforcing collapse BBF-style periodic resets only interrupt for ~1000 steps at a
    # time (see docs/offline_pretrain_runs.md). Kept as a targeted override, not a general
    # per-module scheme, since graphic_encoder is the one branch actually observed to collapse.
    encoder_lr: float | None = None
    encoder_weight_decay: float | None = None

    n_quantiles: int = 8  # IQN quantile samples per forward pass during training

    # n-step / gamma annealing (BBF): n_step counts DOWN, gamma counts UP, over the same
    # fraction of total training. n_step_start=10 (back to BBF's original value): Unit.prefab's
    # DecisionRequester now runs at DecisionPeriod=2 (25Hz decisions over a 50Hz/0.02s physics
    # tick, TakeActionsBetweenDecisions=1 repeats the last action on the skipped tick -- see
    # DecisionRequester.cs/RpcCommunicator.DecideBatch, which never even calls back to Python on
    # that skipped tick, so every env_step_count here already IS one real forward-pass decision
    # point spanning 0.04s of game time, not 0.02s). That doubles each step's temporal reach, so
    # n_step=10 here now covers the same ~0.4s local "spot an item 2-3 tiles away, approach, pick
    # it up" sequence (unit speeds 4-6 units/s on a 24x24 grid) that n_step=20 covered at the old
    # 50Hz rate -- same reach, back to BBF's lower-variance/less-stale value instead of paying
    # 2x the n-step-return variance and off-policy staleness for no extra reach.
    n_step_start: int = 10
    n_step_end: int = 3
    gamma_start: float = 0.97
    gamma_end: float = 0.997
    anneal_frac: float = 0.1  # fraction of total_env_steps over which annealing completes

    # SPR (vision-only self-predictive auxiliary loss)
    spr_k: int = 5
    spr_loss_weight: float = 1.0

    # Behavior-cloning auxiliary loss (discrete-action cross-entropy variant, structurally like
    # Discrete BCQ's imitation head -- Fujimoto et al. 2019, "Benchmarking Batch Deep RL",
    # sfujim/BCQ's discrete_BCQ.py): cross_entropy(q_values, dataset_action) per own-team unit.
    # Added for offline_pretrain.py specifically -- a purely offline TD/SPR objective can hit
    # low loss without the net ever needing to parse graphic_encoder's output accurately, since
    # a heuristic-only dataset's outcomes are already consistent/predictable from agent_states
    # alone (no online self-play feedback loop to punish that shortcut, see
    # docs/offline_pretrain_runs.md). Directly supervising "predict the exact direction the
    # heuristic took" forces the net to actually explain wall-avoidance/item-seeking decisions
    # that visibly depend on the vision channel.
    #
    # bc_loss_alpha (0.0 = off, matches all prior behavior exactly) is a TD3+BC-style (Fujimoto
    # & Gu 2021, "A Minimalist Approach to Offline RL") *adaptive* multiplier, not a raw loss
    # weight: the BC term is rescaled every step to iqn_loss's own current magnitude before
    # bc_loss_alpha is applied (see _forward_and_loss), so alpha=1.0 means "BC and TD loss
    # contribute equally regardless of their raw scales" -- deliberately not a fixed weight like
    # discrete BCQ's literal 1.0, because that paper's TD loss and ours differ by ~2 orders of
    # magnitude in this environment (their Atari Huber TD loss vs. our IQN quantile loss, which
    # stays ~0.01-0.05 across an entire run -- a literal weight=1.0 here would let a fresh
    # cross-entropy term starting at ln(8)=2.08 dominate the total loss throughout training).
    # Not intended for online training (self-play data isn't a reference policy worth imitating
    # once the net starts outperforming it), so this stays 0.0 unless offline_pretrain.py's
    # --bc-loss-alpha sets it.
    #
    # Only demonstration transitions (SequentialReplayBuffer.demo: the stream's own team was played
    # by a heuristic) are cloned -- the static dataset, the heuristic side of self-vs-heuristic
    # matches, online phase-1 bootstrap and heuristic-opponent streams. Rows the net itself played
    # carry its own greedy actions, and since BC uses the Q-values as logits it would raise the Q
    # of exactly those actions -- in Run 5 that meant reinforcing the wall-walking the blocked
    # penalty was trying to remove.
    bc_loss_alpha: float = 0.0
    # Drop directions that walk into a wall from every greedy choice -- acting, collecting, and the
    # Double-DQN bootstrap action (never from the Q of a stored action; see model/action_mask.py).
    # A blocked unit re-picks the same wall-ward direction forever because the state it observes
    # barely changes, which is why Run 6 spent ~25% of unit-ticks blocked against the heuristics'
    # ~1.5%; masking measured 24.7% -> 0.6% on the same checkpoint.
    action_masking: bool = True
    # Fixed share of every batch drawn from each replay source (indexed like
    # replay_buffer.SOURCE_NAMES: dataset, self_vs_heuristic, self_play). When set, buffer_a/b hold
    # only the static dataset and are never written after loading, and each on-policy source with a
    # nonzero share gets its own small FIFO buffer pair (onpolicy_buffers, onpolicy_buffer_capacity
    # rows per stream), each buffer with its own PER priorities. Previously on-policy data shared
    # buffer_a/b's ring and overwrote the dataset's earliest episodes (Run 5 lost ~41% of it).
    # A source whose buffer can't be sampled yet has its share spread over the others.
    # offline_pretrain.py sets this from its --onpolicy-*-frac collection ratios. None = sample
    # buffer_a/b alone (the online trainer, where one mixed buffer is the design).
    batch_source_fracs: tuple[float, float, float] | None = None
    onpolicy_buffer_capacity: int = 262_144
    ema_tau: float = 0.99

    # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_eps: float = 1e-3
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    train_every: int = 4          # env steps between train_step() calls
    grad_steps_per_call: int = 1  # gradient updates per train_step() call (replay-ratio knob)
    target_update_interval: int = 500  # train (gradient) steps between target hard-syncs

    # Periodic shrink-and-perturb reset (Ash & Adams 2020 / BBF), applied to both
    # graphic_encoder (CNN) and the attention trunk on the same schedule. reset_interval=0
    # disables it entirely. alpha = how much of the OLD weights to KEEP (see
    # reset_utils.shrink_and_perturb) -- kept deliberately gentle: the attention trunk carries
    # more of the network's already-learned behavior than the CNN front-end, so it gets perturbed
    # less (5-10% -> keep ~92.5%) than the CNN (20% -> keep 80%).
    reset_interval: int = 0
    reset_alpha_cnn: float = 0.8
    reset_alpha_attention: float = 0.925

    # Linear warmup (env steps, counted from the current reset cycle's start -- see
    # _anneal_cycle_start_step) for graphic_encoder's own AdamW param group LR only, ramping
    # 0 -> encoder_lr. _reset_submodule() wipes this submodule's Adam momentum/variance state on
    # every reset, so the first post-reset steps run with no second-moment history -- exactly the
    # high-variance regime Adam warmup schedules exist to smooth over, and graphic_encoder is the
    # one component repeatedly found fragile right after reset/init (docs/offline_pretrain_runs.md
    # -- unlike attention, whose grad_norm has stayed healthy throughout, so this is scoped to the
    # encoder param group rather than the whole optimizer). 0 (default) disables it (no-op, LR
    # stays at encoder_lr always). Also applies once at true training start (env step 0), since
    # the first reset fires there too (env_step_count % reset_interval == 0).
    reset_warmup_steps: int = 0

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    # TensorBoard logging (see train/tb_logger.py). None disables it entirely. Defaults to a
    # fresh timestamped subdirectory per process (default_run_dir) rather than a fixed "runs" --
    # two concurrent runs writing into the exact same log_dir let one run's startup/cleanup
    # delete the directory out from under the other's already-open SummaryWriter, which
    # previously crashed training outright (TBLogger now also survives that if it still happens).
    tb_log_dir: str | None = field(default_factory=default_run_dir)
    tb_log_interval: int = 1000  # train steps between loss/grad/weight-norm scalars (matches the env-step print cadence in run())

    # Defaults to a fresh timestamped subdirectory per process (default_run_dir, same scheme as
    # tb_log_dir above) rather than a fixed "checkpoints" -- otherwise two runs launched around
    # the same time (or a resumed run) would silently overwrite each other's step_*.pt files.
    checkpoint_dir: str = field(default_factory=lambda: default_run_dir(base="checkpoints"))
    checkpoint_interval: int = 5_000  # env steps
    device: str = "cpu"
    compile: bool = False  # torch.compile net/target_net/ema_net to cut kernel-launch overhead


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
    def __init__(self, env: BlackOutEnv | None, config: QMIXConfig) -> None:
        """
        env=None builds the network/buffers/optimizer with no live Unity process at all --
        everything collect_step()/_reset_env()/run() touch on self.env, so those simply can't
        be called this way. Used by offline_pretrain.py, which only ever calls train_step()
        against a dataset loaded straight into buffer_a/buffer_b (see that script).
        """
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

        if config.compile:
            # Deep-copy above happens on the eager modules first -- compiling, then deepcopy-ing
            # the OptimizedModule wrapper is the fragile order. select_actions() calls these nets
            # at a small batch with n_quantiles=cfg.n_quantiles (net/ema_net) while train_step
            # calls them at batch_size/batch_size*spr_k with n_quantiles in {cfg.n_quantiles, 1}
            # (net+target_net, ema_net) -- expect one recompile per distinct (batch, n_quantiles)
            # shape encountered, not a recompile every call, since each shape repeats every step.
            self.net = torch.compile(self.net)
            self.target_net = torch.compile(self.target_net)
            self.ema_net = torch.compile(self.ema_net)
            print("[compile] net/target_net/ema_net wrapped with torch.compile")

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

        # graphic_encoder gets its own AdamW param group (see QMIXConfig.encoder_lr/
        # encoder_weight_decay) so its lr/weight_decay can be tuned independently of the rest
        # of the net -- defaults to the exact same lr/weight_decay as everything else, so this
        # is a no-op until those fields are explicitly overridden.
        encoder_params = list(self.net.graphic_encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.net.parameters() if id(p) not in encoder_param_ids]
        other_params += list(self.dist_mixer.parameters()) + list(self.spr_predictor.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay},
                {
                    "params": encoder_params,
                    "lr": config.encoder_lr if config.encoder_lr is not None else config.lr,
                    "weight_decay": (
                        config.encoder_weight_decay
                        if config.encoder_weight_decay is not None
                        else config.weight_decay
                    ),
                },
            ]
        )
        # Reference to the encoder's own param group (index fixed by construction order above),
        # plus its target (post-warmup) lr -- see reset_warmup_steps / _apply_encoder_lr_warmup().
        self._encoder_param_group = self.optimizer.param_groups[1]
        self._encoder_target_lr = self._encoder_param_group["lr"]

        if env is not None:
            graphic_shape = env.observation_space(team_a_agents()[0])["graphic"].shape
        else:
            # Mirrors BlackOutEnv's own default semantic-config resolution and preprocessor
            # construction (see BlackOutEnv._DEFAULT_CONFIG / __init__) so the channel count
            # matches exactly without needing a live Unity process just to ask it. Map size is
            # fixed at 24x24 everywhere else in this file (see main()'s BlackOutEnv(map_w=24,
            # map_h=24, ...) and GraphicEncoder's hardcoded assumption), so it's safe to inline
            # here too.
            default_config = Path(__file__).resolve().parent.parent / "semantic_map_config.json"
            preprocessor = MyObsPreprocessor(
                load_semantic_config(default_config), n_items=config.n_items, n_classes=config.n_classes
            )
            graphic_shape = (24, 24, preprocessor.n_graphic_channels)
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
        self.onpolicy_buffers: dict[int, tuple[SequentialReplayBuffer, SequentialReplayBuffer]] = {}
        if config.batch_source_fracs is not None:
            onpolicy_kwargs = dict(buffer_kwargs, capacity=config.onpolicy_buffer_capacity)
            for source in (SOURCE_SELF_VS_HEURISTIC, SOURCE_SELF_PLAY):
                if config.batch_source_fracs[source] > 0:
                    self.onpolicy_buffers[source] = (
                        SequentialReplayBuffer(**onpolicy_kwargs),
                        SequentialReplayBuffer(**onpolicy_kwargs),
                    )

        # ---- Heuristic bootstrap (see QMIXConfig.heuristic_fill_frac docstring) ----
        # Sizes computed off buffer_a.capacity (not config.buffer_capacity) because
        # SequentialReplayBuffer rounds capacity up to the next power of 2 internally.
        self._heuristic_fill_size = max(1, round(config.heuristic_fill_frac * self.buffer_a.capacity))
        self._train_start_size = max(1, round(config.bootstrap_train_start_frac * self.buffer_a.capacity))
        self.heuristic_a = HeuristicPolicyMixture(seed=config.heuristic_seed_a)
        self.heuristic_b = HeuristicPolicyMixture(seed=config.heuristic_seed_b)
        self._was_bootstrapping = True  # collect_step() flips this and logs the phase-1->2 transition once
        # env_step_count offset for epsilon() -- reset to the transition step so phase 2 always
        # starts exploring at eps_start instead of inheriting however far env_step_count already
        # decayed it during phase 1 (see epsilon()).
        self._phase2_epsilon_anchor_step = 0
        # env_step_count offset for _anneal_frac() -- bumped to the current step every time
        # maybe_reset() actually fires, so n_step/gamma re-anneal from scratch each reset cycle
        # (see _anneal_frac()) instead of only ever once across the whole run.
        self._anneal_cycle_start_step = 0

        self.team_a_agents = team_a_agents()
        self.team_b_agents = team_b_agents()

        self.env_step_count = 0
        self.train_step_count = 0
        self._total_env_steps_hint = 1  # set properly in run(); avoids div-by-zero if train_step() is called standalone
        self._online_is_team_a = True  # re-randomized every episode in _reset_env()
        self._opponent_is_heuristic = False  # re-randomized every episode in _reset_env()
        self._prev_absorption_time_left: float | None = None  # set in _reset_env(); see collect_step

        # ---- TensorBoard logging (train/tb_logger.py) ----
        self.tb = TBLogger(self.cfg.tb_log_dir)

        # Named submodules for per-part weight/gradient-norm logging. MyModel's own
        # sub-encoders/heads plus the two auxiliary networks trained alongside it
        # (dist_mixer, spr_predictor) -- see MyModel's docstring for what each part does.
        # attention_proj (GQA: q/k/v/o projections) and attention_ffn (SwiGLU) logged
        # separately rather than as one combined "attention" norm -- they have very different
        # weight/gradient scales, so lumping all 4 layers' worth of both together into one
        # number masked whichever one was actually exploding/vanishing. nn.ModuleList is
        # itself an nn.Module, so grouping existing submodules this way (no copies -- these
        # are views over self.net.attention.layers[i]'s own gqa/ffn) works directly with
        # TBLogger.weight_norms/grad_norms, which just calls .parameters() on whatever it's given.
        attention_proj = nn.ModuleList(layer.gqa for layer in self.net.attention.layers)
        attention_ffn = nn.ModuleList(layer.ffn for layer in self.net.attention.layers)
        self._tb_net_parts: dict[str, nn.Module] = {
            "graphic_encoder": self.net.graphic_encoder,
            "vector_encoder": self.net.vector_encoder,
            "attention_proj": attention_proj,
            "attention_ffn": attention_ffn,
            "token_type_emb": self.net.token_type_emb,
            "spr_head": self.net.spr_head,
            "q_head": self.net.q_head,
            "dist_mixer": self.dist_mixer,
            "spr_predictor": self.spr_predictor,
        }

        # Populated by _forward_and_loss() each call -- purely additive bookkeeping read back
        # by train_step()/run() for TB logging; does not affect the loss/training math.
        self._last_iqn_loss: float | None = None
        self._last_spr_loss: float | None = None
        self._last_bc_loss: float | None = None
        # Only populated on TB-logging steps (collect_diagnostics=True) -- each .item() is a
        # device sync, so these aren't paid for on every gradient step.
        self._last_diagnostics: dict[str, float] = {}
        self._last_td_error_mean: float | None = None
        self._last_q_mean: float | None = None
        self._last_q_std: float | None = None
        # Per-layer attention logit RMS, populated by _forward_and_loss() only when asked
        # (see collect_attention_stats there) -- empty otherwise.
        self._last_attention_logit_rms: list[float] = []

        # Episode-outcome bookkeeping for TB (return/win-rate), independent of training.
        self._episode_return_a = 0.0
        self._episode_return_b = 0.0
        self._episode_count = 0
        self._episode_wins_a = 0
        self._episode_wins_b = 0
        # Same win tally, but by self-play ROLE (online net vs its slow EMA opponent, see
        # _online_is_team_a / module docstring) rather than by physical team -- team_a/b win
        # rate is confounded because which physical team plays "online" is re-randomized every
        # episode, so it can't tell you whether the online policy is actually improving.
        self._episode_wins_online = 0
        self._episode_wins_opponent = 0
        self._episode_draws = 0
        # Same online win tally, split by what kind of opponent this episode had (see
        # heuristic_opponent_frac) -- lets win_rate_selfplay's "online" figure be checked
        # against a real, fully-executing heuristic instead of only ever ema_net (a slow copy
        # of itself, which can't reveal a mutually-collapsed self-play equilibrium).
        self._episode_count_vs_heuristic = 0
        self._episode_wins_vs_heuristic = 0
        self._episode_count_vs_selfplay = 0
        self._episode_wins_vs_selfplay = 0

        # Wall-clock breakdown accumulators (reset every print window in run()) — lets you see
        # whether wall-clock is spent waiting on Unity (env.step, gRPC round-trip) vs on-policy
        # forward pass (select_actions) vs gradient updates (train_step), instead of guessing
        # from CPU/GPU utilization alone.
        self._time_select = 0.0
        self._time_env_step = 0.0
        self._time_train = 0.0
        self._time_prep = 0.0
        self._time_forward = 0.0
        self._time_backward = 0.0
        self._time_priority = 0.0

    def _sync(self) -> None:
        """Blocks until queued device work finishes -- MPS/CUDA dispatch is async, so without
        this, perf_counter() boundaries around device ops would time queuing, not execution."""
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    @property
    def _bootstrapping(self) -> bool:
        """
        True while still in phase 1 (pure-heuristic buffer fill, see QMIXConfig.heuristic_fill_frac).
        Buffer size only ever grows (SequentialReplayBuffer._size is a monotonic high-water mark,
        even once it starts wrapping), so this is a one-way gate: True -> False exactly once
        per run, never back.
        """
        return len(self.buffer_a) < self._heuristic_fill_size or len(self.buffer_b) < self._heuristic_fill_size

    def epsilon(self) -> float:
        # Offset by _phase2_epsilon_anchor_step (set once, when phase 1 ends -- see collect_step)
        # so the schedule always starts at eps_start right as phase 2 begins, instead of
        # inheriting however far raw env_step_count already decayed it during phase 1's
        # heuristic-only collection (during which epsilon() isn't even used for action
        # selection, but would otherwise have been silently ticking down anyway).
        frac = min(1.0, max(0.0, self.env_step_count - self._phase2_epsilon_anchor_step) / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _apply_encoder_lr_warmup(self) -> None:
        """Linear warmup of the encoder param group's lr from 0 -> _encoder_target_lr over the
        first reset_warmup_steps env steps of the current reset cycle -- see
        QMIXConfig.reset_warmup_steps. No-op (lr stays at _encoder_target_lr) when disabled."""
        if self.cfg.reset_warmup_steps <= 0:
            return
        steps_since_reset = self.env_step_count - self._anneal_cycle_start_step
        scale = min(1.0, max(0.0, steps_since_reset / self.cfg.reset_warmup_steps))
        self._encoder_param_group["lr"] = self._encoder_target_lr * scale

    def _anneal_frac(self) -> float:
        # BBF resets AND re-anneals n_step/gamma every cycle -- annealing once across the
        # whole run would leave both frozen at their END values (n_step_end, gamma_end) for
        # nearly all of training after the first cycle, defeating half the point of resetting:
        # a freshly shrink-and-perturbed, less-converged net benefits from short-horizon,
        # low-variance n-step returns again, same as at the very start (see maybe_reset()'s
        # reset_alpha docstring / Ash & Adams 2020). With periodic reset disabled
        # (reset_interval == 0, the default), cycle_len falls back to the whole run,
        # reproducing the original one-shot anneal exactly.
        cycle_len = self.cfg.reset_interval if self.cfg.reset_interval > 0 else self._total_env_steps_hint
        anneal_steps = max(1, self.cfg.anneal_frac * cycle_len)
        return (self.env_step_count - self._anneal_cycle_start_step) / anneal_steps

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

    def _heuristic_direction_idx(self, obs: dict[str, dict[str, np.ndarray]], agents: list[str], heuristic) -> np.ndarray:
        """
        Runs one heuristic policy for one team's 5 agents and snaps its continuous (dx,dy)
        output to the nearest of the 8 compass DIRECTION_VECTORS (see direction_vector_to_idx).
        Returns [N_TEAM] int array, in `agents` slot order.
        """
        team_obs = {a: obs[a] for a in agents}
        heuristic_actions = heuristic.act(team_obs)
        vectors = np.stack([heuristic_actions[a] for a in agents])  # [N_TEAM, 2]
        return direction_vector_to_idx(vectors)

    def _pack_direction_idx(self, direction_idx: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """direction_idx: [2, N_TEAM] (team A row, team B row) -> (env_actions dict, full_direction_idx [10])."""
        env_actions = {}
        full_direction_idx = np.zeros(N_AGENTS, dtype=np.int64)
        for team_idx, agents in enumerate((self.team_a_agents, self.team_b_agents)):
            for slot, agent in enumerate(agents):
                d = int(direction_idx[team_idx, slot])
                env_actions[agent] = DIRECTION_VECTORS[d]
                full_direction_idx[team_idx * N_TEAM + slot] = d
        return env_actions, full_direction_idx

    def select_actions_heuristic(self, obs: dict[str, dict[str, np.ndarray]]):
        """
        Phase 1 (see `_bootstrapping`): BOTH teams act via HeuristicPolicyMixture -- no
        net/ema_net forward pass at all, since Q isn't driving any decision yet. The SNAPPED
        direction (not the heuristic's raw continuous vector) is what actually gets sent to
        env.step(), so the action stored in the replay buffer always matches what was physically
        executed -- required for off-policy Q-learning to train against the right transition.

        With probability `heuristic_bootstrap_noise_frac` (0 by default -- exactly the old,
        pure-heuristic behavior), each unit's action is instead a uniformly random compass
        direction. Pure heuristic-vs-heuristic data only ever visits the narrow slice of
        (state, action) space the heuristics themselves choose to visit, which starves offline
        Q-learning of the off-heuristic-action coverage it needs to evaluate alternatives the
        model might pick later -- a little uniform-random noise widens that coverage without
        giving up "the trajectories still mostly look like competent play" the way pure random
        rollouts would.
        """
        dir_a = self._heuristic_direction_idx(obs, self.team_a_agents, self.heuristic_a)
        dir_b = self._heuristic_direction_idx(obs, self.team_b_agents, self.heuristic_b)
        direction_idx = np.stack([dir_a, dir_b])
        noise_frac = self.cfg.heuristic_bootstrap_noise_frac
        if noise_frac > 0:
            random_dirs = np.random.randint(0, 8, size=direction_idx.shape)
            direction_idx = np.where(np.random.rand(*direction_idx.shape) < noise_frac, random_dirs, direction_idx)
        return self._pack_direction_idx(direction_idx)

    @torch.no_grad()
    def select_actions(self, obs: dict[str, dict[str, np.ndarray]], epsilon: float):
        """
        Phase 2 only (see `_bootstrapping`). Returns (env_actions, full_direction_idx) where
        env_actions is the dict[agent,(dx,dy)] BlackOutEnv.step() expects, and
        full_direction_idx is [10] (physical unit order, both teams) for the replay buffer.

        This episode's "opponent" side (see `self._online_is_team_a`, module docstring) acts
        through `ema_net` instead of `net` for self-play stability -- except when this episode
        was sampled (see `heuristic_opponent_frac`) to have a full-heuristic opponent instead,
        in which case the opponent's row is entirely overridden by its HeuristicPolicyMixture
        action below rather than ema_net's greedy Q.
        """
        obs_a, obs_b = obs[self.team_a_agents[0]], obs[self.team_b_agents[0]]
        # Team B acts on the canonical (mirrored) view the network is trained on -- see
        # _mirror_batch / blackout_env.env.team_frame. Its own units land in rows 0-4 there,
        # in ascending physical order, so `greedy[1]` still lines up slot-for-slot with
        # team_b_agents; only the chosen compass index has to be reflected back to world.
        graphic, team_state, agent_states = self._to_batch(obs_a, canonical_obs(obs_b))

        q_online, *_ = self.net(graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles)  # [2,10,8]
        q_ema, *_ = self.ema_net(graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles)

        opponent_idx = 1 if self._online_is_team_a else 0
        q_values = q_online.clone()
        q_values[opponent_idx] = q_ema[opponent_idx]

        own_q = _own_team_rows(q_values, agent_states)  # [2, N_TEAM, 8]
        greedy = (
            masked_greedy(own_q, graphic, agent_states) if self.cfg.action_masking else own_q.argmax(dim=-1)
        ).cpu().numpy()  # [2, N_TEAM]
        greedy[1] = mirror_direction_idx(greedy[1])  # canonical -> world, team B only

        # Explore branch: this episode's heuristic-mixture action (guided toward objectives,
        # not a uniform-random compass walk -- see conversation), snapped to the discrete
        # compass space. Still costs one heuristic .act() per team per step regardless of how
        # small epsilon has annealed to; cheap relative to the net/ema_net forward passes
        # above, so not worth conditioning on the epsilon draw first.
        dir_a = self._heuristic_direction_idx(obs, self.team_a_agents, self.heuristic_a)
        dir_b = self._heuristic_direction_idx(obs, self.team_b_agents, self.heuristic_b)
        heuristic_dirs = np.stack([dir_a, dir_b])  # [2, N_TEAM]

        direction_idx = np.where(
            np.random.rand(2, N_TEAM) < epsilon,
            heuristic_dirs,
            greedy,
        )

        if self._opponent_is_heuristic:
            # Full-episode heuristic opponent: override its row entirely (ignore ema_net's
            # greedy Q and the epsilon draw above -- every step this team acts purely through
            # its HeuristicPolicyMixture, not just epsilon's usual exploration fraction of them).
            direction_idx[opponent_idx] = heuristic_dirs[opponent_idx]

        return self._pack_direction_idx(direction_idx)

    def _reset_env(self):
        """
        Resets the env and re-randomizes which physical team is this episode's "online" (live
        `net`) side vs the EMA-controlled "opponent" side (see module docstring).
        """
        obs, info = self.env.reset()
        self._online_is_team_a = bool(np.random.rand() < 0.5)
        self._opponent_is_heuristic = bool(np.random.rand() < self.cfg.heuristic_opponent_frac)
        self._prev_absorption_time_left = float(obs[self.team_a_agents[0]]["team_state"][ABSORPTION_IDX])
        # Explicit re-sample each episode (HeuristicPolicyMixture also self-detects a new
        # episode via a time_left jump in .act(), but resetting here keeps it in lockstep with
        # the rest of this method's per-episode bookkeeping instead of relying on that sniff).
        self.heuristic_a.reset()
        self.heuristic_b.reset()
        return obs, info

    def collect_step(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        bootstrapping = self._bootstrapping
        if self._was_bootstrapping and not bootstrapping:
            # One-way transition (see _bootstrapping) -- anchor epsilon() here so phase 2
            # starts exploring at eps_start instead of wherever raw env_step_count left it.
            self._phase2_epsilon_anchor_step = self.env_step_count
            print(
                f"[bootstrap] buffers reached heuristic_fill_frac at env step {self.env_step_count} "
                "-- switching from pure-heuristic to epsilon-mixed (model + heuristic) action selection"
            )
        self._was_bootstrapping = bootstrapping

        t0 = time.perf_counter()
        if bootstrapping:
            env_actions, full_direction_idx = self.select_actions_heuristic(obs)
        else:
            env_actions, full_direction_idx = self.select_actions(obs, self.epsilon())
        t1 = time.perf_counter()

        obs_a, obs_b = obs[self.team_a_agents[0]], obs[self.team_b_agents[0]]
        next_obs, rewards, terminations, _, infos = self.env.step(env_actions)
        t2 = time.perf_counter()
        self._time_select += t1 - t0
        self._time_env_step += t2 - t1
        done = any(terminations.values())

        # AbsorptionTimer loops in Unity (see GameScenario/TimerManager): the observed
        # absorption_time_left counts down 1.0->0.0 each step and snaps back up near 1.0 on the
        # exact tick an absorption fires, so an increase between consecutive steps IS that event
        # -- there's no separate boolean for it anywhere in obs/info (see reward_proposal.md /
        # ml_agent_design.md for why absorption, not the ~600s full match, is this game's natural
        # reward/credit-assignment horizon). A full match is far too long to bootstrap a single
        # n-step/SPR window over, so each absorption interval is treated as its own training
        # episode boundary here -- same "life lost = episode end" trick classic DQN Atari agents
        # use -- WITHOUT touching the real Unity match/reset (self._online_is_team_a, score,
        # win/loss bookkeeping below all still track the real ~600s match untouched). This does
        # mean the Q-target treats "just after an absorption" as if no further reward exists,
        # which is a deliberate bias traded for a much shorter, more learnable horizon.
        absorption_time_left = float(next_obs[self.team_a_agents[0]]["team_state"][ABSORPTION_IDX])
        absorption_fired = (
            self._prev_absorption_time_left is not None
            and absorption_time_left > self._prev_absorption_time_left + 1e-6
        )
        self._prev_absorption_time_left = absorption_time_left
        buffer_done = done or absorption_fired

        reward_a = sum(rewards[a] for a in self.team_a_agents)
        reward_b = sum(rewards[a] for a in self.team_b_agents)
        self._episode_return_a += reward_a
        self._episode_return_b += reward_b

        # Phase-1 bootstrap rows are pure heuristic play. In phase 2 only a full-episode heuristic
        # opponent's own stream is still a demonstration; the online side (and an ema_net
        # opponent) carries the net's own epsilon-mixed actions (see bc_loss_alpha).
        if bootstrapping:
            source, demo_a, demo_b = SOURCE_DATASET, True, True
        elif self._opponent_is_heuristic:
            source = SOURCE_SELF_VS_HEURISTIC
            demo_a, demo_b = not self._online_is_team_a, self._online_is_team_a
        else:
            source, demo_a, demo_b = SOURCE_SELF_PLAY, False, False
        self.buffer_a.push(obs_a["graphic"], obs_a["team_state"], obs_a["agent_states"], full_direction_idx, reward_a, buffer_done, source, demo_a)
        self.buffer_b.push(obs_b["graphic"], obs_b["team_state"], obs_b["agent_states"], full_direction_idx, reward_b, buffer_done, source, demo_b)

        self.env_step_count += 1
        if self.env_step_count % self.cfg.tb_log_interval == 0:
            self.tb.scalars("reward/step", {"team_a": reward_a, "team_b": reward_b}, self.env_step_count)
            self.tb.scalar("bootstrap/is_heuristic_fill_phase", float(bootstrapping), self.env_step_count)
            self.tb.scalar("bootstrap/is_heuristic_opponent_episode", float(self._opponent_is_heuristic), self.env_step_count)

        if done:
            # Physical winner straight from Unity's own score comparison (see
            # BlackOutEnv._collect_obs / MatchManager.cs) -- NOT episode-return comparison.
            # Per-step shaping is not zero-sum (see match.py), so summed team returns can
            # disagree with who actually won the match; this reads the real outcome instead.
            physical_winner = next(iter(infos.values())).get("winner") if infos else None
            self._episode_count += 1
            team_a_won = physical_winner == 0
            team_b_won = physical_winner == 1
            if team_a_won:
                self._episode_wins_a += 1
            elif team_b_won:
                self._episode_wins_b += 1
            else:
                self._episode_draws += 1
            # self._online_is_team_a still reflects the episode that just ended -- _reset_env()
            # (which re-randomizes it for the NEXT episode) hasn't run yet at this point.
            online_won = (team_a_won and self._online_is_team_a) or (team_b_won and not self._online_is_team_a)
            opponent_won = (team_a_won and not self._online_is_team_a) or (team_b_won and self._online_is_team_a)
            if online_won:
                self._episode_wins_online += 1
            elif opponent_won:
                self._episode_wins_opponent += 1
            # self._opponent_is_heuristic likewise still reflects the episode that just ended.
            if self._opponent_is_heuristic:
                self._episode_count_vs_heuristic += 1
                if online_won:
                    self._episode_wins_vs_heuristic += 1
            else:
                self._episode_count_vs_selfplay += 1
                if online_won:
                    self._episode_wins_vs_selfplay += 1
            self.tb.scalars(
                "episode/return",
                {"team_a": self._episode_return_a, "team_b": self._episode_return_b},
                self.env_step_count,
            )
            self.tb.scalars(
                "episode/win_rate",
                {"team_a": self._episode_wins_a / self._episode_count, "team_b": self._episode_wins_b / self._episode_count},
                self.env_step_count,
            )
            self.tb.scalar("episode/draw_rate", self._episode_draws / self._episode_count, self.env_step_count)
            self.tb.scalars(
                "episode/win_rate_selfplay",
                {
                    "online": self._episode_wins_online / self._episode_count,
                    "opponent": self._episode_wins_opponent / self._episode_count,
                },
                self.env_step_count,
            )
            if self._episode_count_vs_heuristic > 0:
                self.tb.scalar(
                    "episode/win_rate_online_vs_heuristic",
                    self._episode_wins_vs_heuristic / self._episode_count_vs_heuristic,
                    self.env_step_count,
                )
            if self._episode_count_vs_selfplay > 0:
                self.tb.scalar(
                    "episode/win_rate_online_vs_selfplay",
                    self._episode_wins_vs_selfplay / self._episode_count_vs_selfplay,
                    self.env_step_count,
                )
            self._episode_return_a = self._episode_return_b = 0.0
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

    # Fields that live on the anchor-sampled batch and are safe to concatenate along dim 0
    # across streams (buffer_a, buffer_b) before a single shared forward pass. "indices" is
    # deliberately excluded -- each stream's indices stay meaningful only against that
    # stream's own buffer, so they're kept separate for update_priorities.
    _MERGE_FIELDS = (
        "is_weights", "graphic", "team_state", "agent_states", "actions",
        "n_step_return", "not_done", "gamma_eff",
        "boot_graphic", "boot_team_state", "boot_agent_states",
        "future_graphic", "future_team_state", "future_agent_states",
        "action_window", "valid_mask", "source", "demo",
    )

    def _sample_batch(
        self, buffer: SequentialReplayBuffer, batch_size: int, n_step: int, gamma: float, beta: float,
        normalize: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Pure-numpy anchor sampling + n-step/SPR window bookkeeping for ONE stream. Deliberately
        does no tensor/device work and no network calls, so buffer_a's and buffer_b's batches
        can be concatenated into a single forward pass per network (see _forward_and_loss)
        instead of two separate ones -- MPS/CUDA pay a largely fixed per-launch dispatch cost,
        so halving the number of forward invocations matters more here than the FLOPs saved
        (measured: forward was ~60% of train_step wall time, dominated by launch count, not
        compute, at this batch size).
        """
        window = max(n_step, self.cfg.spr_k)
        batch = buffer.sample(batch_size, window=window, beta=beta, normalize=normalize)
        indices = batch["indices"]

        n_step_return, bootstrap_idx, not_done, gamma_eff = compute_n_step_return(
            buffer.reward, buffer.done, indices, buffer.capacity, n_step, gamma
        )

        offsets_future = np.arange(1, self.cfg.spr_k + 1)
        offsets_action = np.arange(0, self.cfg.spr_k)
        valid_mask = compute_spr_valid_mask(buffer.done, indices, buffer.capacity, self.cfg.spr_k)

        return {
            "indices": indices,
            "is_weights": batch["is_weights"],
            "graphic": batch["graphic"],
            "team_state": batch["team_state"],
            "agent_states": batch["agent_states"],
            "actions": batch["actions"],
            "n_step_return": n_step_return,
            "not_done": not_done,
            "gamma_eff": gamma_eff,
            "boot_graphic": buffer.graphic[bootstrap_idx],
            "boot_team_state": buffer.team_state[bootstrap_idx],
            "boot_agent_states": buffer.agent_states[bootstrap_idx],
            "future_graphic": buffer.read_window(indices, offsets_future, "graphic"),  # [b, K, H, W, C]
            "future_team_state": buffer.read_window(indices, offsets_future, "team_state"),
            "future_agent_states": buffer.read_window(indices, offsets_future, "agent_states"),
            "action_window": buffer.read_window(indices, offsets_action, "actions"),  # [b, K, 10]
            "valid_mask": valid_mask,  # [b, K] bool
            "source": batch["source"],  # [b] int8
            "demo": batch["demo"],  # [b] bool, BC target
        }

    @staticmethod
    def _mirror_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Maps a team-B stream batch into the canonical (team-A-looking) frame -- see
        blackout_env.env.team_frame. Stored transitions stay in raw world coordinates; the
        reflection is applied here, per sampled batch, so the network only ever sees one
        orientation and the two streams' data reinforce each other instead of splitting the
        network's capacity across two mirrored versions of the same task.
        """
        mirrored = dict(batch)
        for field in ("graphic", "boot_graphic", "future_graphic"):
            mirrored[field] = mirror_graphic(batch[field])
        for field in ("agent_states", "boot_agent_states", "future_agent_states"):
            mirrored[field] = mirror_agent_states(batch[field])
        for field in ("actions", "action_window"):
            mirrored[field] = mirror_action_idx(batch[field])
        return mirrored

    def _merge_stream_batches(self, batch_a: dict[str, np.ndarray], batch_b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Concatenates two streams' sampled batches along dim 0 for one shared forward pass."""
        return {field: np.concatenate([batch_a[field], batch_b[field]], axis=0) for field in self._MERGE_FIELDS}

    def _stream_buffers(self, stream: int) -> list[tuple[int, SequentialReplayBuffer]]:
        """(source, buffer) for every buffer one stream (0 = team A, 1 = team B) samples from."""
        main = self.buffer_a if stream == 0 else self.buffer_b
        return [(SOURCE_DATASET, main)] + [(src, pair[stream]) for src, pair in self.onpolicy_buffers.items()]

    def _sample_stream(
        self, stream: int, batch_size: int, n_step: int, gamma: float, beta: float
    ) -> tuple[dict[str, np.ndarray], list[tuple[SequentialReplayBuffer, int]]]:
        """
        One stream's batch plus (buffer, n_rows) parts in batch order, for routing priority updates
        back. With batch_source_fracs set, each source contributes its fixed quota (largest-remainder
        rounding); a source whose buffer can't serve the n-step/SPR window yet gives its share to the
        rest. Importance weights are normalized over the combined batch, so they correct priority
        skew within each buffer but leave the chosen between-source ratio alone.
        """
        if self.cfg.batch_source_fracs is None:
            main = self.buffer_a if stream == 0 else self.buffer_b
            batch = self._sample_batch(main, batch_size, n_step, gamma, beta)
            return (batch if stream == 0 else self._mirror_batch(batch)), [(main, batch_size)]

        min_rows = max(n_step, self.cfg.spr_k) + 1
        usable = [(buf, self.cfg.batch_source_fracs[src]) for src, buf in self._stream_buffers(stream) if len(buf) > min_rows]
        fracs = np.array([frac for _, frac in usable], dtype=np.float64)
        raw = fracs / fracs.sum() * batch_size
        quotas = np.floor(raw).astype(np.int64)
        remainder = batch_size - int(quotas.sum())
        if remainder > 0:
            quotas[np.argsort(-(raw - quotas), kind="stable")[:remainder]] += 1

        parts = [(buf, int(q)) for (buf, _), q in zip(usable, quotas) if q > 0]
        batches = [self._sample_batch(buf, q, n_step, gamma, beta, normalize=False) for buf, q in parts]
        batch = {field: np.concatenate([b[field] for b in batches], axis=0) for field in batches[0]}
        batch["is_weights"] = batch["is_weights"] / batch["is_weights"].max()
        return (batch if stream == 0 else self._mirror_batch(batch)), parts

    def _log_source_stats(self, merged: dict[str, np.ndarray], td_error_np: np.ndarray) -> None:
        """Splits buffer composition and this batch's return/TD-error by transition origin
        (static dataset vs on-policy collection), so a drift in Q can be traced to which data
        is driving it."""
        buffers = self._stream_buffers(0) + self._stream_buffers(1)
        total = sum(buf.source_counts for _, buf in buffers)
        n_stored = max(1, int(total.sum()))
        self.tb.scalars(
            "buffer_source_frac",
            {name: float(total[i]) / n_stored for i, name in enumerate(SOURCE_NAMES)},
            self.train_step_count,
        )
        source = merged["source"]
        returns = merged["n_step_return"]
        batch_frac, batch_return, batch_td = {}, {}, {}
        for i, name in enumerate(SOURCE_NAMES):
            mask = source == i
            batch_frac[name] = float(mask.mean())
            if mask.any():
                batch_return[name] = float(returns[mask].mean())
                batch_td[name] = float(td_error_np[mask].mean())
        self.tb.scalars("batch_source_frac", batch_frac, self.train_step_count)
        self.tb.scalars(
            "buffer_rows",
            {SOURCE_NAMES[src]: float(sum(len(buf) for s, buf in buffers if s == src)) for src, _ in self._stream_buffers(0)},
            self.train_step_count,
        )
        self.tb.scalars(
            "per_max_priority",
            {SOURCE_NAMES[src]: max(buf._max_priority for s, buf in buffers if s == src) for src, _ in self._stream_buffers(0)},
            self.train_step_count,
        )
        if batch_return:
            self.tb.scalars("batch_n_step_return", batch_return, self.train_step_count)
            self.tb.scalars("batch_td_error", batch_td, self.train_step_count)

    def _to_tensors(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Numpy -> device tensors for every _MERGE_FIELDS entry except the SPR future_* window
        fields, which stay numpy here and are reshaped+converted in _forward_and_loss right
        before the (B*K)-flattened ema_net call -- unchanged from the original single-stream
        code's ordering, just relocated."""
        return {
            "graphic": self._to_graphic_tensor(batch["graphic"]),
            "team_state": torch.tensor(batch["team_state"], dtype=torch.float32, device=self.device),
            "agent_states": torch.tensor(batch["agent_states"], dtype=torch.float32, device=self.device),
            "actions_full": torch.tensor(batch["actions"], dtype=torch.long, device=self.device),  # [B, 10]
            "is_weights": torch.tensor(batch["is_weights"], dtype=torch.float32, device=self.device),
            "n_step_return": torch.tensor(batch["n_step_return"], dtype=torch.float32, device=self.device),
            "not_done": torch.tensor(batch["not_done"], dtype=torch.float32, device=self.device),
            "gamma_eff": torch.tensor(batch["gamma_eff"], dtype=torch.float32, device=self.device),
            "boot_graphic": self._to_graphic_tensor(batch["boot_graphic"]),
            "boot_team_state": torch.tensor(batch["boot_team_state"], dtype=torch.float32, device=self.device),
            "boot_agent_states": torch.tensor(batch["boot_agent_states"], dtype=torch.float32, device=self.device),
            "future_graphic": batch["future_graphic"],
            "future_team_state": batch["future_team_state"],
            "future_agent_states": batch["future_agent_states"],
            "action_window": torch.tensor(batch["action_window"], dtype=torch.long, device=self.device),  # [B, K, 10]
            "valid_mask": torch.tensor(batch["valid_mask"], dtype=torch.float32, device=self.device),  # [B, K]
            "bc_mask": torch.tensor(batch["demo"], dtype=torch.float32, device=self.device),  # [B]
        }

    def _forward_and_loss(
        self, t: dict[str, torch.Tensor], collect_attention_stats: bool = False, collect_diagnostics: bool = False
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Runs net/target_net/ema_net/mixers ONCE on the (already-merged) batch `t` and
        returns (total_loss_per_sample [B], td_error [B]). Same math as before the A/B merge --
        every op here is per-sample (self-attention within a sample's own tokens, no
        cross-sample mixing), so batching two streams together is equivalent to running them
        separately and concatenating the results, just fewer/larger kernel launches.

        collect_attention_stats: caller (train_step, only on TB-logging steps) wants
        self._last_attention_logit_rms populated from THIS call specifically -- the online net
        is also called again below for the bootstrap action (under no_grad), which would
        otherwise silently overwrite each layer's GroupedQueryAttention.last_logit_norm with
        the wrong call's numbers if we read it after both calls instead of right after this one.
        """
        graphic, team_state, agent_states = t["graphic"], t["team_state"], t["agent_states"]
        actions_full, is_weights = t["actions_full"], t["is_weights"]
        own_actions = _own_team_rows(actions_full.unsqueeze(-1), agent_states).squeeze(-1)  # [B, N_TEAM]

        if collect_attention_stats:
            for layer in self.net.attention.layers:
                layer.gqa.log_attention_stats = True

        # ---- online forward (current state): Q-learning prediction + SPR rollout start ----
        q_values, quantile_values, tau, vision_latent, global_latent = self.net(
            graphic, team_state, agent_states, n_quantiles=self.cfg.n_quantiles
        )

        if collect_attention_stats:
            self._last_attention_logit_rms = [layer.gqa.last_logit_norm for layer in self.net.attention.layers]
            for layer in self.net.attention.layers:
                layer.gqa.log_attention_stats = False

        own_quantiles = _own_team_rows(quantile_values, agent_states)  # [B, N_TEAM, Q, n_actions]
        chosen_quantiles = torch.gather(
            own_quantiles, 3, own_actions.view(*own_actions.shape, 1, 1).expand(-1, -1, self.cfg.n_quantiles, 1)
        ).squeeze(-1)  # [B, N_TEAM, Q]
        q_tot_online = self.dist_mixer(chosen_quantiles, global_latent)  # [B, Q]

        # ---- Double DQN target: online net picks the bootstrap action, target net evaluates it ----
        with torch.no_grad():
            boot_q_online, *_ = self.net(
                t["boot_graphic"], t["boot_team_state"], t["boot_agent_states"], n_quantiles=self.cfg.n_quantiles
            )
            boot_own_q_online = _own_team_rows(boot_q_online, t["boot_agent_states"])  # [B, N_TEAM, A]
            boot_greedy = (
                masked_greedy(boot_own_q_online, t["boot_graphic"], t["boot_agent_states"])
                if self.cfg.action_masking
                else boot_own_q_online.argmax(dim=-1)
            )  # [B, N_TEAM]

            _, boot_quantiles_target, _, _, boot_global_target = self.target_net(
                t["boot_graphic"], t["boot_team_state"], t["boot_agent_states"], n_quantiles=self.cfg.n_quantiles
            )
            boot_own_quantiles_target = _own_team_rows(boot_quantiles_target, t["boot_agent_states"])  # [B, N_TEAM, Q', A]
            boot_chosen_target = torch.gather(
                boot_own_quantiles_target, 3, boot_greedy.view(*boot_greedy.shape, 1, 1).expand(-1, -1, self.cfg.n_quantiles, 1)
            ).squeeze(-1)  # [B, N_TEAM, Q']
            q_tot_target = self.target_dist_mixer(boot_chosen_target, boot_global_target)  # [B, Q']

            target = (
                t["n_step_return"].unsqueeze(1)
                + t["gamma_eff"].unsqueeze(1) * t["not_done"].unsqueeze(1) * q_tot_target
            )

        iqn_loss = quantile_huber_loss(q_tot_online, tau, target)  # [B]
        with torch.no_grad():
            td_error = (q_tot_online.mean(dim=1) - target.mean(dim=1)).abs()
            # Q-magnitude bookkeeping for TB: DQN-family divergence typically shows up here
            # (mean/std creeping up) well before it shows up in the loss curve.
            self._last_q_mean = q_tot_online.mean().item()
            self._last_q_std = q_tot_online.std().item()

        # ---- SPR: open-loop K-step latent rollout vs EMA target encoder ----
        # vision_latent is already [B, hidden] -- MyModel's dedicated SPR CLS token (see its
        # class docstring "SPR CLS token"), not a mean-pool over the 36 vision tokens anymore.
        pooled_vision = vision_latent  # [B, hidden]
        Bsz = graphic.shape[0]
        K = self.cfg.spr_k
        future_graphic, future_team_state, future_agent_states = (
            t["future_graphic"], t["future_team_state"], t["future_agent_states"]
        )
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
            target_pooled = ema_vision_latent.view(Bsz, K, -1)  # [B, K, hidden] -- already pooled by ema_net's SPR CLS token

        z = pooled_vision
        spr_losses = []
        for k in range(K):
            z = self.spr_predictor.step(z, t["action_window"][:, k, :])
            spr_losses.append(self.spr_predictor.loss(z, target_pooled[:, k, :]))
        spr_losses = torch.stack(spr_losses, dim=1)  # [B, K]
        spr_loss = (spr_losses * t["valid_mask"]).sum(dim=1) / (t["valid_mask"].sum(dim=1) + 1e-6)  # [B]

        # ---- Behavior cloning (see QMIXConfig.bc_loss_alpha): cross-entropy between the
        # online net's own-team Q-values (as classification logits) and the actual action the
        # dataset's behavior policy took in this state.
        own_q_values = _own_team_rows(q_values, agent_states)  # [B, N_TEAM, n_actions]
        bc_loss_raw = F.cross_entropy(
            own_q_values.reshape(-1, own_q_values.shape[-1]), own_actions.reshape(-1), reduction="none"
        ).view(own_actions.shape).mean(dim=1)  # [B]
        # Clone only demonstration transitions (see QMIXConfig.bc_loss_alpha). With every sample a
        # demonstration this reduces exactly to the unmasked loss; otherwise each cloned sample
        # keeps the same weight and net-played samples contribute nothing.
        bc_mask = t["bc_mask"]
        n_bc = bc_mask.sum().clamp_min(1.0)
        # TD3+BC-style adaptive rescale (Fujimoto & Gu 2021): match bc_loss's current magnitude
        # to iqn_loss's, THEN apply bc_loss_alpha -- so alpha is a dimensionless "how much BC
        # relative to TD" knob that stays meaningful throughout training even though the two
        # losses' raw scales don't (see QMIXConfig.bc_loss_alpha for why a literal fixed weight
        # doesn't transfer well here). detached on both sides: this is a scale correction, not a
        # gradient path, and no_grad avoids paying autograd bookkeeping on a ratio of scalars
        # that's immediately used only to rescale (not backprop through) bc_loss_raw.
        with torch.no_grad():
            bc_raw_mean = (bc_loss_raw.detach() * bc_mask).sum() / n_bc
            bc_scale = self.cfg.bc_loss_alpha * (
                iqn_loss.detach().mean().clamp_min(1e-6) / bc_raw_mean.clamp_min(1e-6)
            )
        bc_loss = bc_scale * bc_loss_raw * bc_mask

        total_loss = is_weights * iqn_loss + self.cfg.spr_loss_weight * spr_loss + bc_loss  # [B]

        # Pure bookkeeping for TB logging (train_step reads these back) -- does not feed into
        # the returned loss/td_error at all.
        self._last_iqn_loss = iqn_loss.mean().item()
        self._last_spr_loss = spr_loss.mean().item()
        self._last_bc_loss = bc_loss.mean().item()
        if collect_diagnostics:
            with torch.no_grad():
                # bc_loss above is rescaled to iqn_loss's magnitude every step, so its logged
                # value just mirrors loss/iqn -- the raw CE and argmax agreement are the only way
                # to see whether behavior cloning itself is actually converging.
                top2 = own_q_values.topk(2, dim=-1).values
                agreement = (own_q_values.argmax(dim=-1) == own_actions).float().mean(dim=1)  # [B]
                n_model = (1.0 - bc_mask).sum()
                self._last_diagnostics = {
                    "bc_raw": bc_raw_mean.item(),
                    "bc_scale": float(bc_scale),
                    "bc_active_frac": bc_mask.mean().item(),
                    # Argmax agreement with demonstration actions (what BC trains) and, separately,
                    # with the actions the net itself took in net-played transitions -- the latter
                    # is just how consistent the net stays with its own recent behavior, not an
                    # imitation target.
                    "bc_accuracy": ((agreement * bc_mask).sum() / n_bc).item(),
                    "model_action_agreement": (
                        ((agreement * (1.0 - bc_mask)).sum() / n_model).item() if n_model > 0 else None
                    ),
                    # Per-unit gap between the best and second-best action's Q: how decisive the
                    # greedy policy is, compared against the reward scale of a single move.
                    "q_action_margin": (top2[..., 0] - top2[..., 1]).mean().item(),
                    "q_action_range": (own_q_values.max(dim=-1).values - own_q_values.min(dim=-1).values).mean().item(),
                }

        return total_loss, td_error.cpu().numpy()

    def train_step(self) -> float | None:
        if len(self.buffer_a) < self._train_start_size or len(self.buffer_b) < self._train_start_size:
            return None

        n_step = self.current_n_step()
        gamma = self.current_gamma()
        beta = self.current_per_beta()
        half = self.cfg.batch_size // 2

        losses = []
        for _ in range(self.cfg.grad_steps_per_call):
            t_prep0 = time.perf_counter()
            batch_a, parts_a = self._sample_stream(0, half, n_step, gamma, beta)
            batch_b, parts_b = self._sample_stream(1, half, n_step, gamma, beta)
            merged = self._merge_stream_batches(batch_a, batch_b)
            tensors = self._to_tensors(merged)
            self._sync()
            t_prep1 = time.perf_counter()
            self._time_prep += t_prep1 - t_prep0

            log_tb_this_step = self.tb.enabled and self.train_step_count % self.cfg.tb_log_interval == 0

            total_loss, td_error_np = self._forward_and_loss(
                tensors, collect_attention_stats=log_tb_this_step, collect_diagnostics=log_tb_this_step
            )
            loss = total_loss.mean()
            self._last_td_error_mean = float(td_error_np.mean())
            self._sync()
            t_fwd1 = time.perf_counter()
            self._time_forward += t_fwd1 - t_prep1

            t_bwd0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss.backward()

            if log_tb_this_step:
                # Pre-clip grad norms -- clip_grad_norm_ below mutates grads in place, so this
                # has to run first to see the raw (un-clipped) per-part magnitude.
                self.tb.grad_norms("grad_norm", self._tb_net_parts, self.train_step_count)

            total_grad_norm = nn.utils.clip_grad_norm_(
                itertools.chain(self.net.parameters(), self.dist_mixer.parameters(), self.spr_predictor.parameters()),
                self.cfg.grad_clip,
            )
            self._apply_encoder_lr_warmup()
            self.optimizer.step()
            self._sync()
            t_bwd1 = time.perf_counter()
            self._time_backward += t_bwd1 - t_bwd0

            if log_tb_this_step:
                self.tb.scalar("grad_norm/total_preclip", float(total_grad_norm), self.train_step_count)
                self.tb.weight_norms("weight_norm", self._tb_net_parts, self.train_step_count)
                self.tb.scalars(
                    "attention_logit_rms",
                    {f"layer_{i}": v for i, v in enumerate(self._last_attention_logit_rms)},
                    self.train_step_count,
                )
                self.tb.scalars(
                    "loss",
                    {
                        "total": loss.item(),
                        "iqn": self._last_iqn_loss,
                        "spr": self._last_spr_loss,
                        "bc": self._last_bc_loss,
                        "bc_raw": self._last_diagnostics["bc_raw"],
                    },
                    self.train_step_count,
                )
                self.tb.scalars(
                    "bc",
                    {
                        "accuracy": self._last_diagnostics["bc_accuracy"],
                        "scale": self._last_diagnostics["bc_scale"],
                        "active_frac": self._last_diagnostics["bc_active_frac"],
                        "model_action_agreement": self._last_diagnostics["model_action_agreement"],
                    },
                    self.train_step_count,
                )
                self.tb.scalar("td_error/mean", self._last_td_error_mean, self.train_step_count)
                self.tb.scalars(
                    "q_value",
                    {
                        "mean": self._last_q_mean,
                        "std": self._last_q_std,
                        "action_margin": self._last_diagnostics["q_action_margin"],
                        "action_range": self._last_diagnostics["q_action_range"],
                    },
                    self.train_step_count,
                )
                self._log_source_stats(merged, td_error_np)
                # Monotonicity "clamp pressure" (see clamp_pressure_stats docstring): how hard
                # each mixer hypernetwork's raw (pre-abs) output is being pushed negative by
                # gradient descent, i.e. how much the QMIX non-negative-weight constraint is
                # actively fighting the loss -- the diagnostic for "is monotonicity a real
                # bottleneck here, or would an unconstrained mixer (QPLEX/QTRAN) not actually
                # help" (see qplex_migration_criteria.md for how to read this over training).
                w1_frac_neg, w1_neg_mag = clamp_pressure_stats(self.mixer.last_raw_w1)
                w2_frac_neg, w2_neg_mag = clamp_pressure_stats(self.mixer.last_raw_w2)
                shape_frac_neg, shape_neg_mag = clamp_pressure_stats(self.dist_mixer.last_raw_shape_w)
                self.tb.scalars(
                    "mixer_clamp_pressure/frac_negative",
                    {"hyper_w1": w1_frac_neg, "hyper_w2": w2_frac_neg, "shape_weight": shape_frac_neg},
                    self.train_step_count,
                )
                self.tb.scalars(
                    "mixer_clamp_pressure/neg_magnitude",
                    {"hyper_w1": w1_neg_mag, "hyper_w2": w2_neg_mag, "shape_weight": shape_neg_mag},
                    self.train_step_count,
                )
                self.tb.scalars(
                    "schedule",
                    {
                        "epsilon": self.epsilon(),
                        "n_step": n_step,
                        "gamma": gamma,
                        "per_beta": beta,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    self.train_step_count,
                )

            # merged rows are batch_a's parts in order, then batch_b's -- same order as parts_a + parts_b.
            merged_indices = np.concatenate([batch_a["indices"], batch_b["indices"]])
            offset = 0
            for buf, n_rows in parts_a + parts_b:
                buf.update_priorities(merged_indices[offset:offset + n_rows], td_error_np[offset:offset + n_rows])
                offset += n_rows

            ema_update(self.ema_net, self.net, self.cfg.ema_tau)
            ema_update(self.spr_predictor.target_projector, self.spr_predictor.projector, self.cfg.ema_tau)

            self.train_step_count += 1
            if self.train_step_count % self.cfg.target_update_interval == 0:
                self.target_net.load_state_dict(self.net.state_dict())
                self.target_dist_mixer.load_state_dict(self.dist_mixer.state_dict())

            losses.append(loss.item())
            self._time_priority += time.perf_counter() - t_bwd1

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

        self._reset_submodule("graphic_encoder", lambda: GraphicEncoder(hidden_size=self.cfg.hidden_size), self.cfg.reset_alpha_cnn)
        self._reset_submodule(
            "attention",
            lambda: AttentionLayers(self.cfg.hidden_size, N_ATTENTION_HEADS, ATTENTION_DEPTH),
            self.cfg.reset_alpha_attention,
        )

        # Re-anneal n_step/gamma from scratch for the new cycle -- see _anneal_frac().
        self._anneal_cycle_start_step = self.env_step_count

    def _reset_submodule(self, attr_name: str, reinit_fn, alpha: float) -> None:
        """Shrink-and-perturb `getattr(self.net, attr_name)` in place, then propagate the
        result everywhere else that submodule's weights are mirrored -- see maybe_reset()."""
        submodule = getattr(self.net, attr_name)
        shrink_and_perturb(
            submodule,
            # .to(self.device): reinit_fn builds on CPU by default: mixing its plain params
            # into a CUDA-resident net's parameters in-place would otherwise fail with a
            # device-mismatch error the first time reset actually fires under CUDA training.
            reinit_fn=lambda: reinit_fn().to(self.device),
            alpha=alpha,
        )

        # A partial reinit changes what each parameter's Adam momentum was computed against --
        # keeping stale momentum would actively fight the newly-perturbed weights, defeating
        # half the point of resetting (Ash & Adams 2020 / BBF both reset optimizer state
        # alongside parameters for exactly this reason).
        for p in submodule.parameters():
            self.optimizer.state.pop(p, None)

        # target_net / ema_net would otherwise keep evaluating the OLD (pre-reset)
        # representation for up to target_update_interval steps / at EMA's slow pace, while
        # the online net has already jumped to a partly-perturbed one -- bootstrapping a
        # Q-learning target against such a mismatched representation right after a reset is
        # exactly the kind of thing that destabilizes training, so sync both immediately
        # instead of letting them lag.
        getattr(self.target_net, attr_name).load_state_dict(submodule.state_dict())
        getattr(self.ema_net, attr_name).load_state_dict(submodule.state_dict())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, total_env_steps: int) -> None:
        self._total_env_steps_hint = total_env_steps
        obs, _ = self._reset_env()
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        window_t0 = time.time()
        recent_losses: list[float] = []
        try:
            self._run_loop(obs, total_env_steps, ckpt_dir, t0, window_t0, recent_losses)
        except KeyboardInterrupt:
            # Ctrl+C during a long run would otherwise lose everything since the last periodic
            # checkpoint_interval save -- catch it here (not in main()) since ckpt_dir/env_step_count
            # only live on self, then re-raise so the process still exits with interrupt semantics
            # and main()'s `finally: env.close()` still runs.
            print(f"\n[checkpoint] KeyboardInterrupt at step {self.env_step_count} -- saving before exit")
            self.save(ckpt_dir / f"interrupted_step_{self.env_step_count}.pt")
            raise
        finally:
            self.tb.close()

    def _run_loop(
        self,
        obs: dict[str, dict[str, np.ndarray]],
        total_env_steps: int,
        ckpt_dir: Path,
        t0: float,
        window_t0: float,
        recent_losses: list[float],
    ) -> None:
        while self.env_step_count < total_env_steps:
            obs = self.collect_step(obs)
            self.maybe_reset()

            if self.env_step_count % self.cfg.train_every == 0:
                t_train0 = time.perf_counter()
                loss = self.train_step()
                self._time_train += time.perf_counter() - t_train0
                if loss is not None:
                    recent_losses.append(loss)

            if self.env_step_count % 1000 == 0:
                elapsed = time.time() - t0
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
                window_wall = time.time() - window_t0
                window = self._time_select + self._time_env_step + self._time_train
                other = max(window_wall - window, 0.0)
                print(
                    f"[step {self.env_step_count}] eps={self.epsilon():.3f} n_step={self.current_n_step()} "
                    f"gamma={self.current_gamma():.4f} avg_loss={avg_loss:.4f} "
                    f"({self.env_step_count / max(elapsed, 1e-9):.1f} steps/s) | "
                    f"breakdown over last {window_wall:.1f}s: "
                    f"env.step={self._time_env_step:.1f}s ({100 * self._time_env_step / window_wall:.0f}%) "
                    f"select_actions={self._time_select:.1f}s ({100 * self._time_select / window_wall:.0f}%) "
                    f"train_step={self._time_train:.1f}s ({100 * self._time_train / window_wall:.0f}%) "
                    f"other={other:.1f}s ({100 * other / window_wall:.0f}%)"
                )
                print(
                    f"  train_step internals: prep(sample+to_device)={self._time_prep:.1f}s "
                    f"({100 * self._time_prep / max(self._time_train, 1e-9):.0f}% of train_step) "
                    f"forward(net/target/ema+SPR)={self._time_forward:.1f}s "
                    f"({100 * self._time_forward / max(self._time_train, 1e-9):.0f}%) "
                    f"backward+optim={self._time_backward:.1f}s "
                    f"({100 * self._time_backward / max(self._time_train, 1e-9):.0f}%) "
                    f"priority+ema={self._time_priority:.1f}s "
                    f"({100 * self._time_priority / max(self._time_train, 1e-9):.0f}%)"
                )
                self.tb.scalar("perf/steps_per_sec", self.env_step_count / max(elapsed, 1e-9), self.env_step_count)
                self.tb.scalars(
                    "perf/wall_time_s",
                    {
                        "env_step": self._time_env_step,
                        "select_actions": self._time_select,
                        "train_step": self._time_train,
                        "other": other,
                        "train_prep": self._time_prep,
                        "train_forward": self._time_forward,
                        "train_backward": self._time_backward,
                        "train_priority_ema": self._time_priority,
                    },
                    self.env_step_count,
                )
                self.tb.scalars(
                    "replay_buffer",
                    {
                        "size_a": len(self.buffer_a),
                        "size_b": len(self.buffer_b),
                        "max_priority_a": self.buffer_a._max_priority,
                        "max_priority_b": self.buffer_b._max_priority,
                    },
                    self.env_step_count,
                )

                recent_losses.clear()
                self._time_select = self._time_env_step = self._time_train = 0.0
                self._time_prep = self._time_forward = self._time_backward = self._time_priority = 0.0
                window_t0 = time.time()

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
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError as e:
            # Checkpoints saved before the graphic_encoder param group split (see
            # QMIXConfig.encoder_lr/encoder_weight_decay) have a single AdamW param group;
            # torch.optim.Optimizer.load_state_dict() hard-requires the same number of groups
            # (and same per-group param counts) and raises ValueError otherwise. Net/mixer/spr
            # weights above already loaded fine -- only Adam's per-parameter momentum/variance
            # history is lost here, which is the same cost BBF's own periodic resets already
            # pay for the params they touch (see maybe_reset()), so this is a safe fallback
            # rather than a hard failure.
            print(f"[load] optimizer_state incompatible with current param groups ({e}) -- "
                  f"keeping freshly-initialized optimizer state (net/mixer/spr weights still loaded)")
        self.env_step_count = ckpt["env_step_count"]
        self.train_step_count = ckpt["train_step_count"]


def _auto_device() -> str:
    """cuda > mps > cpu -- picks the fastest backend actually available on this machine
    instead of defaulting to cpu (which silently trained without any GPU acceleration at all
    unless --device was passed explicitly)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, help="Path to the Unity build executable")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total env steps to train for")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale")
    parser.add_argument("--graphics", action="store_true", help="Show the Unity window instead of headless")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint dir; default is a fresh timestamped folder under checkpoints/ (see default_run_dir) so separate runs never overwrite each other's step_*.pt files",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Force heuristic_fill_frac=0 (phase 1 ends almost immediately). Use this when "
        "--resume points at an offline_pretrain.py checkpoint -- the loaded weights already "
        "saw heuristic-vs-heuristic (+ exploration noise) data offline, so re-running phase "
        "1's pure-heuristic collection into this run's now-empty buffer_a/buffer_b would just "
        "delay handing control to the (already pretrained) net for no benefit.",
    )
    parser.add_argument(
        "--seed-dataset-dir",
        default=None,
        help="Dir with buffer_a.npz/buffer_b.npz (from collect_heuristic_dataset.py) to preload "
        "buffer_a/buffer_b with before this run starts collecting online -- otherwise a "
        "--resume'd run starts with empty buffers (only net/optimizer state is checkpointed, "
        "never replay data) and train_step() has nothing but slowly-arriving fresh online "
        "transitions to sample until bootstrap_train_start_frac*capacity of them accumulate. "
        "The preloaded data isn't permanent: SequentialReplayBuffer is a plain FIFO ring, so as "
        "fresh online transitions get pushed they naturally evict the oldest (offline) ones "
        "first, phasing this dataset out on its own once the buffer has cycled through one "
        "capacity's worth of new steps -- same mechanism the phase-1 heuristic bootstrap's "
        "offline->online handoff already relies on (see QMIXConfig.heuristic_fill_frac). "
        "Typically the same dataset --resume's checkpoint was offline-pretrained on, paired "
        "with --skip-bootstrap.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (e.g. cpu/cuda/mps). Default (unset) auto-picks the best available: "
        "cuda > mps > cpu (see _auto_device()).",
    )
    parser.add_argument(
        "--heuristic-opponent-frac",
        type=float,
        default=None,
        help="Fraction of phase-2 episodes whose opponent plays a full HeuristicPolicyMixture "
        "match instead of ema_net (see QMIXConfig.heuristic_opponent_frac). Default (None) "
        "keeps the dataclass default.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile net/target_net/ema_net (cuts kernel-launch overhead; first calls at "
        "each distinct batch/n_quantiles shape pay a one-time recompile)",
    )
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=None,
        help="Env steps between periodic shrink-and-perturb resets of graphic_encoder + the "
        "attention trunk (see QMIXConfig.reset_interval). 0/unset disables resetting entirely "
        "(the default) -- e.g. 200_000 gives 5 resets over a 1,000,000-step run, matching BBF's "
        "resets-per-training-budget ratio.",
    )
    parser.add_argument(
        "--reset-alpha-cnn", type=float, default=None,
        help="Fraction of graphic_encoder's OLD weights kept across a reset (see "
        "QMIXConfig.reset_alpha_cnn; 1.0=no-op, 0.0=full reinit). Default (None) keeps the "
        "dataclass default (0.8, i.e. a gentle 20%% perturbation).",
    )
    parser.add_argument(
        "--reset-alpha-attention", type=float, default=None,
        help="Same as --reset-alpha-cnn but for the attention trunk (see "
        "QMIXConfig.reset_alpha_attention). Default (None) keeps the dataclass default (0.925, "
        "i.e. a ~7.5%% perturbation -- gentler than the CNN's since the trunk carries more of "
        "the network's already-learned behavior).",
    )
    parser.add_argument(
        "--reset-warmup-steps", type=int, default=None,
        help="Linear lr warmup (env steps) for graphic_encoder's own param group only, ramping "
        "0 -> encoder_lr at the start of each reset cycle (see QMIXConfig.reset_warmup_steps) -- "
        "_reset_submodule() wipes this submodule's Adam state on every reset, so the first "
        "post-reset steps have no momentum/variance history. Default (None) keeps the dataclass "
        "default (0, i.e. off).",
    )
    parser.add_argument(
        "--tb-log-dir",
        default=None,
        help="TensorBoard log dir; default is a fresh timestamped folder under runs/ (see default_run_dir), pass '' to disable",
    )
    parser.add_argument(
        "--unity-log-file",
        default=None,
        help="Where Unity's own player log (Debug.Log output -- 'Unit dead', 'Absorption', etc.) "
        "goes, via -logFile. Default is a fresh timestamped path under unity_logs/ (same scheme "
        "as --checkpoint-dir/--tb-log-dir) so it doesn't interleave with this script's own "
        "[step N]/[checkpoint] console output. Pass '' to fall back to Unity's own default "
        "(platform log location, or stdout in some configurations).",
    )
    args = parser.parse_args()

    unity_log_file = args.unity_log_file
    if unity_log_file is None:
        unity_log_file = f"{default_run_dir(base='unity_logs')}.log"
    additional_args = None
    if unity_log_file:
        # Absolute, since Unity resolves a relative -logFile path against its own working
        # directory (e.g. inside the .app bundle), not the cwd this script was launched from.
        unity_log_file = str(Path(unity_log_file).resolve())
        Path(unity_log_file).parent.mkdir(parents=True, exist_ok=True)
        additional_args = ["-logFile", unity_log_file]
        print(f"[unity] player log -> {unity_log_file}")

    device = args.device or _auto_device()
    print(f"[device] using {device}" + ("" if args.device else " (auto-selected: cuda > mps > cpu)"))

    env = BlackOutEnv(
        env_path=args.build,
        # GraphicEncoder assumes a 24x24 input (GRID_H=GRID_W=6 after two stride-2 convs,
        # see my_model.py) — must match the Unity build's RenderTextureSensor resolution.
        map_w=24,
        map_h=24,
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
        additional_args=additional_args,
    )
    config_kwargs = dict(device=device, compile=args.compile)
    if args.heuristic_opponent_frac is not None:
        config_kwargs["heuristic_opponent_frac"] = args.heuristic_opponent_frac
    if args.reset_interval is not None:
        config_kwargs["reset_interval"] = args.reset_interval
    if args.reset_alpha_cnn is not None:
        config_kwargs["reset_alpha_cnn"] = args.reset_alpha_cnn
    if args.reset_alpha_attention is not None:
        config_kwargs["reset_alpha_attention"] = args.reset_alpha_attention
    if args.reset_warmup_steps is not None:
        config_kwargs["reset_warmup_steps"] = args.reset_warmup_steps
    if args.skip_bootstrap:
        config_kwargs["heuristic_fill_frac"] = 0.0
    if args.checkpoint_dir is not None:
        config_kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.tb_log_dir is not None:
        config_kwargs["tb_log_dir"] = args.tb_log_dir or None  # '' -> disable
    config = QMIXConfig(**config_kwargs)
    trainer = QMIXTrainer(env, config)

    if args.seed_dataset_dir is not None:
        seed_dir = Path(args.seed_dataset_dir)
        n_a = load_dataset_into(trainer.buffer_a, seed_dir / "buffer_a.npz")
        n_b = load_dataset_into(trainer.buffer_b, seed_dir / "buffer_b.npz")
        print(
            f"[seed] preloaded {n_a}/{n_b} transitions from {seed_dir} into buffer_a/buffer_b "
            f"(buffer_a now holds {len(trainer.buffer_a)}, buffer_b {len(trainer.buffer_b)} "
            f"of capacity {trainer.buffer_a.capacity})"
        )

    if args.resume:
        trainer.load(Path(args.resume))

    try:
        trainer.run(args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
