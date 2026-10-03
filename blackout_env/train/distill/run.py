"""Asynchronous DAgger distillation of a relabelling teacher into MyModel.

Collection and fitting run at the same time:

* **collector** -- ``--workers`` Unity processes play labelled matches in the background
  (collect.py), the student running on their CPUs.  In a match each learner unit follows the
  teacher with probability beta and the student otherwise; the teacher labels every state either
  way.  Each finished match lands as a shard on disk;
* **learner** -- this process fits the net (set-valued cross-entropy, loss.py) on the aggregate
  of every shard so far (the most recent ``--max-rows`` rows), on ``--device``.

They meet at two points:

* every ``--sync-every`` steps the learner collects finished matches, scores the current net on
  their rows *before* they enter the replay ring (``fresh_*``: states it has never trained on),
  loads them, and queues more matches -- only while it has trained at least ``--replay-ratio``
  samples per collected row, so the (much faster) collector cannot flood the ring with rows the
  learner never sees;
* every ``--publish-every`` steps it atomically replaces ``latest.pt``; a worker reloads the
  student when it starts its next match.  Each publish is a new student *version*; beta decays
  per version, ``beta_start * beta_decay ** (version - 1)`` down to ``beta_min`` (version 0 = no
  student yet = teacher only, which also fills the ring up to ``--min-rows`` before fitting).
  Every ``--eval-every`` versions ``2 x --eval-seeds`` student-only matches against
  ``--eval-opponent`` on fixed seeds are queued ahead of collection; their states are labelled
  too.  A new run first plays the teacher on the same seeds (``eval_teacher``), the target.

Output (``--out-dir``):

  metrics.jsonl    every record (kinds: train, collect, eval, eval_teacher, probe), also in tb/
  matches.jsonl    one summary per match, with raw counts (train/distill/metrics.py)
  shards/          one per match: states, labels, executed and student moves (dataset.py)
  run_info.json    git commit and dirty files, build, versions, arguments (one entry per start)
  probe.npz        fixed teacher-state rows, scored at every publish
  latest.pt        policy_state (MyModel weights; load_my_policy_checkpoint and
                   examples/gui_match.py play it) + optimizer and counters to resume from;
                   v_NNNN.pt keeps weights every --snapshot-every versions

Re-running with the same ``--out-dir`` resumes.  ``examples/distill_report.py`` prints the tables.

Example (smoke test)::

    python -m blackout_env.train.distill.run --build build/mac/BlackOut.app --out-dir checkpoints/distill/smoke \\
        --steps 120 --workers 2 --min-rows 2000 --publish-every 40 --eval-every 2 --eval-seeds 1 --batch-size 64
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from blackout_env.model.direction_bins import ACTION_BINS, SET_HALF_WIDTH_DEG, bins_of_width, resample_rows
from blackout_env.model.my_model import ANGLE_KEY_PREFIX, POLICY_HEADS, TEMPORAL_PREFIX, MyModel
from blackout_env.train.accel import adamw, check_amp, pick_device
from blackout_env.train.distill.collect import CollectorPool, MatchTask, teacher_arg
from blackout_env.train.distill.inference import INFERENCE
from blackout_env.train.distill.dataset import (
    ANGLES, FIELDS, GRAPHIC_SCALE, KIND_INDEX, N_TEAM, DistillDataset, angles_of, new_shards,
)
from blackout_env.train.distill.loss import angle_offset_loss, set_valued_cross_entropy
from blackout_env.train.distill.metrics import (
    CLASS_COLS, MATCH_SECONDS, OPENING_SECONDS, LabelAccuracy, MetricsLog, summarize_matches,
)
from blackout_env.train.distill.opponents import (
    check_opponent, compatible_checkpoints, draw_opponent, opponent_category, parse_pool,
)

HELDOUT_SEEDS = range(900_000, 1_000_000)
# --compile: the heavy submodules only.  Inductor on MPS (torch 2.14) multiplies the gradient of
# an embedding broadcast over the batch (token_type_emb, vec_slot_emb, spr_cls_emb) by the batch
# size, so the embeddings stay eager; compiled this way every gradient matches CPU eager.
COMPILED_MODULES = ("graphic_encoder", "vector_encoder", "attention", "q_head")  # examples/distill_heldout_eval.py
# On CUDA that bug does not occur (tests/test_accel.py), but the whole MyModel compiled under fp16
# autocast gives wrong outputs there (value/train.compile_value_net), so CUDA compiles these too.


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--out-dir", type=Path, default=None, help="default: checkpoints/distill/<timestamp>")
    p.add_argument("--teacher", type=teacher_arg, default="v17-teach", help="a TEACHERS name or act:<opponent descriptor>")
    p.add_argument("--init", type=Path, default=None, help="start from these MyModel weights (any checkpoint with policy_state)")
    p.add_argument("--steps", type=int, default=100_000, help="total gradient steps")
    # collector
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--opponents", nargs="+", default=["mixture=0.75", "checkpoints=0.25"],
                   help="kind=weight pool drawn per collection match: mixture (all heuristics, V17-V19 perturbed "
                   "too), checkpoints (--opponent-checkpoints), or a heuristic id (opponents.py)")
    p.add_argument("--opponent-checkpoints", nargs="+", default=["checkpoints/offline/*/*.pt"],
                   help="globs of past MyModel checkpoints; those that no longer load are skipped")
    p.add_argument("--beta-start", type=float, default=1.0, help="beta of the first student version")
    p.add_argument("--beta-decay", type=float, default=0.9, help="per published version")
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--opponent-extra", nargs="+", default=[],
                   help="descriptors the 'extra' pool kind draws uniformly from (opponents.py: perturb:..., "
                   "heuristic:key=value, teacher:..., ...)")
    p.add_argument("--eval-opponent", default="strategic_v17")
    p.add_argument("--eval-seeds", type=int, default=3, help="student-only eval matches = 2 x this")
    p.add_argument("--baseline-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="a new run first plays the teacher on the eval seeds (eval_teacher)")
    p.add_argument("--seed", type=int, default=20260929)
    # synchronisation
    p.add_argument("--min-rows", type=int, default=20_000, help="teacher-only rows to collect before fitting")
    p.add_argument("--replay-ratio", type=float, default=2.0,
                   help="trained samples per collected row; below it no new matches are queued")
    p.add_argument("--queue-depth", type=int, default=2,
                   help="matches queued per worker: most finish in seconds, so a worker must have its next one "
                   "waiting instead of idling until the next sync")
    p.add_argument("--sync-every", type=int, default=50, help="steps between replay syncs")
    p.add_argument("--publish-every", type=int, default=1000, help="steps between student versions")
    p.add_argument("--eval-every", type=int, default=5, help="versions between eval batches")
    p.add_argument("--snapshot-every", type=int, default=5, help="versions between kept v_NNNN.pt weights")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--fresh-rows", type=int, default=256, help="rows per new shard scored before it enters the ring")
    p.add_argument("--probe-rows", type=int, default=2048, help="fixed teacher-state rows scored at every publish")
    # learner
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4, help="peak learning rate (see --lr-schedule)")
    p.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                   help="lr over the run, a function of the global step and --steps (lr_at): constant = --lr throughout "
                   "(the default, unchanged); cosine / linear decay from --lr after the warm-up to --lr x --lr-final "
                   "at the last step.  Resume-safe: a resumed run continues the same curve; resuming with a larger "
                   "--steps recomputes it from the current step (the decay stretches, so the lr can jump up)")
    p.add_argument("--lr-warmup", type=int, default=0, help="steps of linear warm-up from 0 to --lr (any schedule)")
    p.add_argument("--lr-final", type=float, default=0.1,
                   help="cosine/linear: the last step's lr as a fraction of --lr")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--z-loss", type=float, default=1e-4)
    p.add_argument("--class-weighting", choices=["none", "balanced"], default="none",
                   help="balanced: each unit class (Collector/Hunter/Carrier) gets equal total weight per batch, "
                   "so V17's 3-Hunter composition does not crowd out the one Collector's labels")
    p.add_argument("--train-quantiles", type=int, default=32,
                   help="IQN quantiles sampled per training forward (the loss reads their mean); the Q head is "
                   "half the forward at 32 (examples/distill_step_bench.py).  Ignored with --policy-head logits")
    p.add_argument("--policy-head", choices=POLICY_HEADS, default="iqn",
                   help="logits (P5): plain per-unit logits instead of the IQN quantile head, the same loss.  --init "
                   "from an IQN checkpoint folds its tau embedding at inference's fixed taus into the head "
                   "(MyModel.iqn_to_logits_state; train/distill/head_fit.py fits it closer)")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                   help="autocast precision of the training forward (weights and optimizer stay fp32)")
    p.add_argument("--compile", action="store_true", help="torch.compile COMPILED_MODULES (~11 s warm-up)")
    p.add_argument("--tf32", action="store_true",
                   help="CUDA Ampere+: fp32 matmuls/convs on TF32 tensor cores (outside autocast; P14 on the RTX 3060)")
    p.add_argument("--stand-action", action="store_true",
                   help="a 9th output, STAND, trained where the teacher has no label: it stands there (interacting, "
                   "waiting, holding a slot), and an 8-way student dithers back and forth instead -- 46%% of D4's "
                   "reversals.  --init from an 8-way checkpoint keeps its directions and adds the new row")
    p.add_argument("--action-bins", type=int, choices=ACTION_BINS, default=8,
                   help="K heading bins per unit (P4; bin k at 2 pi k / K, bins 0, K/8, ... = the compass): the head emits "
                   "K logits (+ STAND), labels are K-bit sets.  --init from a checkpoint with another K interpolates its "
                   "direction rows (the same moves at step 0).  8 = the compass, unchanged")
    p.add_argument("--label-width-deg", type=float, default=SET_HALF_WIDTH_DEG,
                   help="--action-bins: a set teacher's allowed direction labels the bins within this many degrees of it")
    p.add_argument("--act-arc-bins", type=int, default=0,
                   help="--action-bins: an act: teacher's raw heading labels its nearest bin and this many either side")
    p.add_argument("--angle-head", action="store_true",
                   help="P11 (--policy-head logits): beside the K bin logits, a per-bin offset within +-half a bin "
                   "(zero-initialised: an --init without one moves exactly as before), trained on act: teachers' raw "
                   "headings (shard field angles); inference executes the arg max bin's centre + its offset")
    p.add_argument("--angle-loss-weight", type=float, default=1.0,
                   help="--angle-head: weight of the offset loss (squared error in half-bin units, loss.angle_offset_loss: "
                   "1/3 untrained, 0.13 at 2 deg for K = 32 -- the cross-entropy's order)")
    p.add_argument("--angle-arc-bins", type=int, default=0,
                   help="--angle-head: each offset spans +-(N + 1/2) bins and the N bins either side of the teacher's are "
                   "trained to point at its angle too -- with --act-arc-bins N the set-valued CE makes any bin of the arc "
                   "the arg max (P10: a neighbour 34%% of the time), which a half-bin offset cannot correct.  0 = the "
                   "teacher's bin only, +-half a bin")
    p.add_argument("--opening-weight", type=float, default=1.0,
                   help="multiplies the per-row loss (cross-entropy and angle) of rows in the first --opening-seconds of "
                   "play (the race to the sanctuary); 1 = off")
    p.add_argument("--opening-seconds", type=float, default=OPENING_SECONDS,
                   help="--opening-weight's window: seconds of play, from team_state's time left")
    p.add_argument("--inference", choices=INFERENCE, default="iqn",
                   help="how collection and eval students turn logits into a move (inference.py; arc_mean: circular "
                   "mean of the top contiguous arc, for K bins)")
    p.add_argument("--temporal", action="store_true",
                   help="a student with memory (MyModel(temporal=True): per-slot KDA between trunk and head), trained on "
                   "windows of consecutive ticks; --init from a stateless checkpoint warm-starts it with identical outputs")
    p.add_argument("--windows", type=int, default=8, help="--temporal: windows per batch")
    p.add_argument("--window", type=int, default=32, help="--temporal: trained ticks per window")
    p.add_argument("--burn-in", type=int, default=32, help="--temporal: ticks before each window that only build memory")
    p.add_argument("--max-rows", type=int, default=600_000, help="replay ring size (~8 KB per row in RAM)")
    p.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    args = p.parse_args(argv)
    if args.angle_head and args.policy_head != "logits":
        p.error("--angle-head needs --policy-head logits")
    return args


LR_SCHEDULES = ("constant", "cosine", "linear")


def lr_at(step: int, args) -> float:
    """The lr of the gradient step taken after ``step`` completed steps (0 = the first).

    Warm-up (``--lr-warmup`` W > 0): ``lr * (step + 1) / W`` for step < W, so the first step is
    not wasted at 0 and step W - 1 reaches ``lr``.  Then constant, or cosine / linear from ``lr``
    at step W to ``lr * --lr-final`` at the last step (``--steps - 1``), held there after it.
    Stateless -- only the step and the args -- so nothing is saved and a resume continues it."""
    lr, warmup = args.lr, args.lr_warmup
    if step < warmup:
        return lr * (step + 1) / warmup
    if args.lr_schedule == "constant":
        return lr
    progress = min(1.0, max(0.0, (step - warmup) / max(1, args.steps - 1 - warmup)))
    if args.lr_schedule == "cosine":
        shape = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif args.lr_schedule == "linear":
        shape = 1.0 - progress
    else:
        raise ValueError(f"unknown --lr-schedule {args.lr_schedule!r}")
    return lr * (args.lr_final + (1.0 - args.lr_final) * shape)


def lr_is_scheduled(args) -> bool:
    """False for the default (constant, no warm-up): the optimizer's lr is then never touched,
    bit-exact with the trainer before schedules (incl. a resume keeping the saved optimizer's lr)."""
    return args.lr_schedule != "constant" or args.lr_warmup > 0


def beta_for(version: int, args) -> float:
    if version == 0:
        return 1.0
    return max(args.beta_min, args.beta_start * args.beta_decay ** (version - 1))


def collect_seed(run_seed: int, index: int, reserved: set[int]) -> int:
    """A random-looking map seed for collection match ``index``: reproducible (and so resumable)
    from the run seed, never an eval or held-out seed."""
    for attempt in range(100):
        seed = int(np.random.SeedSequence((run_seed, index, attempt)).generate_state(1)[0] % 2**31)
        if seed not in reserved and seed not in HELDOUT_SEEDS:
            return seed
    raise RuntimeError("no free seed")


def wants_matches(trained_samples: int, collected_rows: int, dataset_rows: int, args) -> bool:
    """Queue more collection?  Always while filling up to --min-rows; afterwards only while the
    learner has used each collected row at least --replay-ratio times."""
    return dataset_rows < args.min_rows or trained_samples >= args.replay_ratio * collected_rows


def class_weights(classes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, 5, 3] one-hot classes -> [B, 5] weights giving each present class equal total weight."""
    counts = (classes * valid.unsqueeze(-1)).sum(dim=(0, 1))  # [3]
    present = (counts > 0).sum().clamp_min(1)
    per_class = torch.where(counts > 0, valid.sum() / (present * counts.clamp_min(1)), torch.zeros_like(counts))
    return (classes * per_class).sum(dim=-1)


STAND_INIT_MARGIN = 4.0  # logits


def load_weights(net: MyModel, state: dict) -> None:
    """Load ``state`` into ``net``, adapting the head's output rows when the two differ:

    * K bins: the net's bin rows are the state's direction rows interpolated in angle
      (``direction_bins.resample_rows``; bins on a source direction are exact copies), so a
      K-bin net from an 8-way checkpoint picks the same headings as it did -- the same logits on
      the compass bins, lower ones between them;
    * policy head: an IQN state into a logits net is folded first (``MyModel.iqn_to_logits_state``,
      the mean tau embedding at inference's fixed taus); a logits state into an IQN net raises;
    * STAND: a state without one into a --stand-action net starts STAND's row at zero weights
      and a bias STAND_INIT_MARGIN below the lowest direction's, so the inherited student keeps
      moving until STAND is trained (at the lowest bias alone, STAND won half of the labelled
      ticks of D4's weights); a state with one keeps it;
    * angle head (P11): a state without one into an --angle-head net keeps the net's zero init
      (the same moves at step 0); a state with one into a net without drops it (with a note)."""
    if net.policy_head == "logits":
        state = MyModel.iqn_to_logits_state(state)
    elif "q_head.in_scale" in state:
        raise ValueError("a logits-head state cannot initialise an IQN-head net (--policy-head iqn)")
    angle_keys = [k for k in state if k.startswith(ANGLE_KEY_PREFIX)]
    if angle_keys and not net.angle_head:
        print("[distill] --init has an angle head; this net has none: its offsets are dropped", flush=True)
        state = {k: v for k, v in state.items() if k not in angle_keys}
        angle_keys = []
    key = "q_head.value_head.1"
    dst = bins_of_width(net.state_dict()[f"{key}.weight"].shape[0])
    src = bins_of_width(state[f"{key}.weight"].shape[0])
    if src != dst:
        (src_bins, src_stand), (dst_bins, dst_stand) = src, dst
        state = dict(state)
        weight, bias = state[f"{key}.weight"], state[f"{key}.bias"]
        new_weight = resample_rows(weight[:src_bins], src_bins, dst_bins)
        new_bias = resample_rows(bias[:src_bins], src_bins, dst_bins)
        if dst_stand:
            if src_stand:
                stand_weight, stand_bias = weight[src_bins:], bias[src_bins:]
            else:
                stand_weight, stand_bias = torch.zeros_like(weight[:1]), (bias[:src_bins].min() - STAND_INIT_MARGIN).reshape(1)
            new_weight, new_bias = torch.cat((new_weight, stand_weight)), torch.cat((new_bias, stand_bias))
        state[f"{key}.weight"], state[f"{key}.bias"] = new_weight, new_bias
        if angle_keys:  # offsets per bin of another K: start them over
            state = {k: v for k, v in state.items() if k not in angle_keys}
    if "q_head.angle_scale" in state and net.angle_head:  # this run's --angle-arc-bins sets the range
        state = {**state, "q_head.angle_scale": net.state_dict()["q_head.angle_scale"]}
    state = net.with_init_angle_keys(state)
    if net.is_temporal and not any(k.startswith(TEMPORAL_PREFIX) for k in state):
        net.load_warm_start(state)  # a stateless checkpoint: the memory block starts as the identity
    else:
        net.load_state_dict(state)


class Learner:
    """The net, its optimizer, and running training statistics (summed on-device: no per-step
    host sync)."""

    def __init__(self, args):
        self.args = args
        self.device = pick_device(args.device)
        check_amp(self.device, args.amp)
        if getattr(args, "tf32", False) and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.net = MyModel(n_actions=args.action_bins + int(args.stand_action), temporal=args.temporal,
                           policy_head=args.policy_head, angle_head=args.angle_head,
                           angle_arc_bins=args.angle_arc_bins).to(self.device)
        self.angle = bool(args.angle_head)
        if args.compile:
            for name in COMPILED_MODULES:  # in place: state_dict keys stay the same
                getattr(self.net, name).compile()
        self.optimizer = adamw(self.net.parameters(), self.device, lr=args.lr, weight_decay=args.weight_decay)
        self._bit_index = torch.arange(args.action_bins, device=self.device)
        self.accuracy = LabelAccuracy(self.device)
        self._totals = defaultdict(lambda: torch.zeros((), device=self.device))
        self._steps = 0
        self._amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
        # fp16's narrow exponent underflows small gradients: scale the loss (a no-op otherwise)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=args.amp == "fp16")
        self.mark_weights()

    def _tensors(self, rows: dict[str, np.ndarray]):
        # K > 8 labels are uint64 bit sets: the same bits as int64 (MPS has no uint64 shifts;
        # an arithmetic shift still reads bit 63 right after the & 1)
        # (no pinning: the step needs the batch at once, so an async copy overlaps nothing and the
        # extra pinned memcpy only costs host time)
        batch = {k: torch.from_numpy(rows[k].view(np.int64) if rows[k].dtype == np.uint64 else rows[k]).to(self.device)
                 for k in FIELDS}
        graphic = batch["graphic"].permute(0, 3, 1, 2).float() / GRAPHIC_SCALE
        masks = ((batch["labels"].long().unsqueeze(-1) >> self._bit_index) & 1) == 1  # dataset.unpack_labels
        classes = batch["agent_states"][:, :N_TEAM, CLASS_COLS]
        kinds = batch["kinds"]
        if self.args.stand_action:  # no label = the teacher stands: target STAND alone
            stand = batch["labels"] == 0
            masks = torch.cat((masks, stand.unsqueeze(-1)), dim=-1)
            kinds = torch.where(stand, torch.full_like(kinds, KIND_INDEX["stand"]), kinds)
        # P11: the teacher's raw headings (NaN = none) and seconds of play (team_state[2] = time left)
        aux = {"angles": torch.from_numpy(np.ascontiguousarray(angles_of(rows))).to(self.device),
               "elapsed": (1.0 - batch["team_state"][:, 2].float()) * MATCH_SECONDS}
        return graphic, batch["team_state"], batch["agent_states"], masks, classes, kinds, aux

    def set_lr(self, step: int) -> None:
        """Apply the --lr-schedule value for global ``step`` to every param group."""
        if lr_is_scheduled(self.args):
            lr = lr_at(step, self.args)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    @property
    def rows_per_step(self) -> int:
        """Trained rows per gradient step (what --replay-ratio counts)."""
        return self.args.windows * self.args.window if self.args.temporal else self.args.batch_size

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self._amp) if self._amp else contextlib.nullcontext()

    def step(self, rows: dict[str, np.ndarray]) -> None:
        """One gradient step on independent rows (a stateless student)."""
        self.net.train()
        graphic, team_state, agent_states, masks, classes, kinds, aux = self._tensors(rows)
        with self._autocast():  # the loss and statistics below stay fp32
            out = self.net(graphic, team_state, agent_states, n_quantiles=self.args.train_quantiles, q_units=N_TEAM,
                           **self._angle_kw)
        offsets = out[5].float() if self.angle else None
        self._fit(out[0].float(), masks, classes, kinds, aux, offsets)

    @property
    def _angle_kw(self) -> dict:
        return {"angle": True} if self.angle else {}

    def _window_tensors(self, windows: dict[str, np.ndarray]):
        """dataset.sample_windows output -> [B, T, ...] tensors; label masks are cleared where
        ``loss_mask`` is off (padding, burn-in), so those positions carry no target."""
        b, t = windows["valid"].shape
        flat = {k: windows[k].reshape(b * t, *windows[k].shape[2:]) for k in FIELDS + (ANGLES,) if k in windows}
        graphic, team_state, agent_states, masks, classes, kinds, aux = self._tensors(flat)
        seq = lambda x: x.reshape(b, t, *x.shape[1:])
        loss_mask = torch.from_numpy(windows["loss_mask"]).to(self.device)
        masks = seq(masks) & loss_mask[:, :, None, None]
        angles = torch.where(loss_mask[:, :, None], seq(aux["angles"]), torch.full_like(seq(aux["angles"]), float("nan")))
        aux = {"angles": angles, "elapsed": seq(aux["elapsed"])}
        reset = torch.from_numpy(windows["reset"]).to(self.device)
        return seq(graphic), seq(team_state), seq(agent_states), masks, seq(classes), seq(kinds), reset, aux

    def step_windows(self, windows: dict[str, np.ndarray]) -> None:
        """One gradient step on windows of consecutive ticks (a --temporal student): memory is
        built over the burn-in without gradients, then trained through the window."""
        self.net.train()
        graphic, team_state, agent_states, masks, classes, kinds, reset, aux = self._window_tensors(windows)
        burn = self.args.burn_in
        with self._autocast():
            out = self.net.forward_sequence(graphic, team_state, agent_states, reset=reset, burn_in=burn,
                                            n_quantiles=self.args.train_quantiles, q_units=N_TEAM, **self._angle_kw)
        q_values = out[0]
        flat = lambda x: x[:, burn:].reshape(-1, *x.shape[2:])
        offsets = out[3].float().reshape(-1, *out[3].shape[2:]) if self.angle else None
        self._fit(q_values.float().reshape(-1, *q_values.shape[2:]), flat(masks), flat(classes), flat(kinds),
                  {k: flat(v) for k, v in aux.items()}, offsets)

    def row_weights(self, valid: torch.Tensor, classes: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor | None:
        """[B, 5] per-unit loss weights: --class-weighting times --opening-weight on the opening rows
        (None = uniform)."""
        weights = class_weights(classes, valid) if self.args.class_weighting == "balanced" else None
        if self.args.opening_weight != 1.0:
            opening = torch.where(elapsed < self.args.opening_seconds, self.args.opening_weight, 1.0)[:, None]
            weights = opening.expand_as(valid) if weights is None else weights * opening
        return weights

    def _fit(self, logits: torch.Tensor, masks: torch.Tensor, classes: torch.Tensor, kinds: torch.Tensor,
             aux: dict | None = None, offsets: torch.Tensor | None = None) -> None:
        valid = masks.any(dim=-1).float()
        elapsed = aux["elapsed"] if aux is not None else torch.full(valid.shape[:1], MATCH_SECONDS, device=valid.device)
        weights = self.row_weights(valid, classes, elapsed)
        loss, stats = set_valued_cross_entropy(logits, masks, weights, self.args.z_loss)
        if offsets is not None and self.args.angle_loss_weight > 0:
            angle_loss, angle_stats = angle_offset_loss(offsets, aux["angles"], weights, self.args.angle_arc_bins)
            loss = loss + self.args.angle_loss_weight * angle_loss
            stats.update(angle_stats)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.scaler.step(self.optimizer)  # skipped when fp16 gradients overflowed
        self.scaler.update()
        finite = torch.isfinite(grad_norm)
        stats["grad_norm"] = torch.where(finite, grad_norm, torch.zeros_like(grad_norm))
        if self.scaler.is_enabled():
            stats["amp_skipped"] = (~finite).float()

        with torch.no_grad():
            for key, value in stats.items():
                self._totals[key] += value.detach()
            # Logit health: is the head drifting in scale, or collapsing to certainty?
            n = valid.sum().clamp_min(1)
            log_p = torch.log_softmax(logits, dim=-1)
            self._totals["entropy"] += (-(log_p.exp() * log_p).sum(-1) * valid).sum() / n
            self._totals["set_prob"] += ((log_p.exp() * masks).sum(-1) * valid).sum() / n
            self._totals["logit_std"] += logits.std(dim=-1).mean()
            self._totals["logit_abs_max"] += logits.abs().amax()
            self._totals["valid_labels_per_row"] += valid.sum(-1).mean()
            angles = aux["angles"] if aux is not None else None
            self.accuracy.add(logits, masks, classes, kinds, angles, offsets, elapsed)
        self._steps += 1

    @torch.no_grad()
    def score(self, rows: dict[str, np.ndarray], accuracy: LabelAccuracy, chunk: int = 512) -> None:
        """Label accuracy of the current net on ``rows`` (no training)."""
        self.net.eval()
        for start in range(0, len(rows["labels"]), chunk):
            part = {k: rows[k][start:start + chunk] for k in FIELDS + (ANGLES,) if k in rows}
            graphic, team_state, agent_states, masks, classes, kinds, aux = self._tensors(part)
            out = self.net(graphic, team_state, agent_states, q_units=N_TEAM, **self._angle_kw)  # temporal: from empty memory
            accuracy.add(out[0], masks, classes, kinds, aux["angles"], out[5] if self.angle else None, aux["elapsed"])

    @torch.no_grad()
    def score_sequence(self, rows: dict[str, np.ndarray], accuracy: LabelAccuracy, burn_in: int) -> None:
        """Label accuracy on consecutive ticks of one match (a --temporal student), the first
        ``burn_in`` only building memory."""
        self.net.eval()
        n = len(rows["labels"])
        windows = {k: rows[k][None] for k in FIELDS + (ANGLES,) if k in rows}
        windows["valid"] = np.ones((1, n), bool)
        windows["reset"] = np.zeros((1, n), bool)
        windows["loss_mask"] = windows["valid"].copy()
        windows["loss_mask"][:, :burn_in] = False
        graphic, team_state, agent_states, masks, classes, kinds, reset, aux = self._window_tensors(windows)
        out = self.net.forward_sequence(graphic, team_state, agent_states, reset=reset, burn_in=burn_in, q_units=N_TEAM,
                                        **self._angle_kw)
        accuracy.add(out[0][0], masks[0, burn_in:], classes[0, burn_in:], kinds[0, burn_in:], aux["angles"][0, burn_in:],
                     out[3][0] if self.angle else None, aux["elapsed"][0, burn_in:])

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.net.parameters()]).clone()

    def mark_weights(self) -> None:
        """Measure ``param_update_norm`` from the weights as they are now (e.g. after loading)."""
        self._params_at_pop = self._flat_params()

    def pop_stats(self) -> dict[str, float]:
        """Means since the last call, plus weight norm and how far the weights moved."""
        out = {f"train/{k}": float(v) / max(1, self._steps) for k, v in self._totals.items()}
        out.update(self.accuracy.pop("train"))
        params = self._flat_params()
        out["train/param_norm"] = float(params.norm())
        out["train/param_update_norm"] = float((params - self._params_at_pop).norm())
        out["train/lr"] = self.optimizer.param_groups[0]["lr"]
        if self.scaler.is_enabled():
            out["train/loss_scale"] = self.scaler.get_scale()  # one host sync per log, not per step
        self._params_at_pop = params
        self._totals.clear()
        self._steps = 0
        return out


def _run_info(args) -> dict:
    def git(*cmd: str) -> str:
        try:
            return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return ""

    build = Path(args.build)
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()],
        "build": str(build.resolve()),
        "build_mtime": datetime.fromtimestamp(build.stat().st_mtime).isoformat(timespec="seconds") if build.exists() else None,
        "python": platform.python_version(), "torch": torch.__version__, "host": platform.node(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }


class Run:
    """The learner's side of the loop: counters, checkpoints, queueing and logging."""

    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir or Path("checkpoints/distill") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.shards = self.out_dir / "shards"  # collection only: the replay ring reads these
        self.eval_shards = self.out_dir / "eval_shards"  # scored as held-out, never trained on
        self.shards.mkdir(parents=True, exist_ok=True)
        self.eval_shards.mkdir(exist_ok=True)
        self._eval_seen: set[str] = set()
        self.eval_seeds = [args.seed - 1 - i for i in range(args.eval_seeds)]
        self.opponent_pool = parse_pool(args.opponents)
        self.opponent_checkpoints = self._resolve_checkpoints()
        for descriptor in args.opponent_extra:
            check_opponent(descriptor)
        self.latest = self.out_dir / "latest.pt"
        self.learner = Learner(args)
        self.step = self.version = self.queued = self.collected_rows = 0
        info_path = self.out_dir / "run_info.json"
        history = json.loads(info_path.read_text()) if info_path.exists() else []
        info_path.write_text(json.dumps(history + [_run_info(args)], indent=1))
        self.dataset = DistillDataset(self.shards, args.max_rows, action_bins=args.action_bins)
        self.dataset.refresh()
        self.metrics = MetricsLog(self.out_dir)
        self.match_log = open(self.out_dir / "matches.jsonl", "a")
        self.fresh = {"collect": LabelAccuracy(self.learner.device), "eval": LabelAccuracy(self.learner.device)}
        self.probe: dict[str, np.ndarray] | None = None
        self._collect_window: list[dict] = []
        self._evals: dict[tuple[str, int], list[dict]] = defaultdict(list)
        self.resumed = self._restore()  # last: an --init start publishes (and scores the probe) right away
        self.learner.mark_weights()
        self._pipeline = defaultdict(float)  # sync-level counters since the last log
        self._window_started = time.time()
        self._matches_received = 0
        self._rng = np.random.default_rng(args.seed + self.step)

    def _resolve_checkpoints(self) -> list[str]:
        """Checkpoint opponents, fixed for the run (saved, so a resume draws from the same list)."""
        path = self.out_dir / "opponent_checkpoints.json"
        if path.exists():
            return json.loads(path.read_text())["loadable"]
        if "checkpoints" not in self.opponent_pool:
            return []
        loadable, rejected = compatible_checkpoints(self.args.opponent_checkpoints)
        path.write_text(json.dumps({"loadable": loadable, "rejected": rejected}, indent=1))
        print(f"[distill] checkpoint opponents: {len(loadable)} loadable, {len(rejected)} skipped", flush=True)
        return loadable

    # ----------------------------------------------------------------------- checkpoints

    def _restore(self) -> bool:
        if self.latest.exists():
            ckpt = torch.load(self.latest, map_location=self.learner.device, weights_only=True)
            self.learner.net.load_state_dict(ckpt["policy_state"])
            self.learner.optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scaler_state" in ckpt:
                self.learner.scaler.load_state_dict(ckpt["scaler_state"])
            state = ckpt["distill"]
            self.step, self.version = ckpt["train_step_count"], state["version"]
            self.queued, self.collected_rows = state["queued"], state["collected_rows"]
            print(f"[distill] resuming at step {self.step}, student v{self.version}")
            return True
        if self.args.init is not None:
            init = torch.load(self.args.init, map_location=self.learner.device, weights_only=True)
            load_weights(self.learner.net, init["policy_state"])
            self.publish()
        return False

    def publish(self) -> None:
        """A new student version: replace latest.pt atomically (workers may be reading it)."""
        self.version += 1
        payload = {
            "policy_state": self.learner.net.state_dict(),
            "optimizer_state": self.learner.optimizer.state_dict(),
            "scaler_state": self.learner.scaler.state_dict(),
            "train_step_count": self.step,
            "distill": {"version": self.version, "teacher": self.args.teacher, "queued": self.queued,
                        "collected_rows": self.collected_rows, "temporal": self.args.temporal,
                        "action_bins": self.args.action_bins, "stand_action": self.args.stand_action,
                        "policy_head": self.args.policy_head, "lr": self.args.lr,
                        "angle_head": self.args.angle_head, "angle_loss_weight": self.args.angle_loss_weight,
                        "angle_arc_bins": self.args.angle_arc_bins,
                        "opening_weight": self.args.opening_weight, "opening_seconds": self.args.opening_seconds,
                        "lr_schedule": self.args.lr_schedule, "lr_warmup": self.args.lr_warmup,
                        "lr_final": self.args.lr_final, "steps": self.args.steps},
        }
        tmp = self.latest.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.latest)
        if self.version % self.args.snapshot_every == 0:
            torch.save({"policy_state": payload["policy_state"], "distill": payload["distill"]},
                       self.out_dir / f"v_{self.version:04d}.pt")
        if self.probe is not None:
            accuracy = LabelAccuracy(self.learner.device)
            self.learner.score(self.probe, accuracy)
            self.metrics.write("probe", self.step, {"version": self.version, **accuracy.pop("probe")})

    def _load_probe(self) -> None:
        """Fixed rows drawn once from the teacher-only warm-up: one curve comparable across the run."""
        path = self.out_dir / "probe.npz"
        if not path.exists():
            np.savez_compressed(path, **self.dataset.sample(self.args.probe_rows, np.random.default_rng(self.args.seed)))
        with np.load(path) as probe:
            self.probe = {field: probe[field] for field in FIELDS}
            self.probe[ANGLES] = angles_of(probe)  # all NaN for a probe from before P11

    # ------------------------------------------------------------------------- collection

    def _task(self, tag: str, seed: int, swap: bool, opponent: str, beta: float) -> MatchTask:
        index = self.queued
        self.queued += 1
        folder = self.shards if tag == "collect" else self.eval_shards
        return MatchTask(seed=seed, swap=swap, opponent=opponent, beta=beta, teacher=self.args.teacher,
                         student=str(self.latest) if beta < 1.0 else None,
                         shard=str(folder / f"{index:06d}_{tag}_v{self.version:04d}.npz"),
                         tag=tag, version=self.version, inference=self.args.inference,
                         action_bins=self.args.action_bins, label_width_deg=self.args.label_width_deg,
                         act_arc_bins=self.args.act_arc_bins)

    def queue_eval(self, pool: CollectorPool, tag: str = "eval") -> None:
        # the same seeds every time, so versions compare on equal maps
        beta = 1.0 if tag == "eval_teacher" else 0.0
        for i in range(2 * self.args.eval_seeds):
            pool.submit(self._task(tag, self.eval_seeds[i // 2], bool(i % 2), self.args.eval_opponent, beta))

    def top_up(self, pool: CollectorPool) -> None:
        """Fill idle workers while the learner wants data, counting the rows the matches already
        running will bring (at the mean rows per match so far) as collected."""
        args = self.args
        self._pipeline["syncs"] += 1
        self._pipeline["queued_per_worker"] += pool.in_flight / pool.workers
        throttled = False
        while pool.in_flight < args.queue_depth * pool.workers:
            expected = self.collected_rows + pool.in_flight * self._rows_per_match()
            if not wants_matches(self.step * self.learner.rows_per_step, expected, self.dataset.size, args):
                throttled = True
                break
            n = self.queued
            rng = np.random.default_rng((args.seed, n))
            opponent = draw_opponent(rng, self.opponent_pool, self.opponent_checkpoints, args.opponent_extra)
            pool.submit(self._task("collect", collect_seed(args.seed, n, set(self.eval_seeds)), bool(rng.integers(2)),
                                   opponent, beta_for(self.version, args)))
        self._pipeline["throttled_syncs"] += throttled

    def _rows_per_match(self) -> float:
        return self.collected_rows / self._matches_received if self._matches_received else 2000.0

    def sync(self, pool: CollectorPool, timeout: float = 0.0) -> None:
        self._record(pool.poll(timeout))
        for rows, meta in self.dataset.new_shards():  # fresh = the current net has not trained on these yet
            self._score_fresh(rows, "collect")
            self.dataset.append(rows, meta.get("student_version", 0))
        for rows, meta in new_shards(self.eval_shards, self._eval_seen, self.args.action_bins):  # and never will
            if meta.get("tag") == "eval":
                self._score_fresh(rows, "eval")
        self.top_up(pool)

    def _score_fresh(self, rows: dict[str, np.ndarray], tag: str) -> None:
        if self.step and len(rows["labels"]):
            n = len(rows["labels"])
            if self.args.temporal:  # one contiguous stretch, its memory built over a burn-in first
                length = min(n, self.args.burn_in + self.args.fresh_rows)
                start = int(self._rng.integers(n - length + 1))
                burn = min(self.args.burn_in, length - 1)
                self.learner.score_sequence({k: rows[k][start:start + length] for k in FIELDS + (ANGLES,)},
                                            self.fresh[tag], burn)
                return
            pick = self._rng.choice(n, min(self.args.fresh_rows, n), replace=False)
            self.learner.score({k: rows[k][np.sort(pick)] for k in FIELDS + (ANGLES,)}, self.fresh[tag])

    def _record(self, summaries: list[dict]) -> None:
        for summary in summaries:
            if summary["student"] is not None:
                summary["version_lag"] = self.version - summary["student_version"]
            self.match_log.write(json.dumps(summary) + "\n")
            self.collected_rows += summary["rows"]
            self._matches_received += 1
            self._pipeline["matches"] += 1
            self._pipeline["match_wall_s"] += summary["wall_s"]
            # the part of this match's wall time that fell inside the current log window
            self._pipeline["busy_s"] += min(summary["wall_s"], time.time() - self._window_started)
            if summary["tag"].startswith("eval"):
                self._log_eval(summary)
            else:
                self._collect_window.append(summary)
        self.match_log.flush()

    def _log_eval(self, summary: dict) -> None:
        tag, version = summary["tag"], summary["version"]
        group = self._evals[(tag, version)]
        group.append(summary)
        if len(group) == 2 * self.args.eval_seeds:
            self._write_eval(tag, version)

    def _write_eval(self, tag: str, version: int) -> None:
        group = self._evals.pop((tag, version))
        stats = summarize_matches(group)
        self.metrics.write(tag, self.step, {"version": version, **stats})
        who = "teacher" if tag == "eval_teacher" else f"v{version}"
        print(f"[distill] {tag} {who} vs {self.args.eval_opponent}: margin {stats['mean_margin']:+.1f}"
              f"±{stats['margin_se']:.1f} win {stats['win_rate']:.2f} | H/Ca/C {stats['units_hunter']:.1f}/"
              f"{stats['units_carrier']:.1f}/{stats['units_collector']:.1f} | agree {stats.get('agree/all', float('nan')):.2f} "
              f"| deliv/1k {stats['learner/deliveries_per_1000']:.1f} lost/1k {stats['learner/cargo_lost_per_1000']:.1f} "
              f"| blocked/1k {stats['movement/blocked_per_1000']:.1f}"
              + (f" | angle {stats['angle_err_deg/all']:.1f} deg (<=2: {stats['angle_within2/all']:.2f}), first 10 s "
                 f"{stats.get('angle_err_deg/opening', float('nan')):.1f} ({stats.get('angle_within2/opening', float('nan')):.2f})"
                 if "angle_err_deg/all" in stats else "")
              + ("" if len(group) == 2 * self.args.eval_seeds else f" (incomplete: {len(group)} matches)"), flush=True)

    def log(self, started: float, window_started: float) -> None:
        args, p = self.args, self._pipeline
        seconds = max(1e-6, time.time() - window_started)
        stats = {
            "version": self.version, "beta": beta_for(self.version, args),
            "replay_ratio": self.step * self.learner.rows_per_step / max(1, self.collected_rows),
            "collected_rows": self.collected_rows, "elapsed_s": time.time() - started,
            "pipeline/steps_per_s": args.log_every / seconds,
            "pipeline/matches_per_min": 60 * p["matches"] / seconds,
            "pipeline/mean_match_wall_s": p["match_wall_s"] / max(1, p["matches"]),
            # worker time spent inside this window on matches that finished in it, over worker time
            "pipeline/worker_utilization": p["busy_s"] / (args.workers * seconds),
            "pipeline/queued_per_worker": p["queued_per_worker"] / max(1, p["syncs"]),
            "pipeline/throttled_sync_frac": p["throttled_syncs"] / max(1, p["syncs"]),
            "system/peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if platform.system() == "Darwin" else 2**20),
            **self.dataset.composition(self.version),
            **self.learner.pop_stats(),
        }
        for tag, accuracy in self.fresh.items():
            stats.update(accuracy.pop(f"fresh_{tag}"))
        self._pipeline.clear()
        self.metrics.write("train", self.step, stats)
        if self._collect_window:
            self.metrics.write("collect", self.step, {"version": self.version, **summarize_matches(self._collect_window)})
            by_category = defaultdict(list)
            for summary in self._collect_window:
                by_category[opponent_category(summary.get("opponent_played", summary["opponent"]))].append(summary)
            for category, group in by_category.items():  # e.g. collect_top_tier: margin against V17-V19
                self.metrics.write(f"collect_{category}", self.step, {"version": self.version, **summarize_matches(group)})
            self._collect_window = []
        fresh = f" fresh {stats['fresh_collect/accuracy']:.3f}" if "fresh_collect/accuracy" in stats else ""
        if "fresh_collect/angle_err_deg/all" in stats:
            fresh += (f" angle {stats['fresh_collect/angle_err_deg/all']:.2f} deg"
                      f" (<=2 {stats['fresh_collect/angle_within2/all']:.2f}, 10 s {stats.get('fresh_collect/angle_err_deg/opening', float('nan')):.2f})")
        print(f"[distill] step {self.step} v{self.version} beta {stats['beta']:.2f} rows {self.dataset.size} "
              f"ratio {stats['replay_ratio']:.1f} | ce {stats['train/ce']:.3f} acc {stats['train/accuracy']:.3f}{fresh} "
              f"| {stats['pipeline/steps_per_s']:.1f} step/s, workers {stats['pipeline/worker_utilization']:.0%} "
              f"| {stats['elapsed_s']:.0f}s", flush=True)

    # ------------------------------------------------------------------------------- loop

    def run(self) -> None:
        args = self.args
        started = time.time()
        with CollectorPool(args.build, args.workers, args.time_scale) as pool:
            if not self.resumed and args.baseline_eval:
                self.queue_eval(pool, "eval_teacher")
            self.top_up(pool)
            while self.dataset.size < args.min_rows:  # teacher-only warm-up
                self.sync(pool, timeout=5.0)
            self._load_probe()
            print(f"[distill] {self.dataset.size} rows, fitting", flush=True)
            window_started = self._window_started = time.time()
            while self.step < args.steps:
                self.learner.set_lr(self.step)
                if args.temporal:
                    self.learner.step_windows(self.dataset.sample_windows(args.windows, args.window, args.burn_in, self._rng))
                else:  # with STAND, all-unlabelled rows are targets too (the teacher stands everywhere there)
                    self.learner.step(self.dataset.sample(args.batch_size, self._rng, labelled_only=not args.stand_action))
                self.step += 1
                if self.step % args.sync_every == 0:
                    self.sync(pool)
                if self.step % args.publish_every == 0:
                    self.publish()
                    if self.version % args.eval_every == 0:
                        self.queue_eval(pool)
                if self.step % args.log_every == 0:
                    self.log(started, window_started)
                    window_started = self._window_started = time.time()
            self.publish()
            self.queue_eval(pool)  # the final weights are always evaluated
            # running matches and every queued eval finish into the logs (and shards)
            self._record(pool.drain(keep=lambda task: task.tag.startswith("eval")))
            for tag, version in list(self._evals):
                self._write_eval(tag, version)
        self.metrics.close()
        self.match_log.close()
        print(f"[distill] done: {self.latest} (v{self.version})")


def main(argv=None) -> None:
    Run(parse_args(argv)).run()


if __name__ == "__main__":
    main()
