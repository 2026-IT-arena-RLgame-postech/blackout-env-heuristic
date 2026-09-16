"""
Standalone entry point: pure offline batch-RL pretraining on a fixed dataset collected by
collect_heuristic_dataset.py. No Unity/BlackOutEnv process at all -- QMIXTrainer(env=None, ...)
builds the exact same net/mixer/optimizer as the online trainer, and this script only ever
calls train_step() (gradient updates against sampled batches) against buffers loaded straight
from disk, never collect_step()/env.step().

--steps is an ABSOLUTE target on trainer.train_step_count, same convention as qmix_trainer.py's
own --steps/--resume: with --resume, the run continues from wherever the checkpoint left off and
stops once train_step_count reaches --steps (so a second call with a larger --steps just does
the remaining gradient steps, not another --steps from scratch).

The output checkpoint is written in the same format QMIXTrainer.save()/load() already use, so
it plugs directly into the online trainer's --resume:

    python -m blackout_env.train.qmix_trainer --build <build> --steps 1000000 \\
        --resume <checkpoint_dir>/final.pt --skip-bootstrap --seed-dataset-dir datasets/run1

--skip-bootstrap matters here: buffer_a/buffer_b start empty again in that online run (only net
weights + optimizer state are checkpointed, not replay data), so without it phase 1 would
re-trigger and spend its first heuristic_fill_frac*capacity steps re-collecting pure-heuristic
data into a fresh buffer before the pretrained net gets to act at all -- exactly the redundant
detour this two-stage pipeline (offline pretrain -> online fine-tune) is meant to skip.
--seed-dataset-dir (usually the same dataset this script just pretrained on) preloads that
online run's buffers instead of leaving them empty, so early train_step() calls after the
handoff still sample real data rather than starving until enough fresh online steps arrive --
see qmix_trainer.py's own docstring on that flag for how the FIFO buffer phases it out.

Usage:
    python -m blackout_env.train.offline_pretrain \\
        --dataset-dir datasets/run1 --steps 200000 --checkpoint-dir checkpoints/offline_run1

    # resume an interrupted or already-finished run to do more gradient steps:
    python -m blackout_env.train.offline_pretrain \\
        --dataset-dir datasets/run1 --steps 400000 \\
        --checkpoint-dir checkpoints/offline_run1 --resume checkpoints/offline_run1/final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import TEAM_A_INDICES, TEAM_B_INDICES
from blackout_env.heuristics import RecommendedStrategicHeuristic
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.offline_dataset import load_dataset_into
from blackout_env.train.onpolicy_collect import collect_onpolicy_data
from blackout_env.train.periodic_eval import run_periodic_eval
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer, default_run_dir
from blackout_env.train.replay_buffer import SOURCE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Dir with buffer_a.npz/buffer_b.npz from collect_heuristic_dataset.py")
    parser.add_argument("--steps", type=int, required=True, help="Absolute train_step_count target (see module docstring re: --resume)")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path to continue from (net/optimizer/dist_mixer/spr + train_step_count "
        "-- same format qmix_trainer.py saves/loads). The dataset is still reloaded fresh from "
        "--dataset-dir either way; only the checkpoint's replay buffer is never restored.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Default: fresh timestamped folder under checkpoints/offline/ -- kept apart from "
        "online qmix_trainer.py runs (checkpoints/<ts>/) so offline pretrain checkpoints don't "
        "mix in with them in listings. Pass the same dir back in via --resume to keep a run's "
        "checkpoints together across an interruption.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5_000, help="Gradient steps between periodic checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="Override QMIXConfig.lr")
    parser.add_argument("--batch-size", type=int, default=None, help="Override QMIXConfig.batch_size")
    parser.add_argument(
        "--spr-loss-weight",
        type=float,
        default=None,
        help="Override QMIXConfig.spr_loss_weight (default 1.0, i.e. equal weight with the IQN "
        "loss -- see total_loss in QMIXTrainer.train_step). The original BBF paper's own config "
        "(google-research/bigger_better_faster, BBF.gin) uses BBFAgent.spr_weight=5, 5x heavier "
        "than the RL loss, not 1:1.",
    )
    parser.add_argument(
        "--bc-loss-alpha",
        type=float,
        default=None,
        help="Override QMIXConfig.bc_loss_alpha (default 0.0, i.e. off). Adds "
        "cross_entropy(q_values, dataset_action) per own-team unit to total_loss -- a discrete- "
        "BCQ-style (Fujimoto et al. 2019) behavior-cloning auxiliary loss that supervises the "
        "net to reproduce the exact direction the heuristic dataset actually took, forcing it "
        "to explain wall-avoidance/item-seeking decisions that depend on the vision channel "
        "(unlike pure TD/SPR losses, which a purely offline heuristic dataset can satisfy "
        "without the net ever needing to parse graphic_encoder's output -- see "
        "docs/offline_pretrain_runs.md). NOT a raw loss weight: the BC term is rescaled every "
        "step to iqn_loss's own current magnitude first (TD3+BC-style, Fujimoto & Gu 2021), "
        "THEN multiplied by this alpha -- so alpha=1.0 means BC and TD contribute equally "
        "regardless of their differing raw scales (cross-entropy over 8 actions starts near "
        "ln(8)=2.08, ~2 orders of magnitude above this environment's iqn_loss). Applied only to "
        "heuristic-played transitions: the static dataset and the heuristic side of "
        "self-vs-heuristic matches -- rows the net itself played carry its own actions and are "
        "excluded.",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Override QMIXConfig.encoder_lr -- separate AdamW lr for MyModel.graphic_encoder "
        "only, independent of --lr for the rest of the net. Default: None, i.e. same as --lr. "
        "Added after observing grad_norm/graphic_encoder decay to ~1e-7 (vs 1e-2..1e-1 for "
        "every other component) over a 200k-step run -- see docs/offline_pretrain_runs.md.",
    )
    parser.add_argument(
        "--encoder-weight-decay",
        type=float,
        default=None,
        help="Override QMIXConfig.encoder_weight_decay -- separate AdamW weight_decay for "
        "MyModel.graphic_encoder only. Default: None, i.e. same as --lr's weight_decay "
        "(config.weight_decay).",
    )
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=None,
        help="Gradient steps between BBF-style shrink-and-perturb resets of graphic_encoder/"
        "attention (see QMIXConfig.reset_interval), which also re-anneals n_step/gamma from "
        "scratch each cycle (see QMIXTrainer._anneal_frac). Previously this script only ever "
        "called train_step() and never maybe_reset(), so n_step/gamma annealed exactly once "
        "over the whole run and then sat frozen at their END values -- the opposite of BBF's "
        "repeated resets-with-re-annealing. Default: steps//5 (BBF resets ~5x over a run); "
        "pass 0 to disable resets and keep the old one-shot-anneal behavior.",
    )
    parser.add_argument(
        "--reset-warmup-steps",
        type=int,
        default=None,
        help="Override QMIXConfig.reset_warmup_steps -- linear lr warmup (env steps) for "
        "graphic_encoder's own param group only, ramping 0 -> encoder_lr at the start of each "
        "reset cycle. _reset_submodule() wipes this submodule's Adam state on every reset, so "
        "the first post-reset steps have no momentum/variance history -- exactly the regime "
        "Adam warmup schedules exist to smooth over, scoped to graphic_encoder since that's the "
        "one component repeatedly found fragile right after reset/init (see "
        "docs/offline_pretrain_runs.md). Default None (0, i.e. off).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        help="Train steps between periodic heuristic-match eval windows (win/loss/margin + "
        "idle/blocked movement diagnostics, see blackout_env/train/periodic_eval.py and "
        "movement_monitor.py -- same idle/blocked definitions examples/benchmark_heuristics.py "
        "uses). Catches a degenerate policy (e.g. the graphic_encoder gradient-vanishing issue "
        "that produced a walks-into-walls agent, docs/offline_pretrain_runs.md) within a few "
        "windows instead of only at the end of a many-hour run. Pass 0 to disable (no Unity "
        "process is started at all in that case).",
    )
    parser.add_argument(
        "--eval-build", type=Path, default=Path("build/mac/BlackOut.app"), help="Unity build used only for periodic eval matches"
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 202, 303], help="Each seed is played both non-swapped and swapped")
    parser.add_argument("--eval-time-scale", type=float, default=100.0, help="Unity time scale for eval matches (headless, so fast by default)")
    parser.add_argument("--eval-graphics", action="store_true", help="Show the Unity window during eval matches (default: headless)")
    parser.add_argument(
        "--blocked-penalty",
        type=float,
        default=0.0,
        help="Per-blocked-unit-tick reward penalty (see reward_shaping.blocked_penalty_adjustment) "
        "applied to both the static dataset (retroactively, at load time) and any on-policy data "
        "collected via --onpolicy-*-frac below. 'Blocked' = commanded movement, no actual "
        "displacement (walking into a wall/obstacle) -- see diagnose_stopping2.py's Run4 finding "
        "of 41.66% blocked unit-ticks and docs/offline_pretrain_runs.md for why nothing in the "
        "actual game reward (reward_config.json) penalizes this directly. Default 0.0 (off, i.e. "
        "exact prior behavior); ~0.02 was the value discussed against this dataset's own reward "
        "scale (typical nonzero |reward| ~0.005, max single-tick nav-shaping ~0.08).",
    )
    parser.add_argument(
        "--onpolicy-self-vs-heuristic-frac",
        type=float,
        default=0.0,
        help="Target fraction of the dataset's size worth of self(candidate)-vs-heuristic ticks to "
        "collect over the whole run (into this source's own FIFO buffer, see "
        "--onpolicy-buffer-capacity), spread evenly across "
        "--eval-interval windows (skips the step-0 baseline window, since that checkpoint is "
        "untrained). Also sets this source's fixed share of every training batch: batches are "
        "drawn dataset : self-vs-heuristic : self-play = (1 - both fracs) : this : "
        "--onpolicy-self-play-frac, each source with its own PER priorities (a source with no "
        "data yet has its share spread over the others). Default 0.0 (off).",
    )
    parser.add_argument(
        "--onpolicy-self-play-frac",
        type=float,
        default=0.0,
        help="Same as --onpolicy-self-vs-heuristic-frac but for candidate-vs-itself matches, run "
        "after the self-vs-heuristic collection in each window; also this source's batch share. "
        "Default 0.0 (off).",
    )
    parser.add_argument(
        "--onpolicy-buffer-capacity",
        type=int,
        default=262_144,
        help="Rows per stream in each on-policy source's own FIFO buffer (rounded up to a power "
        "of 2). The static dataset stays in its own buffer and is never overwritten; on-policy "
        "data only ever displaces older on-policy data of the same source. ~30KB per row per "
        "stream, allocated lazily. Default 262144 (~17 eval windows of self-vs-heuristic data at 0.3, ~7.9GB per stream when full).",
    )
    parser.add_argument(
        "--onpolicy-max-matches",
        type=int,
        default=50,
        help="Safety cap on matches played per phase (self-vs-heuristic, self-play) per window, "
        "in case matches turn out much shorter than expected and the target tick count would "
        "otherwise take unboundedly many matches to reach.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--tb-log-dir",
        default=None,
        help="Default: fresh timestamped folder under runs/offline/ -- kept apart from online "
        "qmix_trainer.py runs (runs/<ts>/) so TensorBoard's run list doesn't mix pretrain and "
        "online curves together; open both dirs as separate runs to compare loss curves across "
        "the pretrain->online handoff. Pass '' to disable.",
    )
    args = parser.parse_args()
    # Runs are normally launched with stdout redirected to a log file, where Python block-buffers
    # and the log can lag the real step count by tens of thousands of steps.
    sys.stdout.reconfigure(line_buffering=True)

    dataset_dir = Path(args.dataset_dir)

    config_kwargs = dict(
        # Sized off the max shard length below, once loaded -- placeholder here, overwritten
        # after the dataset load knows the real transition count.
        buffer_capacity=1,
        device=args.device,
        compile=args.compile,
        checkpoint_interval=args.checkpoint_interval,
    )
    if args.lr is not None:
        config_kwargs["lr"] = args.lr
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.spr_loss_weight is not None:
        config_kwargs["spr_loss_weight"] = args.spr_loss_weight
    if args.bc_loss_alpha is not None:
        config_kwargs["bc_loss_alpha"] = args.bc_loss_alpha
    if args.encoder_lr is not None:
        config_kwargs["encoder_lr"] = args.encoder_lr
    if args.encoder_weight_decay is not None:
        config_kwargs["encoder_weight_decay"] = args.encoder_weight_decay
    if args.reset_warmup_steps is not None:
        config_kwargs["reset_warmup_steps"] = args.reset_warmup_steps
    onpolicy_frac = args.onpolicy_self_vs_heuristic_frac + args.onpolicy_self_play_frac
    if not 0.0 <= onpolicy_frac <= 1.0 or min(args.onpolicy_self_vs_heuristic_frac, args.onpolicy_self_play_frac) < 0.0:
        parser.error("--onpolicy-self-vs-heuristic-frac and --onpolicy-self-play-frac must be >= 0 and sum to <= 1")
    if onpolicy_frac > 0.0:
        config_kwargs["batch_source_fracs"] = (
            1.0 - onpolicy_frac,
            args.onpolicy_self_vs_heuristic_frac,
            args.onpolicy_self_play_frac,
        )
        config_kwargs["onpolicy_buffer_capacity"] = args.onpolicy_buffer_capacity
    config_kwargs["reset_interval"] = (
        args.reset_interval if args.reset_interval is not None else max(1, args.steps // 5)
    )
    # Grouped under an "offline/" subfolder in both cases -- kept apart from online
    # qmix_trainer.py runs (checkpoints/<ts>/, runs/<ts>/) rather than defaulting to
    # QMIXConfig's own top-level default_run_dir(), so listing either directory doesn't mix
    # pretrain and online-run artifacts together.
    config_kwargs["checkpoint_dir"] = args.checkpoint_dir if args.checkpoint_dir is not None else default_run_dir(base="checkpoints/offline")
    if args.tb_log_dir is not None:
        config_kwargs["tb_log_dir"] = args.tb_log_dir or None  # '' -> disable
    else:
        config_kwargs["tb_log_dir"] = default_run_dir(base="runs/offline")

    # buffer_capacity has to be known before QMIXTrainer() builds buffer_a/buffer_b, so peek at
    # the dataset's size first (cheap -- .npz headers only, no full array load) rather than
    # loading twice.
    n_a = np.load(dataset_dir / "buffer_a.npz")["graphic"].shape[0]
    n_b = np.load(dataset_dir / "buffer_b.npz")["graphic"].shape[0]
    config_kwargs["buffer_capacity"] = max(n_a, n_b)

    config = QMIXConfig(**config_kwargs)
    trainer = QMIXTrainer(env=None, config=config)
    print(f"[offline] checkpoint_dir={config.checkpoint_dir}")
    print(f"[offline] tb_log_dir={config.tb_log_dir or '(disabled)'}")
    print(f"[offline] reset_interval={config.reset_interval or '(disabled -- single anneal over the whole run)'}")
    print(
        f"[offline] batch_source_fracs (dataset, self_vs_heuristic, self_play)="
        f"{config.batch_source_fracs or '(off -- dataset buffer only)'}"
        + "".join(f", {SOURCE_NAMES[src]} FIFO buffer {a.capacity} rows/stream" for src, (a, _) in trainer.onpolicy_buffers.items())
    )

    if args.resume:
        trainer.load(Path(args.resume))
        print(f"[offline] resumed from {args.resume} at train_step_count={trainer.train_step_count}")

    print(f"[offline] loading dataset from {dataset_dir} ...")
    load_dataset_into(trainer.buffer_a, dataset_dir / "buffer_a.npz", TEAM_A_INDICES, args.blocked_penalty)
    load_dataset_into(trainer.buffer_b, dataset_dir / "buffer_b.npz", TEAM_B_INDICES, args.blocked_penalty)
    print(f"[offline] loaded buffer_a={len(trainer.buffer_a)}, buffer_b={len(trainer.buffer_b)} transitions"
          f" (blocked_penalty={args.blocked_penalty})")

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Lazily started (only if eval is actually enabled) so --eval-interval 0 never touches
    # Unity at all, same as the rest of this script.
    eval_env: BlackOutEnv | None = None
    eval_opponent = RecommendedStrategicHeuristic()
    if args.eval_interval > 0:
        eval_env = BlackOutEnv(
            str(args.eval_build), time_scale=args.eval_time_scale, no_graphics=not args.eval_graphics
        )
        print(f"[offline] periodic eval enabled: every {args.eval_interval} steps, "
              f"{len(args.eval_seeds) * 2} matches vs {type(eval_opponent).__name__}")

    def run_eval(step: int) -> None:
        candidate = MyPolicy(trainer.net, device=args.device)
        trainer.net.eval()
        try:
            metrics = run_periodic_eval(eval_env, candidate, eval_opponent, args.eval_seeds)
        finally:
            trainer.net.train()
        trainer.tb.scalars("eval", metrics, step)
        print(
            f"[offline] eval @ step {step}: win_rate={metrics['win_rate']:.2f} "
            f"loss_rate={metrics['loss_rate']:.2f} draw_rate={metrics['draw_rate']:.2f} "
            f"mean_margin={metrics['mean_margin']:.2f} "
            f"candidate_idle/1k={metrics['candidate_idle_per_1000_ticks']:.1f} "
            f"candidate_blocked/1k={metrics['candidate_blocked_per_1000_ticks']:.1f}\n"
            # Scoring pipeline next to the heuristic's own numbers from the same matches -- the
            # approach/pickup gap is what Run 6's post-mortem pinned the loss on (see
            # train/objective_monitor.py), so it belongs in the console line, not only in TB.
            f"           objectives (candidate vs opponent): "
            f"approach_battery={metrics['candidate_approach_battery']:+.4f}/{metrics['opponent_approach_battery']:+.4f} "
            f"pickups/1k={metrics['candidate_pickups_per_1000_ticks']:.1f}/{metrics['opponent_pickups_per_1000_ticks']:.1f} "
            f"deliveries/1k={metrics['candidate_deliveries_per_1000_ticks']:.1f}/{metrics['opponent_deliveries_per_1000_ticks']:.1f} "
            f"cargo_lost/1k={metrics['candidate_cargo_lost_per_1000_ticks']:.1f}/{metrics['opponent_cargo_lost_per_1000_ticks']:.1f} "
            f"| by side: margin={metrics.get('as_team_a/mean_margin', float('nan')):.1f}/"
            f"{metrics.get('as_team_b/mean_margin', float('nan')):.1f} "
            f"blocked/1k={metrics.get('as_team_a/blocked_per_1000_ticks', float('nan')):.1f}/"
            f"{metrics.get('as_team_b/blocked_per_1000_ticks', float('nan')):.1f}"
        )

    n_eval_windows = max(1, args.steps // args.eval_interval) if args.eval_interval > 0 else 0
    target_svh_per_window = int(args.onpolicy_self_vs_heuristic_frac * config.buffer_capacity / max(1, n_eval_windows))
    target_sp_per_window = int(args.onpolicy_self_play_frac * config.buffer_capacity / max(1, n_eval_windows))
    onpolicy_enabled = eval_env is not None and (target_svh_per_window > 0 or target_sp_per_window > 0)
    if onpolicy_enabled:
        print(
            f"[offline] on-policy data collection enabled: ~{target_svh_per_window} self-vs-heuristic "
            f"+ ~{target_sp_per_window} self-play ticks per eval window ({n_eval_windows} windows), "
            f"blocked_penalty={args.blocked_penalty}"
        )

    def collect_onpolicy(step: int) -> None:
        candidate = MyPolicy(trainer.net, device=args.device)
        trainer.net.eval()
        try:
            stats = collect_onpolicy_data(
                eval_env,
                trainer.onpolicy_buffers,
                candidate,
                eval_opponent,
                target_svh_per_window,
                target_sp_per_window,
                args.blocked_penalty,
                seed_start=10_000 + step,
                max_matches=args.onpolicy_max_matches,
            )
        finally:
            trainer.net.train()
        trainer.tb.scalars("onpolicy", stats, step)
        svh_summary = ""
        if "self_vs_heuristic/win_rate" in stats:
            svh_summary = (
                f" | svh W/L/D={stats['self_vs_heuristic/win_rate']:.2f}/{stats['self_vs_heuristic/loss_rate']:.2f}/"
                f"{stats['self_vs_heuristic/draw_rate']:.2f} margin={stats['self_vs_heuristic/mean_margin']:.1f} "
                f"env_r/tick={stats['self_vs_heuristic/candidate_env_reward_per_tick']:.5f} "
                f"penalty/tick={stats['self_vs_heuristic/candidate_blocked_penalty_per_tick']:.5f} "
                f"psi_saturated={stats['self_vs_heuristic/psi_saturated_frac']:.2f} "
                f"approach_battery={stats['self_vs_heuristic/candidate_approach_battery']:+.4f}"
                f"/{stats['self_vs_heuristic/opponent_approach_battery']:+.4f} "
                f"cargo_lost/1k={stats['self_vs_heuristic/candidate_cargo_lost_per_1000_ticks']:.1f}"
                f"/{stats['self_vs_heuristic/opponent_cargo_lost_per_1000_ticks']:.1f}"
            )
        print(
            f"[offline] onpolicy collect @ step {step}: "
            f"self_vs_heuristic={stats['self_vs_heuristic_ticks']} ticks/{stats['self_vs_heuristic_matches']} matches, "
            f"self_play={stats['self_play_ticks']} ticks/{stats['self_play_matches']} matches "
            f"(onpolicy buffer rows a/b: "
            + ", ".join(f"{SOURCE_NAMES[src]}={len(a)}/{len(b)}" for src, (a, b) in trainer.onpolicy_buffers.items())
            + f"){svh_summary}"
        )

    trainer._total_env_steps_hint = args.steps  # spans n_step/gamma/per_beta annealing over [0, steps]
    recent_losses: list[float] = []
    t0 = time.time()
    start_step = trainer.train_step_count
    try:
        if eval_env is not None and start_step == 0:
            run_eval(0)  # baseline before any gradient steps, for comparison against later windows
        while trainer.train_step_count < args.steps:
            trainer.env_step_count = trainer.train_step_count  # drives the annealing schedules above
            if config.reset_interval > 0 and trainer.env_step_count % config.reset_interval == 0:
                print(f"[offline] step {trainer.env_step_count}: BBF reset (shrink-and-perturb + re-anneal n_step/gamma)")
            trainer.maybe_reset()  # gated on env_step_count % reset_interval -- see QMIXTrainer.maybe_reset
            loss = trainer.train_step()  # increments trainer.train_step_count itself
            if loss is not None:
                recent_losses.append(loss)

            step = trainer.train_step_count
            if step % 1000 == 0 or step == args.steps:
                elapsed = time.time() - t0
                done = step - start_step
                avg_loss = sum(recent_losses[-1000:]) / len(recent_losses[-1000:]) if recent_losses else float("nan")
                print(f"[offline] step {step}/{args.steps} ({done / elapsed:.1f} steps/s) avg_loss={avg_loss:.4f}")

            if step % args.checkpoint_interval == 0:
                trainer.save(ckpt_dir / f"step_{step}.pt")

            if eval_env is not None and step % args.eval_interval == 0:
                run_eval(step)
                if onpolicy_enabled:
                    collect_onpolicy(step)
    except KeyboardInterrupt:
        print(f"\n[offline] KeyboardInterrupt at step {trainer.train_step_count} -- saving before exit")
        trainer.save(ckpt_dir / f"interrupted_step_{trainer.train_step_count}.pt")
        raise
    finally:
        trainer.tb.close()
        if eval_env is not None:
            eval_env.close()

    trainer.save(ckpt_dir / "final.pt")
    print(f"[offline] done. next:\n"
          f"  python -m blackout_env.train.qmix_trainer --build <build> --steps 1000000 "
          f"--resume {ckpt_dir / 'final.pt'} --skip-bootstrap --seed-dataset-dir {dataset_dir}")


if __name__ == "__main__":
    main()
