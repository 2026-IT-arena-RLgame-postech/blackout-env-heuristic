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
import time
from pathlib import Path

import numpy as np

from blackout_env.train.offline_dataset import load_dataset_into
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer, default_run_dir


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

    if args.resume:
        trainer.load(Path(args.resume))
        print(f"[offline] resumed from {args.resume} at train_step_count={trainer.train_step_count}")

    print(f"[offline] loading dataset from {dataset_dir} ...")
    load_dataset_into(trainer.buffer_a, dataset_dir / "buffer_a.npz")
    load_dataset_into(trainer.buffer_b, dataset_dir / "buffer_b.npz")
    print(f"[offline] loaded buffer_a={len(trainer.buffer_a)}, buffer_b={len(trainer.buffer_b)} transitions")

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer._total_env_steps_hint = args.steps  # spans n_step/gamma/per_beta annealing over [0, steps]
    recent_losses: list[float] = []
    t0 = time.time()
    start_step = trainer.train_step_count
    try:
        while trainer.train_step_count < args.steps:
            trainer.env_step_count = trainer.train_step_count  # drives the annealing schedules above
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
    except KeyboardInterrupt:
        print(f"\n[offline] KeyboardInterrupt at step {trainer.train_step_count} -- saving before exit")
        trainer.save(ckpt_dir / f"interrupted_step_{trainer.train_step_count}.pt")
        raise
    finally:
        trainer.tb.close()

    trainer.save(ckpt_dir / "final.pt")
    print(f"[offline] done. next:\n"
          f"  python -m blackout_env.train.qmix_trainer --build <build> --steps 1000000 "
          f"--resume {ckpt_dir / 'final.pt'} --skip-bootstrap --seed-dataset-dir {dataset_dir}")


if __name__ == "__main__":
    main()
