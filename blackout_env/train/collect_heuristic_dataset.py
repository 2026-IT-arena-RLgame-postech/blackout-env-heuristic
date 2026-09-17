"""
Standalone entry point: collect a fixed offline dataset of heuristic-vs-heuristic (+ a little
uniform-random exploration noise) transitions, to later pretrain on with offline_pretrain.py
(no Unity needed for that second step).

This is exactly QMIXTrainer's existing phase-1 bootstrap (see qmix_trainer.py's module
docstring and QMIXConfig.heuristic_fill_frac) driven standalone: same select_actions_heuristic
action selection, same absorption-boundary buffer_done bookkeeping in collect_step(), same
per-team SequentialReplayBuffer push -- just run to completion and saved to disk instead of
being consumed live by an online run. heuristic_fill_frac=1.0 (its default) combined with
sizing the buffers to exactly --steps guarantees collect_step() stays in phase 1 for the whole
run: it can only switch to phase 2 once a stream's buffer is full, which happens (if at all)
on the very last step.

Usage:
    python -m blackout_env.train.collect_heuristic_dataset \\
        --build build/mac/BlackOut.app --steps 100000 --noise-frac 0.1 --out datasets/run1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.train.offline_dataset import save_buffer, write_collection_info
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="Path to the Unity build executable")
    parser.add_argument("--steps", type=int, required=True, help="Env steps to collect (per stream)")
    parser.add_argument(
        "--noise-frac",
        type=float,
        default=0.1,
        help="Per-unit probability of a uniformly random compass direction instead of the "
        "heuristic's own choice (QMIXConfig.heuristic_bootstrap_noise_frac) -- widens the "
        "offline dataset's action coverage beyond exactly what the heuristics themselves "
        "would choose. 0 reproduces plain heuristic-vs-heuristic rollouts.",
    )
    parser.add_argument("--out", required=True, help="Output directory for buffer_a.npz/buffer_b.npz")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale")
    parser.add_argument("--graphics", action="store_true", help="Show the Unity window instead of headless")
    parser.add_argument("--heuristic-seed-a", type=int, default=0)
    parser.add_argument("--heuristic-seed-b", type=int, default=1)
    parser.add_argument(
        "--keep-exhausted",
        action="store_true",
        help="Play every match to its Unity end. By default a match ends at the first absorption "
        "after which no battery is left anywhere (~73% of a heuristic match's rows, which "
        "offline_pretrain drops anyway), with the outcome decided by the scores at that point.",
    )
    parser.add_argument(
        "--no-unity-shaping",
        action="store_true",
        help="Launch Unity with -noRewardShaping (~2.2x faster Unity steps). The saved reward then "
        "has no potential shaping, so the dataset is only usable with offline_pretrain --reward v2.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BlackOutEnv(
        env_path=args.build,
        map_w=24,
        map_h=24,
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
        unity_shaping=not args.no_unity_shaping,
    )
    config = QMIXConfig(
        buffer_capacity=args.steps,
        heuristic_fill_frac=1.0,  # stay in phase 1 for the entire run (see module docstring)
        heuristic_bootstrap_noise_frac=args.noise_frac,
        stop_when_exhausted=not args.keep_exhausted,
        heuristic_seed_a=args.heuristic_seed_a,
        heuristic_seed_b=args.heuristic_seed_b,
        tb_log_dir=None,  # this is a data-collection run, not a training run -- nothing to plot
    )
    trainer = QMIXTrainer(env, config)

    try:
        obs, _ = trainer._reset_env()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            obs = trainer.collect_step(obs)
            if step % 1000 == 0 or step == args.steps:
                elapsed = time.time() - t0
                print(
                    f"[collect] step {step}/{args.steps} "
                    f"({step / elapsed:.1f} steps/s, buffer_a={len(trainer.buffer_a)}, buffer_b={len(trainer.buffer_b)}, "
                    f"matches={trainer._episode_count}, ended at exhaustion={trainer.exhausted_stops})"
                )
    finally:
        env.close()

    save_buffer(trainer.buffer_a, out_dir / "buffer_a.npz")
    save_buffer(trainer.buffer_b, out_dir / "buffer_b.npz")
    write_collection_info(out_dir, unity_shaping=not args.no_unity_shaping, stop_when_exhausted=not args.keep_exhausted)
    print(f"[collect] saved {len(trainer.buffer_a)} transitions/stream to {out_dir}")
    print(
        f"[collect] next: python -m blackout_env.train.offline_pretrain "
        f"--dataset-dir {out_dir} --steps <gradient steps> --checkpoint-dir <dir>"
    )


if __name__ == "__main__":
    main()
