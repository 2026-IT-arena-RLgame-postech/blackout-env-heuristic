"""
Two trained checkpoints against each other, both sides per seed, with the Unity window visible.

Each --seeds value is played twice (A as team A, then as team B), so spawn-side advantage
cancels. Prints every match and a summary: A's points (win 1, draw 0.5) out of the match count
and A's mean margin in game points. Passing the same checkpoint to --a and --b is self-play.

Flags: --a / --b label=path (required), --build (default build/mac/BlackOut.app),
--seeds (default 404 505 606), --time-scale (default 3), --headless (no window; the default
shows the Unity window), --device (default cpu).

Used by: models/run11_step80k/run11_pipeline.sh selfplay [ckpt] and vs <ckpt_b> [ckpt].
For a rating across several checkpoints, use examples/elo_checkpoints.py instead.

Usage:
    python examples/evaluate_checkpoint_vs_checkpoint.py \\
        --a run11_80k=checkpoints/offline/<run>/step_80000.pt --b run11_final=checkpoints/offline/<run>/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv, run_match
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer


def _load(spec: str, device: str) -> tuple[str, MyPolicy]:
    label, _, path = spec.partition("=")
    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=device, tb_log_dir=None))
    trainer.load(Path(path))
    trainer.net.eval()
    print(f"[eval] {label}: {path} (train_step_count={trainer.train_step_count})")
    return label, MyPolicy(trainer.net, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="label=checkpoint")
    parser.add_argument("--b", required=True, help="label=checkpoint")
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[404, 505, 606])
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    name_a, policy_a = _load(args.a, args.device)
    name_b, policy_b = _load(args.b, args.device)
    env = BlackOutEnv(str(args.build), time_scale=args.time_scale, no_graphics=args.headless, unity_shaping=False)
    margins, wins_a = [], 0.0
    try:
        for seed in args.seeds:
            for swap in (False, True):
                r = run_match(env, policy_a, policy_b, swap_teams=swap, seed=seed)
                margin = (r.model_a_score - r.model_b_score) * 100
                margins.append(margin)
                wins_a += 1.0 if r.winner == 0 else 0.5 if r.winner is None else 0.0
                label = f"{name_a} wins" if r.winner == 0 else f"{name_b} wins" if r.winner == 1 else "draw"
                side = f"{name_a} as team {'B' if swap else 'A'}"
                print(f"seed={seed} {side}: {label} {r.model_a_score * 100:.0f}-{r.model_b_score * 100:.0f} steps={r.episode_steps}", flush=True)
    finally:
        env.close()
    print(f"summary {name_a} vs {name_b}: {wins_a:.1f}/{len(margins)}, mean margin {np.mean(margins):+.1f}")


if __name__ == "__main__":
    main()
