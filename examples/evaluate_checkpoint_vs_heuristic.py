"""Watch a trained QMIX checkpoint play against a heuristic, with the Unity window visible.

Plays each --seeds value twice (checkpoint as team A, then as team B) against V4
(RecommendedStrategicHeuristic, the same opponent periodic eval uses), printing each result and
a W-L-D / mean-margin summary from the checkpoint's side. Margins are in game points.

Always opens the Unity window (no headless flag); for many headless matches use
examples/elo_checkpoints.py or examples/measure_match_phases.py instead. The model runs on
--device (default cpu, which is the fast choice for single-match inference).

Flags: --checkpoint (required, offline_pretrain.py / qmix_trainer.py .pt), --build
(default build/mac/BlackOut.app), --seeds (default 101 202 303, the periodic-eval seeds),
--time-scale (default 1.0 = real time).

Used by: models/run11_step80k/run11_pipeline.sh gui [ckpt] (seeds 404 505 606, --time-scale 3).

Usage:
    python -m examples.evaluate_checkpoint_vs_heuristic \\
        --checkpoint models/run11_step80k/step_80000.pt --seeds 404 505 606 --time-scale 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from blackout_env import BlackOutEnv, run_match
from blackout_env.heuristics import RecommendedStrategicHeuristic
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="offline_pretrain.py / qmix_trainer.py checkpoint (.pt)")
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--time-scale", type=float, default=1.0, help="1.0 = real-time, matches watching a GUI match")
    args = parser.parse_args()

    # buffer_capacity=1 -- only need this trainer for its net architecture + load(), no replay use.
    config = QMIXConfig(buffer_capacity=1, device=args.device, tb_log_dir=None)
    trainer = QMIXTrainer(env=None, config=config)
    trainer.load(args.checkpoint)
    trainer.net.eval()
    print(f"[eval] loaded {args.checkpoint} (train_step_count={trainer.train_step_count})")

    candidate = MyPolicy(trainer.net, device=args.device)
    opponent = RecommendedStrategicHeuristic()

    env = BlackOutEnv(str(args.build), time_scale=args.time_scale, no_graphics=False, unity_shaping=False)
    results = []
    try:
        for seed in args.seeds:
            for swap in (False, True):
                if hasattr(opponent, "reset"):
                    opponent.reset()
                result = run_match(env, candidate, opponent, swap_teams=swap, seed=seed)
                results.append(result)
                label = "W" if result.winner == 0 else "L" if result.winner == 1 else "D"
                print(f"seed={seed} swap={swap} {label} score={result.model_a_score * 100:.0f}-{result.model_b_score * 100:.0f} steps={result.episode_steps}")
    finally:
        env.close()

    wins = sum(r.winner == 0 for r in results)
    losses = sum(r.winner == 1 for r in results)
    draws = len(results) - wins - losses
    margins = [(r.model_a_score - r.model_b_score) * 100 for r in results]
    print(f"summary W-L-D={wins}-{losses}-{draws}, mean_margin={np.mean(margins):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
