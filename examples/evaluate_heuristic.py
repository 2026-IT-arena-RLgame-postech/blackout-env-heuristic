"""Evaluate the strategic heuristic against random or itself on paired fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from blackout_env import BlackOutEnv, run_match
from blackout_env.heuristics import StrategicHeuristic
from blackout_env.model.base import BaseModel


class RandomPolicy(BaseModel):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return {name: self.rng.uniform(-1, 1, size=2).astype(np.float32) for name in obs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--opponent", choices=("random", "self"), default="random")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--graphics", action="store_true", help="Show the Unity game window")
    args = parser.parse_args()

    candidate = StrategicHeuristic()
    opponent: BaseModel = RandomPolicy(7) if args.opponent == "random" else StrategicHeuristic()
    env = BlackOutEnv(
        str(args.build),
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
    )
    results = []
    try:
        for seed in args.seeds:
            # Paired maps remove side/map luck: candidate plays the same seed on both sides.
            for swap in (False, True):
                candidate.reset()
                if isinstance(opponent, StrategicHeuristic):
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
    return 0 if wins >= losses else 1


if __name__ == "__main__":
    sys.exit(main())
