"""Any two players -- heuristics or checkpoints -- against each other, with the Unity window visible.

A player is a heuristic id (``strategic_v4``) or ``label=path.pt`` for a checkpoint.  ``--a`` plays
every ``--b`` in turn; every seed is played from both physical sides.  ``--early-stop`` ends a game
once no battery is left anywhere (same rule as ``tournament_adaptive.py``).

Usage:
    python examples/gui_match.py --a strategic_v17 --b strategic_v4 strategic_v12 --early-stop
    python examples/gui_match.py --a run11=checkpoints/offline/<run>/final.pt \\
        --b run16=checkpoints/offline/<run>/final.pt --early-stop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from evaluate_checkpoint_vs_heuristic import play
except ImportError:
    from examples.evaluate_checkpoint_vs_heuristic import play

from blackout_env import BlackOutEnv
from blackout_env.heuristics import POLICY_REGISTRY
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer


class Player:
    """Heuristics are rebuilt per game (they keep per-match state); checkpoints are loaded once."""

    def __init__(self, spec: str, device: str):
        if spec in POLICY_REGISTRY:
            self.label, self._policy = spec, None
            return
        label, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"unknown player {spec!r}: use a heuristic id or label=checkpoint.pt")
        trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=device, tb_log_dir=None))
        trainer.load(Path(path))
        trainer.net.eval()
        print(f"[eval] {label}: {path} (train_step_count={trainer.train_step_count})")
        self.label, self._policy = label, MyPolicy(trainer.net, device=device)

    def for_game(self):
        return self._policy if self._policy is not None else POLICY_REGISTRY[self.label]()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="heuristic id or label=checkpoint.pt")
    parser.add_argument("--b", nargs="+", required=True, help="one or more opponents, same format")
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--early-stop", action="store_true", help="end a game once no battery is left anywhere")
    args = parser.parse_args()

    player_a = Player(args.a, args.device)
    opponents = [Player(spec, args.device) for spec in args.b]

    env = BlackOutEnv(str(args.build), time_scale=args.time_scale, no_graphics=args.headless, unity_shaping=False)
    results: dict[str, list[dict]] = {}
    try:
        for player_b in opponents:
            games = results.setdefault(player_b.label, [])
            for seed in args.seeds:
                for swapped in (False, True):
                    r = play(env, player_a.for_game(), player_b.for_game(), swapped, seed, args.early_stop)
                    games.append(r)
                    label = f"{player_a.label} wins" if r["winner"] == 0 else \
                        f"{player_b.label} wins" if r["winner"] == 1 else "draw"
                    early = " early" if r["decided_early"] else ""
                    print(f"{player_a.label} vs {player_b.label} seed={seed} {player_a.label} as team "
                          f"{'B' if swapped else 'A'}: {label} {r['own']}-{r['opp']} steps={r['steps']}{early}",
                          flush=True)
    finally:
        env.close()

    print(f"\nsummary ({player_a.label} perspective):")
    for opponent_label, games in results.items():
        wins = sum(g["winner"] == 0 for g in games)
        losses = sum(g["winner"] == 1 for g in games)
        draws = len(games) - wins - losses
        margin = np.mean([g["own"] - g["opp"] for g in games])
        print(f"  vs {opponent_label:16s} W-L-D={wins}-{losses}-{draws}  "
              f"win_rate={(wins + 0.5 * draws) / len(games):.2f}  mean_margin={margin:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
