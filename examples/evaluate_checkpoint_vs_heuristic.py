"""Watch a trained QMIX checkpoint play against heuristics, with the Unity window visible.

Every seed is played from both physical sides against every opponent.  With ``--early-stop`` a
game ends as soon as no battery is left anywhere (on tiles, storages included, or held by a unit):
batteries never respawn and only batteries score, so the score is frozen and the leader wins --
the same rule as ``tournament_adaptive.py``.

Usage:
    python -m examples.evaluate_checkpoint_vs_heuristic \\
        --checkpoint checkpoints/offline/mixv2_20260913/step_20000.pt \\
        --opponents strategic_v1 strategic_v4 strategic_v17 --early-stop
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import POLICY_REGISTRY
from blackout_env.model.my_policy import MyPolicy
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

BATTERY_CHANNEL = 8     # graphic: battery count per tile, storages included
HELD_BATTERY_COLUMN = 4  # agent_states: holding_item one-hot slot 1 = battery count


def play(env: BlackOutEnv, candidate, opponent, swapped: bool, seed: int, early_stop: bool) -> dict:
    """One game from the candidate's point of view; winner 0 = candidate, 1 = opponent, None = draw."""
    candidate_names = set(team_b_agents() if swapped else team_a_agents())
    candidate_team = 1 if swapped else 0
    obs, _ = env.reset(seed=seed)
    steps, scores, winner, decided_early = 0, (0.0, 0.0), None, False
    while env.agents:
        infos = None
        if not obs:
            obs, _, _, _, infos = env.step({})
        else:
            first = next(iter(obs))
            team_state = obs[first]["team_state"]
            if int(first.split("_")[1]) < 5:
                scores = (float(team_state[0]), float(team_state[1]))
            else:
                scores = (float(team_state[1]), float(team_state[0]))
            if (early_stop and not obs[first]["graphic"][:, :, BATTERY_CHANNEL].any()
                    and not obs[first]["agent_states"][:, HELD_BATTERY_COLUMN].any()):
                decided_early = True
                break
            actions = {}
            candidate_obs = {n: o for n, o in obs.items() if n in candidate_names}
            opponent_obs = {n: o for n, o in obs.items() if n not in candidate_names}
            if candidate_obs:
                actions.update(candidate.act(candidate_obs))
            if opponent_obs:
                actions.update(opponent.act(opponent_obs))
            obs, _, _, _, infos = env.step(actions)
        steps += 1
        if infos and not env.agents:
            info = next(iter(infos.values()))
            scores = (float(info.get("score_0", scores[0])), float(info.get("score_1", scores[1])))
            physical = info.get("winner")
            winner = None if physical in (None, -1) else 0 if int(physical) == candidate_team else 1
    own, opp = round(scores[candidate_team] * 100), round(scores[1 - candidate_team] * 100)
    if decided_early:
        winner = 0 if own > opp else 1 if opp > own else None
    return {"winner": winner, "own": own, "opp": opp, "steps": steps, "decided_early": decided_early}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="offline_pretrain.py / qmix_trainer.py checkpoint (.pt)")
    parser.add_argument("--opponents", nargs="+", default=["strategic_v4"], choices=sorted(POLICY_REGISTRY))
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--time-scale", type=float, default=1.0, help="1.0 = real-time, matches watching a GUI match")
    parser.add_argument("--early-stop", action="store_true", help="end a game once no battery is left anywhere")
    args = parser.parse_args()

    # buffer_capacity=1 -- only need this trainer for its net architecture + load(), no replay use.
    config = QMIXConfig(buffer_capacity=1, device=args.device, tb_log_dir=None)
    trainer = QMIXTrainer(env=None, config=config)
    trainer.load(args.checkpoint)
    trainer.net.eval()
    print(f"[eval] loaded {args.checkpoint} (train_step_count={trainer.train_step_count})")

    candidate = MyPolicy(trainer.net, device=args.device)

    env = BlackOutEnv(str(args.build), time_scale=args.time_scale, no_graphics=False, unity_shaping=False)
    results: dict[str, list[dict]] = defaultdict(list)
    try:
        for opponent_id in args.opponents:
            for seed in args.seeds:
                for swapped in (False, True):
                    opponent = POLICY_REGISTRY[opponent_id]()
                    r = play(env, candidate, opponent, swapped, seed, args.early_stop)
                    results[opponent_id].append(r)
                    label = "W" if r["winner"] == 0 else "L" if r["winner"] == 1 else "D"
                    early = " early" if r["decided_early"] else ""
                    print(f"vs {opponent_id} seed={seed} swap={swapped} {label} "
                          f"score={r['own']}-{r['opp']} steps={r['steps']}{early}", flush=True)
    finally:
        env.close()

    print("\nsummary (candidate perspective):")
    for opponent_id, games in results.items():
        wins = sum(g["winner"] == 0 for g in games)
        losses = sum(g["winner"] == 1 for g in games)
        draws = len(games) - wins - losses
        margin = np.mean([g["own"] - g["opp"] for g in games])
        print(f"  vs {opponent_id:16s} W-L-D={wins}-{losses}-{draws}  "
              f"win_rate={(wins + 0.5 * draws) / len(games):.2f}  mean_margin={margin:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
