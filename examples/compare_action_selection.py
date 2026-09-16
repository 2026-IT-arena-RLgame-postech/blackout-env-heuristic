"""
Same checkpoint, different rules for turning its Q-table into an action.

Run 6's units spent ~25% of their unit-ticks pressed against a wall: a deterministic greedy
policy that picks a wall-ward direction observes almost no state change and picks it again, for
up to 381 ticks in a row. This script measures whether that loop is actually what costs the
score, by keeping the network fixed and only changing the selection rule -- no retraining.

The answer for Run 6 was no: masking removed the blocking outright (24.7% -> 0.6% of unit-ticks)
and the score margin did not move outside sample noise across 50 matches. See
docs/run6_diagnosis_20260916.md §3. Worth re-running for any checkpoint before spending a run on
blocking-related reward changes.

Modes: argmax, mask, eps<p>, boltz<T>, and mask+eps<p> / mask+boltz<T>.

Usage:
    python -m examples.compare_action_selection --checkpoint checkpoints/offline/<run>/final.pt
    python -m examples.compare_action_selection --checkpoint ... --modes argmax mask eps0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.heuristics import RecommendedStrategicHeuristic
from blackout_env.model.action_mask import world_legal_mask
from blackout_env.model.my_policy import DIRECTION_VECTORS, MyPolicy
from blackout_env.train.movement_monitor import MovementMonitor, aggregate
from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

BLOCKED_KEY = "blocked_0.24s"


class SelectionRule:
    """Turns MyPolicy's world-frame Q rows into actions under one named rule."""

    def __init__(self, mode: str, rng: np.random.Generator) -> None:
        self.mode = mode
        self.rng = rng
        self.mask = "mask" in mode
        self.epsilon = float(mode.split("eps")[1]) if "eps" in mode else 0.0
        self.temperature = float(mode.split("boltz")[1]) if "boltz" in mode else 0.0

    def __call__(self, policy: MyPolicy, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        q_rows = policy.action_values(obs)
        shared = obs[next(iter(obs))]
        legal = world_legal_mask(shared["graphic"], shared["agent_states"]) if self.mask else None

        actions = {}
        for agent, row in q_rows.items():
            values = row.astype(np.float64)
            if legal is not None:
                values = np.where(legal[unit_index(agent)], values, -np.inf)
            allowed = np.flatnonzero(np.isfinite(values))
            if self.temperature > 0:
                shifted = values - values[allowed].max()
                weights = np.zeros_like(values)
                weights[allowed] = np.exp(shifted[allowed] / self.temperature)
                index = int(self.rng.choice(len(values), p=weights / weights.sum()))
            elif self.epsilon > 0 and self.rng.random() < self.epsilon:
                index = int(self.rng.choice(allowed))
            else:
                index = int(np.argmax(values))
            actions[agent] = DIRECTION_VECTORS[index]
        return actions


def play(env, policy: MyPolicy, rule: SelectionRule, opponent, seed: int, swap: bool) -> dict[str, float]:
    team_a, team_b = set(team_a_agents()), set(team_b_agents())
    candidate_names = team_b if swap else team_a
    opponent_names = team_a if swap else team_b
    monitor = MovementMonitor()

    obs, _ = env.reset(seed=seed)
    steps = empty_steps = 0
    final_info: dict = {}
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            if infos:
                final_info = next(iter(infos.values()))
            empty_steps += 1
            steps += 1
            if empty_steps > 200:
                raise RuntimeError("Unity returned empty observations for over 200 steps")
            continue
        empty_steps = 0

        candidate_obs = {a: obs[a] for a in env.agents if a in candidate_names and a in obs}
        opponent_obs = {a: obs[a] for a in env.agents if a in opponent_names and a in obs}
        candidate_actions = rule(policy, candidate_obs) if candidate_obs else {}
        actions = {**candidate_actions, **(opponent.act(opponent_obs) if opponent_obs else {})}

        before = next(iter(obs.values()))["agent_states"].copy()
        next_obs, _, _, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))
        if next_obs:
            monitor.observe(candidate_names, before, next_obs[next(iter(next_obs))]["agent_states"], candidate_actions)
        obs = next_obs
        steps += 1
    monitor.finish()

    score_0 = float(final_info.get("score_0", 0.0)) * 100
    score_1 = float(final_info.get("score_1", 0.0)) * 100
    candidate, other = (score_1, score_0) if swap else (score_0, score_1)
    physical_winner = final_info.get("winner")
    candidate_team = 1 if swap else 0
    winner = None if physical_winner in (None, -1) else (0 if int(physical_winner) == candidate_team else 1)
    blocked = aggregate([monitor.result])[BLOCKED_KEY]
    return {
        "margin": candidate - other,
        "steps": steps,
        "winner": winner,
        "blocked_incident_frac": blocked["incident_ticks"] / max(1, monitor.result.unit_ticks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--modes", nargs="+", default=["argmax", "mask", "eps0.15", "boltz0.5", "mask+eps0.05"])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--time-scale", type=float, default=None)
    args = parser.parse_args()

    trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=args.device, tb_log_dir=None))
    trainer.load(args.checkpoint)
    trainer.net.eval()
    policy = MyPolicy(trainer.net, device=args.device)
    opponent = RecommendedStrategicHeuristic()

    time_scale = args.time_scale if args.time_scale is not None else (1.0 if args.gui else 20.0)
    env = BlackOutEnv(str(args.build), time_scale=time_scale, no_graphics=not args.gui)
    summary = []
    try:
        for mode in args.modes:
            rule = SelectionRule(mode, np.random.default_rng(0))
            results = []
            for seed in args.seeds:
                for swap in (False, True):
                    if hasattr(opponent, "reset"):
                        opponent.reset()
                    results.append(play(env, policy, rule, opponent, seed, swap))
            wins = sum(r["winner"] == 0 for r in results)
            losses = sum(r["winner"] == 1 for r in results)
            row = {
                "mode": mode,
                "record": f"{wins}-{losses}-{len(results) - wins - losses}",
                "margin": float(np.mean([r["margin"] for r in results])),
                "steps": float(np.mean([r["steps"] for r in results])),
                "blocked": float(np.mean([r["blocked_incident_frac"] for r in results])),
            }
            summary.append(row)
            print(f"[{mode}] W-L-D={row['record']} margin={row['margin']:.1f} steps={row['steps']:.0f} "
                  f"blocked={row['blocked']:.3f}", flush=True)
    finally:
        env.close()

    print(f"\n{'mode':16} {'W-L-D':>9} {'margin':>9} {'steps':>7} {'blocked':>9}")
    for row in summary:
        print(f"{row['mode']:16} {row['record']:>9} {row['margin']:9.1f} {row['steps']:7.0f} {row['blocked']:9.3f}")
    print("\nNote: margins swing by tens of points across 10-match samples -- re-run with fresh "
          "--seeds before reading a difference as real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
