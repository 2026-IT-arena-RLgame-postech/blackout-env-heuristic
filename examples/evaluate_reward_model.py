"""
Diagnose whether the Phase 1 / 1.5 potential-based reward shaping (reward_proposal.md
Sections 6, 14, 15) actually behaves as designed, using existing heuristic policies of
known relative skill as test subjects instead of a trained agent.

This does NOT re-benchmark heuristic play quality (see benchmark_heuristics.py for that).
It uses matches between heuristics purely as a source of realistic trajectories, and asks
three questions about the *reward signal* those trajectories produce:

1. Zero-sum (reward_proposal.md 11.1): is team-shaping reward Psi_k exactly zero-sum between
   teams, and does the (intentionally non-zero-sum) individual nav potential leak stay small
   relative to the team signal?
2. Signal quality: does cumulative shaped return (excluding the terminal +-1) actually track
   which team is winning -- i.e. is it a useful advantage estimate, not just noise?
3. Cycle-boundedness (11.2 / 15.5): does a unit that repeatedly approaches and retreats from
   the same item accumulate unbounded positive nav-shaping reward, or does it stay bounded as
   the telescoping-sum argument in the proposal predicts?

Isolating Psi-only vs nav-only contributions does not require any C# change: reward_config.json
is loaded fresh from StreamingAssets every time the Unity player process starts, so we just
rewrite that file and launch a new BlackOutEnv per variant.

Usage
-----
    cd /Users/mac/project/26rl/blackout-env
    ./.venv/bin/python examples/evaluate_reward_model.py --build build/mac/BlackOut.app
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import StrategicHeuristicV1, StrategicHeuristicV4
from blackout_env.model.base import BaseModel

BASE_CONFIG = {
    "itemRewards": [
        {"itemName": "Battery", "value": 0.0},
        {"itemName": "SpeedBuff", "value": 0.0},
    ],
    "killReward": 0.0,
    "deathPenalty": 0.0,
    "teamScoreReward": 0.0,
    "teamScorePenalty": 0.0,
    "potentialEta": 0.25,
    "potentialGamma": 0.99995,
    "potentialScale": 40.0,
    "hazardCoefficient": 0.05,
    "navPotentialEta": 0.08,
    "navPotentialScale": 12.0,
}

VARIANTS = {
    # As shipped: Psi (team) + Phi (individual nav) both active.
    "full": BASE_CONFIG,
    # Isolates Psi_k alone -- must be exactly zero-sum between teams by construction.
    "psi_only": {**BASE_CONFIG, "navPotentialEta": 0.0},
    # Isolates Phi_i alone -- deliberately NOT required to be zero-sum (15.4).
    "nav_only": {**BASE_CONFIG, "potentialEta": 0.0},
}


def write_config(build: Path, variant: str) -> None:
    path = build / "Contents/Resources/Data/StreamingAssets/reward_config.json"
    path.write_text(json.dumps(VARIANTS[variant], indent=4))


@dataclass
class StepRecord:
    team_a_reward: float
    team_b_reward: float
    score_a: float
    score_b: float


@dataclass
class EpisodeLog:
    steps: list[StepRecord] = field(default_factory=list)
    winner: int | None = None
    terminal_a: float = 0.0
    terminal_b: float = 0.0
    seed: int | None = None


def run_logged_match(
    env: BlackOutEnv,
    model_a: BaseModel,
    model_b: BaseModel,
    swap_teams: bool,
    seed: int,
) -> EpisodeLog:
    """Same protocol as competition.match.run_match, but records every step's per-team
    shaped reward and score instead of only the final totals."""
    team_a_names = set(team_a_agents())
    team_b_names = set(team_b_agents())
    if not swap_teams:
        controller = {n: model_a for n in team_a_names}
        controller.update({n: model_b for n in team_b_names})
    else:
        controller = {n: model_b for n in team_a_names}
        controller.update({n: model_a for n in team_b_names})

    obs, _ = env.reset(seed=seed)
    log = EpisodeLog(seed=seed)
    final_info: dict = {}

    while env.agents:
        a_obs = {a: obs[a] for a in env.agents if controller[a] is model_a and a in obs}
        b_obs = {a: obs[a] for a in env.agents if controller[a] is model_b and a in obs}
        actions: dict[str, np.ndarray] = {}
        if a_obs:
            actions.update(model_a.act(a_obs))
        if b_obs:
            actions.update(model_b.act(b_obs))

        obs, rewards, terminations, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))

        r_a = sum(rewards[a] for a in team_a_names)
        r_b = sum(rewards[a] for a in team_b_names)
        log.steps.append(StepRecord(
            team_a_reward=r_a, team_b_reward=r_b,
            score_a=float(final_info.get("score_0", 0.0)),
            score_b=float(final_info.get("score_1", 0.0)),
        ))

    physical_winner = final_info.get("winner")
    log.winner = None if physical_winner in (-1, None) else int(physical_winner)
    return log


def analyze(logs: list[EpisodeLog], candidate_is_team_a: list[bool]) -> dict:
    """Compute zero-sum residual, shaped-return/outcome correlation, and candidate-vs-baseline
    shaped-return comparison across a batch of logged episodes."""
    residuals = []
    cand_returns = []   # pre-terminal cumulative shaped return, candidate side
    base_returns = []
    cand_wins = []       # 1 if candidate won this episode, 0 lost, skipped if draw
    per_episode_corr = []

    for log, cand_a in zip(logs, candidate_is_team_a):
        a_cum = np.cumsum([s.team_a_reward for s in log.steps])
        b_cum = np.cumsum([s.team_b_reward for s in log.steps])
        step_sums = np.array([s.team_a_reward + s.team_b_reward for s in log.steps])
        residuals.extend(np.abs(step_sums).tolist())

        score_diff = np.array([s.score_a - s.score_b for s in log.steps], dtype=float)
        adv = a_cum - b_cum  # cumulative shaped-return advantage of team A
        if np.std(adv) > 1e-9 and np.std(score_diff) > 1e-9:
            per_episode_corr.append(float(np.corrcoef(adv, score_diff)[0, 1]))

        cand_final = a_cum[-1] if cand_a else b_cum[-1]
        base_final = b_cum[-1] if cand_a else a_cum[-1]
        cand_returns.append(cand_final)
        base_returns.append(base_final)

        if log.winner is not None:
            cand_team = 0 if cand_a else 1
            cand_wins.append(1 if log.winner == cand_team else 0)

    return {
        "n_episodes": len(logs),
        "zero_sum_mean_abs_residual": float(np.mean(residuals)) if residuals else float("nan"),
        "zero_sum_max_abs_residual": float(np.max(residuals)) if residuals else float("nan"),
        "mean_corr_adv_vs_scorediff": float(np.mean(per_episode_corr)) if per_episode_corr else float("nan"),
        "candidate_mean_shaped_return": float(np.mean(cand_returns)),
        "baseline_mean_shaped_return": float(np.mean(base_returns)),
        "candidate_win_rate": float(np.mean(cand_wins)) if cand_wins else float("nan"),
    }


class JitterPolicy(BaseModel):
    """Degenerate policy for the cycle-boundedness probe: unit_0 oscillates toward and away
    from whatever is nearest (an item, if one is close, otherwise just moves on a fixed
    to-and-fro axis). All other units stay still. Used only with the nav_only reward variant
    to isolate Phi_i's response to a repeating approach/retreat cycle."""

    def __init__(self, period: int = 60):
        self.period = period
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, obs):
        actions = {}
        for name in obs:
            phase = self.t % self.period
            sign = 1.0 if phase < self.period // 2 else -1.0
            actions[name] = np.array([sign, 0.0], dtype=np.float32) if name == "unit_0" else np.zeros(2, dtype=np.float32)
        self.t += 1
        return actions


def run_cycle_probe(build: Path, n_steps: int, time_scale: float) -> None:
    write_config(build, "nav_only")
    env = BlackOutEnv(str(build), time_scale=time_scale, no_graphics=True)
    try:
        policy = JitterPolicy()
        idle = StrategicHeuristicV1()  # opponent team just plays normally in the background
        obs, _ = env.reset(seed=999)
        cum = {a: 0.0 for a in env.possible_agents}
        history = []
        for step in range(n_steps):
            if not env.agents:
                obs, _ = env.reset(seed=999)
            team_a_obs = {a: obs[a] for a in env.agents if a in team_a_agents() and a in obs}
            team_b_obs = {a: obs[a] for a in env.agents if a in team_b_agents() and a in obs}
            actions = {}
            if team_a_obs:
                actions.update(policy.act(team_a_obs))
            if team_b_obs:
                actions.update(idle.act(team_b_obs))
            obs, rewards, _, _, _ = env.step(actions)
            for a, r in rewards.items():
                cum[a] += r
            if step % (policy.period * 4) == 0:
                history.append((step, cum["unit_0"]))
        print("\n[Cycle probe] unit_0 cumulative nav-shaping reward over repeated approach/retreat cycles:")
        for step, val in history:
            print(f"  step={step:5d}  cumulative={val:+.4f}")
        tail = [v for _, v in history[-5:]]
        print(f"  tail range over last {len(tail)} samples: {min(tail):+.4f} .. {max(tail):+.4f} "
              f"(spread={max(tail) - min(tail):.4f}) -- should stay bounded, not grow linearly with step count")
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--skip-cycle-probe", action="store_true")
    parser.add_argument("--cycle-steps", type=int, default=4000)
    args = parser.parse_args()

    build = args.build.resolve()
    backup = build / "Contents/Resources/Data/StreamingAssets/reward_config.json"
    original = backup.read_text()

    results = {}
    try:
        for variant in ("psi_only", "nav_only", "full"):
            write_config(build, variant)
            env = BlackOutEnv(str(build), time_scale=args.time_scale, no_graphics=True)
            try:
                logs, cand_is_a = [], []
                for seed in args.seeds:
                    for swap in (False, True):
                        v4 = StrategicHeuristicV4()
                        v1 = StrategicHeuristicV1()
                        # v4 = "candidate" (known stronger). swap=False -> v4 is team A.
                        model_a, model_b = (v4, v1) if not swap else (v1, v4)
                        log = run_logged_match(env, model_a, model_b, swap_teams=False, seed=seed)
                        logs.append(log)
                        cand_is_a.append(not swap)
                        label = "?" if log.winner is None else ("A" if log.winner == 0 else "B")
                        print(f"[{variant}] seed={seed} swap={swap} winner_team={label} "
                              f"steps={len(log.steps)} score={log.steps[-1].score_a:.0f}-{log.steps[-1].score_b:.0f}")
                results[variant] = analyze(logs, cand_is_a)
            finally:
                env.close()

        print("\n=== Reward model diagnostics: StrategicHeuristicV4 (candidate, known stronger) "
              "vs StrategicHeuristicV1 (baseline) ===")
        for variant, stats in results.items():
            print(f"\n-- variant={variant} (n={stats['n_episodes']} episodes) --")
            print(f"  zero-sum residual: mean={stats['zero_sum_mean_abs_residual']:.3e}  "
                  f"max={stats['zero_sum_max_abs_residual']:.3e}")
            print(f"  corr(cumulative shaped advantage, actual score diff): "
                  f"{stats['mean_corr_adv_vs_scorediff']:.3f}")
            print(f"  candidate (V4) mean pre-terminal shaped return: {stats['candidate_mean_shaped_return']:+.4f}")
            print(f"  baseline  (V1) mean pre-terminal shaped return: {stats['baseline_mean_shaped_return']:+.4f}")
            print(f"  candidate win rate this batch: {stats['candidate_win_rate']:.2f}")

        if not args.skip_cycle_probe:
            run_cycle_probe(build, args.cycle_steps, args.time_scale)
    finally:
        backup.write_text(original)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
