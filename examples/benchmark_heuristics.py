"""Paired, side-swapped heuristic benchmark with movement-failure diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import (
    StrategicHeuristicV1, StrategicHeuristicV2, StrategicHeuristicV3,
    StrategicHeuristicV4,
)

VERSIONS = {
    "v2": StrategicHeuristicV2,
    "v3": StrategicHeuristicV3,
    "v4": StrategicHeuristicV4,
}


@dataclass
class FailureRuns:
    idle: list[int] = field(default_factory=list)
    blocked: list[int] = field(default_factory=list)
    unit_ticks: int = 0

    @staticmethod
    def _summarize(runs: list[int], threshold: int, unit_ticks: int) -> dict[str, float | int]:
        incidents = [length for length in runs if length >= threshold]
        return {
            "incidents": len(incidents),
            "per_1000_unit_ticks": 1000.0 * len(incidents) / max(1, unit_ticks),
            "worst_ticks": max(runs, default=0),
            "incident_ticks": sum(incidents),
        }

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            "idle_6s": self._summarize(self.idle, 300, self.unit_ticks),
            "blocked_0.24s": self._summarize(self.blocked, 12, self.unit_ticks),
        }


class MovementMonitor:
    """Track consecutive actionless and commanded-but-motionless unit ticks."""

    def __init__(self):
        self.result = FailureRuns()
        self._idle = {}
        self._blocked = {}

    def observe(self, names, before, after, actions):
        for name in names:
            if name not in actions:
                continue
            row = int(name.split("_")[1])
            action_norm = float(np.linalg.norm(actions[name]))
            movement = float(np.linalg.norm(after[row, :2] - before[row, :2]))
            # Respawn/teleport and class/cargo transitions delimit a run rather than count as
            # successful navigation; this prevents unrelated episodes being joined together.
            transition = movement > 0.15 or not np.array_equal(before[row, 3:], after[row, 3:])
            self.result.unit_ticks += 1
            self._update(name, "idle", action_norm <= 0.05 and movement <= 2e-4 and not transition)
            self._update(name, "blocked", action_norm >= 0.35 and movement <= 2e-4 and not transition)

    def _update(self, name, kind, active):
        current = self._idle if kind == "idle" else self._blocked
        runs = self.result.idle if kind == "idle" else self.result.blocked
        if active:
            current[name] = current.get(name, 0) + 1
        elif current.get(name, 0):
            runs.append(current.pop(name))

    def finish(self):
        self.result.idle.extend(self._idle.values())
        self.result.blocked.extend(self._blocked.values())
        self._idle.clear()
        self._blocked.clear()


@dataclass
class Game:
    seed: int
    swapped: bool
    winner: int | None
    candidate_score: float
    baseline_score: float
    steps: int
    candidate_failures: FailureRuns
    baseline_failures: FailureRuns


def play(env, seed: int, swapped: bool, candidate_type) -> Game:
    candidate, baseline = candidate_type(), StrategicHeuristicV1()
    physical_a, physical_b = set(team_a_agents()), set(team_b_agents())
    candidate_names = physical_b if swapped else physical_a
    baseline_names = physical_a if swapped else physical_b
    candidate_team = 1 if swapped else 0
    obs, _ = env.reset(seed=seed)
    candidate_monitor, baseline_monitor = MovementMonitor(), MovementMonitor()
    steps, final_info = 0, {}
    last_scores = (0.0, 0.0)

    while env.agents:
        candidate_obs = {n: obs[n] for n in env.agents if n in candidate_names and n in obs}
        baseline_obs = {n: obs[n] for n in env.agents if n in baseline_names and n in obs}
        actions = {}
        candidate_actions = candidate.act(candidate_obs) if candidate_obs else {}
        baseline_actions = baseline.act(baseline_obs) if baseline_obs else {}
        actions.update(candidate_actions)
        actions.update(baseline_actions)
        before = next(iter(obs.values()))["agent_states"].copy()
        next_obs, _, _, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))
            last_scores = (float(final_info.get("score_0", last_scores[0])),
                           float(final_info.get("score_1", last_scores[1])))
        if next_obs:
            after = next(iter(next_obs.values()))["agent_states"]
            candidate_monitor.observe(candidate_names, before, after, candidate_actions)
            baseline_monitor.observe(baseline_names, before, after, baseline_actions)
        obs = next_obs
        steps += 1

    candidate_monitor.finish()
    baseline_monitor.finish()
    physical_winner = final_info.get("winner")
    if physical_winner in (-1, None):
        winner = None
    else:
        winner = 0 if int(physical_winner) == candidate_team else 1
    candidate_score = last_scores[candidate_team]
    baseline_score = last_scores[1 - candidate_team]
    return Game(seed, swapped, winner, candidate_score, baseline_score, steps,
                candidate_monitor.result, baseline_monitor.result)


def aggregate(games: list[Game], attr: str) -> dict[str, dict[str, float | int]]:
    merged = FailureRuns()
    for game in games:
        result = getattr(game, attr)
        merged.idle.extend(result.idle)
        merged.blocked.extend(result.blocked)
        merged.unit_ticks += result.unit_ticks
    return merged.summary()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-rng", type=int, default=20260912)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument("--candidate", choices=tuple(VERSIONS), default="v4")
    args = parser.parse_args()
    if args.n_seeds < 5 and not args.seeds:
        parser.error("--n-seeds must be at least 5 for a promotion benchmark")
    seeds = args.seeds or np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.n_seeds, replace=False
    ).tolist()
    if len(seeds) < 5:
        parser.error("provide at least 5 seeds")
    print(f"seeds={seeds}", flush=True)

    env = BlackOutEnv(str(args.build), time_scale=args.time_scale,
                      no_graphics=not args.graphics)
    games = []
    try:
        for seed in seeds:
            for swapped in (False, True):
                game = play(env, int(seed), swapped, VERSIONS[args.candidate])
                games.append(game)
                label = "W" if game.winner == 0 else "L" if game.winner == 1 else "D"
                print(f"seed={seed} swapped={swapped} {args.candidate.upper()}={label} "
                      f"score={game.candidate_score*100:.0f}-{game.baseline_score*100:.0f} "
                      f"steps={game.steps}", flush=True)
    finally:
        env.close()

    wins = sum(g.winner == 0 for g in games)
    losses = sum(g.winner == 1 for g in games)
    draws = len(games) - wins - losses
    margins = np.asarray([(g.candidate_score - g.baseline_score) * 100 for g in games])
    paired = [float(np.mean(margins[i:i + 2])) for i in range(0, len(margins), 2)]
    print(f"outcome {args.candidate.upper()} W-L-D={wins}-{losses}-{draws} mean_margin={margins.mean():.2f} "
          f"paired_margins={[round(x, 2) for x in paired]}")
    print(f"{args.candidate.upper()} reliability={aggregate(games, 'candidate_failures')}")
    print(f"V1 reliability={aggregate(games, 'baseline_failures')}")
    return 0 if wins > losses and margins.mean() > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
