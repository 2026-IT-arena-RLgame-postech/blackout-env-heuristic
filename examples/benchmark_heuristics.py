"""Paired, side-swapped heuristic benchmark with movement-failure diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.train.movement_monitor import FailureRuns, MovementMonitor
from blackout_env.train.movement_monitor import aggregate as aggregate_failure_runs
from blackout_env.heuristics import (
    StrategicHeuristicV1, StrategicHeuristicV2, StrategicHeuristicV3,
    StrategicHeuristicV4,
    StrategicHeuristicV5,
    StrategicHeuristicV6,
    V4PolicyFamily,
    StrategicHeuristicV7,
    StrategicHeuristicV8,
    StrategicHeuristicV9,
    StrategicHeuristicV10,
    StrategicHeuristicV11,
    StrategicHeuristicV12,
)

VERSIONS = {
    "v2": StrategicHeuristicV2,
    "v3": StrategicHeuristicV3,
    "v4": StrategicHeuristicV4,
    "v5": StrategicHeuristicV5,
    "v6": StrategicHeuristicV6,
    "v4-near": V4PolicyFamily,
    "v7": StrategicHeuristicV7,
    "v8": StrategicHeuristicV8,
    "v9": StrategicHeuristicV9,
    "v10": StrategicHeuristicV10,
    "v11": StrategicHeuristicV11,
    "v12": StrategicHeuristicV12,
}
BASELINES = {"v1": StrategicHeuristicV1, **VERSIONS}


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
    candidate_variant: str = ""
    respec_attempts: int = 0
    respec_completions: int = 0
    respec_diagnostics: dict[str, int] = field(default_factory=dict)
    candidate_mode_ticks: dict[str, int] = field(default_factory=dict)
    strategy_transitions: list[tuple[int, str]] = field(default_factory=list)


def play(env, seed: int, swapped: bool, candidate_type, baseline_type) -> Game:
    # V4PolicyFamily samples an episode-coherent near-V4 profile from its seed.  Both sides
    # need the same construction rule so round-robin tournaments can place it in either axis.
    candidate = candidate_type(seed=seed) if candidate_type is V4PolicyFamily else candidate_type()
    baseline = baseline_type(seed=seed + 1_000_003) if baseline_type is V4PolicyFamily else baseline_type()
    physical_a, physical_b = set(team_a_agents()), set(team_b_agents())
    candidate_names = physical_b if swapped else physical_a
    baseline_names = physical_a if swapped else physical_b
    candidate_team = 1 if swapped else 0
    obs, _ = env.reset(seed=seed)
    candidate_monitor, baseline_monitor = MovementMonitor(), MovementMonitor()
    steps, final_info = 0, {}
    last_scores = (0.0, 0.0)
    empty_obs_steps = 0
    pending_transition = None
    candidate_mode_ticks: Counter[str] = Counter()

    while env.agents:
        if not obs:
            # Unity can expose one transition boundary with live agent names but no decision
            # observations (especially when multiple workers are starting/stopping nearby).
            # Advance with zero actions instead of indexing an empty observation dictionary.
            obs, _, _, _, infos = env.step({})
            if obs and pending_transition is not None:
                before, candidate_actions, baseline_actions = pending_transition
                after = next(iter(obs.values()))["agent_states"]
                candidate_monitor.observe(candidate_names, before, after, candidate_actions)
                baseline_monitor.observe(baseline_names, before, after, baseline_actions)
                pending_transition = None
            empty_obs_steps += 1
            steps += 1
            if infos:
                final_info = next(iter(infos.values()))
            if empty_obs_steps > 200:
                raise RuntimeError("Unity returned empty observations for over 200 steps")
            continue
        empty_obs_steps = 0
        candidate_obs = {n: obs[n] for n in env.agents if n in candidate_names and n in obs}
        baseline_obs = {n: obs[n] for n in env.agents if n in baseline_names and n in obs}
        actions = {}
        candidate_actions = candidate.act(candidate_obs) if candidate_obs else {}
        mode = getattr(candidate, "current_mode", None)
        if mode is not None and candidate_obs:
            candidate_mode_ticks[str(mode)] += 1
        baseline_actions = baseline.act(baseline_obs) if baseline_obs else {}
        actions.update(candidate_actions)
        actions.update(baseline_actions)
        before = next(iter(obs.values()))["agent_states"].copy()
        # Preserve the last public score even when Unity's terminal info omits score keys.
        public_team_state = obs[next(iter(obs))]["team_state"]
        first_name = next(iter(obs))
        if int(first_name.split("_")[1]) < 5:
            last_scores = (float(public_team_state[0]), float(public_team_state[1]))
        else:
            last_scores = (float(public_team_state[1]), float(public_team_state[0]))
        next_obs, _, _, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))
            last_scores = (float(final_info.get("score_0", last_scores[0])),
                           float(final_info.get("score_1", last_scores[1])))
        if next_obs:
            after = next(iter(next_obs.values()))["agent_states"]
            candidate_monitor.observe(candidate_names, before, after, candidate_actions)
            baseline_monitor.observe(baseline_names, before, after, baseline_actions)
        else:
            pending_transition = (before, candidate_actions, baseline_actions)
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
    variant = ""
    if isinstance(candidate, V4PolicyFamily) and candidate.current_sample is not None:
        variant = candidate.current_sample.profile
    return Game(
        seed, swapped, winner, candidate_score, baseline_score, steps,
        candidate_monitor.result, baseline_monitor.result, variant,
        int(getattr(candidate, "respec_attempts", 0)),
        int(getattr(candidate, "respec_completions", 0)),
        dict(getattr(candidate, "respec_diagnostics", {})),
        dict(candidate_mode_ticks),
        list(getattr(candidate, "strategy_transitions", [])),
    )


def aggregate(games: list[Game], attr: str) -> dict[str, dict[str, float | int]]:
    return aggregate_failure_runs([getattr(game, attr) for game in games])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-rng", type=int, default=20260912)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument("--candidate", choices=tuple(VERSIONS), default="v4")
    parser.add_argument("--baseline", choices=tuple(BASELINES), default="v1")
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
                      no_graphics=not args.graphics,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    games = []
    try:
        for seed in seeds:
            for swapped in (False, True):
                game = play(
                    env, int(seed), swapped, VERSIONS[args.candidate], BASELINES[args.baseline]
                )
                games.append(game)
                label = "W" if game.winner == 0 else "L" if game.winner == 1 else "D"
                print(f"seed={seed} swapped={swapped} {args.candidate.upper()}={label} "
                      f"score={game.candidate_score*100:.0f}-{game.baseline_score*100:.0f} "
                      f"steps={game.steps} variant={game.candidate_variant or '-'} "
                      f"respec={game.respec_completions}/{game.respec_attempts} "
                      f"gates={game.respec_diagnostics.get('all_gates_ticks', 0)} "
                      f"modes={game.candidate_mode_ticks or '-'} "
                      f"switches={len(game.strategy_transitions)}", flush=True)
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
    print(f"{args.baseline.upper()} reliability={aggregate(games, 'baseline_failures')}")
    attempts = sum(game.respec_attempts for game in games)
    completions = sum(game.respec_completions for game in games)
    if attempts:
        print(f"respec lifecycle attempts={attempts} completions={completions} "
              f"rate={completions / attempts:.3f}")
    diagnostics = [game.respec_diagnostics for game in games if game.respec_diagnostics]
    if diagnostics:
        print("respec diagnostics "
              f"max_inactive={max(d['max_inactive_ticks'] for d in diagnostics)} "
              f"trailing_ticks={sum(d['trailing_ticks'] for d in diagnostics)} "
              f"both_hunters_ticks={sum(d['both_hunters_ticks'] for d in diagnostics)} "
              f"all_gates_ticks={sum(d['all_gates_ticks'] for d in diagnostics)}")
    mode_ticks: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for game in games:
        mode_ticks.update(game.candidate_mode_ticks)
        transitions.update(mode for _, mode in game.strategy_transitions)
    if mode_ticks:
        total_mode_ticks = sum(mode_ticks.values())
        mode_share = {
            mode: round(ticks / total_mode_ticks, 3)
            for mode, ticks in sorted(mode_ticks.items())
        }
        print("strategy diversity "
              f"mode_ticks={dict(sorted(mode_ticks.items()))} "
              f"mode_share={mode_share} "
              f"transitions={dict(sorted(transitions.items()))}")
    return 0 if wins > losses and margins.mean() > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
