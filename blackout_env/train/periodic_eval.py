"""
Periodic heuristic-match eval hook for offline_pretrain.py: every --eval-interval train steps,
plays a few short matches between the current checkpoint and a heuristic opponent and reports
win/loss/margin plus the same idle/blocked movement diagnostics
examples/benchmark_heuristics.py's heuristic-reliability tournaments use (see
blackout_env/train/movement_monitor.py) -- so a run collapsing into a degenerate policy (e.g.
the graphic_encoder gradient-vanishing issue that produced a "walks into walls, barely moves"
agent -- see docs/offline_pretrain_runs.md) shows up as a spike in blocked/idle incidents within
a few eval windows, instead of only being caught by a human watching a GUI match after the full
(many-hour) run finishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.model.base import BaseModel
from blackout_env.train.movement_monitor import FailureRuns, MovementMonitor, aggregate


@dataclass
class EvalMatch:
    winner: int | None  # 0 = candidate, 1 = opponent, None = draw
    candidate_score: float
    opponent_score: float
    steps: int
    candidate_failures: FailureRuns
    opponent_failures: FailureRuns


def play_eval_match(
    env: BlackOutEnv, candidate: BaseModel, opponent: BaseModel, swap_teams: bool, seed: int
) -> EvalMatch:
    """
    Same match loop as blackout_env.competition.match.run_match (see its own docstring), plus
    MovementMonitor tracking for both sides and a hard stop on a stuck Unity connection --
    run_match itself stays lean for the common (non-diagnostic) case, this is the diagnostic
    variant used only by the periodic in-training eval below.
    """
    team_a_names = set(team_a_agents())
    team_b_names = set(team_b_agents())

    if not swap_teams:
        controller = {name: candidate for name in team_a_names}
        controller.update({name: opponent for name in team_b_names})
        candidate_team = 0
    else:
        controller = {name: opponent for name in team_a_names}
        controller.update({name: candidate for name in team_b_names})
        candidate_team = 1
    candidate_names = team_a_names if candidate_team == 0 else team_b_names
    opponent_names = team_b_names if candidate_team == 0 else team_a_names

    obs, _ = env.reset(seed=seed)
    candidate_monitor, opponent_monitor = MovementMonitor(), MovementMonitor()
    steps = 0
    final_info: dict = {}
    empty_obs_steps = 0

    while env.agents:
        if not obs:
            # See benchmark_heuristics.py's play(): Unity can expose one transition boundary
            # with live agent names but no decision observations. Advance with zero actions
            # instead of indexing an empty dict.
            obs, _, _, _, infos = env.step({})
            if infos:
                final_info = next(iter(infos.values()))
            empty_obs_steps += 1
            steps += 1
            if empty_obs_steps > 200:
                raise RuntimeError("Unity returned empty observations for over 200 steps")
            continue
        empty_obs_steps = 0

        candidate_obs = {a: obs[a] for a in env.agents if controller[a] is candidate and a in obs}
        opponent_obs = {a: obs[a] for a in env.agents if controller[a] is opponent and a in obs}
        candidate_actions = candidate.act(candidate_obs) if candidate_obs else {}
        opponent_actions = opponent.act(opponent_obs) if opponent_obs else {}
        actions = {**candidate_actions, **opponent_actions}

        before = next(iter(obs.values()))["agent_states"].copy()
        next_obs, _, _, _, infos = env.step(actions)
        if infos:
            final_info = next(iter(infos.values()))
        if next_obs:
            after = next(iter(next_obs.values()))["agent_states"]
            candidate_monitor.observe(candidate_names, before, after, candidate_actions)
            opponent_monitor.observe(opponent_names, before, after, opponent_actions)
        obs = next_obs
        steps += 1

    candidate_monitor.finish()
    opponent_monitor.finish()

    physical_winner = final_info.get("winner")
    winner = None if physical_winner in (-1, None) else (0 if int(physical_winner) == candidate_team else 1)

    score_0 = float(final_info.get("score_0", 0.0))
    score_1 = float(final_info.get("score_1", 0.0))
    candidate_score, opponent_score = (score_0, score_1) if candidate_team == 0 else (score_1, score_0)

    return EvalMatch(
        winner=winner,
        candidate_score=candidate_score,
        opponent_score=opponent_score,
        steps=steps,
        candidate_failures=candidate_monitor.result,
        opponent_failures=opponent_monitor.result,
    )


def run_periodic_eval(
    env: BlackOutEnv, candidate: BaseModel, opponent: BaseModel, seeds: list[int]
) -> dict[str, float]:
    """
    Plays len(seeds)*2 matches (each seed both non-swapped and swapped, for side fairness)
    between `candidate` and `opponent`. Returns a flat dict of TensorBoard-loggable scalars --
    see offline_pretrain.py's call site for the "eval/" tag prefix these get logged under.
    """
    matches = []
    for seed in seeds:
        for swap in (False, True):
            if hasattr(opponent, "reset"):
                opponent.reset()
            matches.append(play_eval_match(env, candidate, opponent, swap_teams=swap, seed=seed))

    wins = sum(m.winner == 0 for m in matches)
    losses = sum(m.winner == 1 for m in matches)
    draws = len(matches) - wins - losses
    margins = [(m.candidate_score - m.opponent_score) * 100 for m in matches]
    candidate_reliability = aggregate([m.candidate_failures for m in matches])
    opponent_reliability = aggregate([m.opponent_failures for m in matches])

    return {
        "win_rate": wins / len(matches),
        "loss_rate": losses / len(matches),
        "draw_rate": draws / len(matches),
        "mean_margin": float(np.mean(margins)),
        "mean_steps": float(np.mean([m.steps for m in matches])),
        "candidate_idle_per_1000_ticks": candidate_reliability["idle_6s"]["per_1000_unit_ticks"],
        "candidate_blocked_per_1000_ticks": candidate_reliability["blocked_0.24s"]["per_1000_unit_ticks"],
        "opponent_idle_per_1000_ticks": opponent_reliability["idle_6s"]["per_1000_unit_ticks"],
        "opponent_blocked_per_1000_ticks": opponent_reliability["blocked_0.24s"]["per_1000_unit_ticks"],
    }
