"""
Small reward post-processing helpers shared between the static offline dataset (loaded via
offline_dataset.load_dataset_into) and freshly-collected on-policy data (onpolicy_collect.py), so
both get the exact same treatment.
"""

from __future__ import annotations

import numpy as np


def blocked_penalty_adjustment(
    agent_states: np.ndarray,
    done: np.ndarray,
    team_indices: tuple[int, ...],
    penalty_per_unit: float,
) -> np.ndarray:
    """
    Returns a [T] array of <=0 adjustments to add to a team's stored per-tick reward: one entry
    per transition, `-penalty_per_unit` for each of `team_indices`' units that was "blocked" that
    tick (commanded movement, no actual displacement -- walking into a wall/obstacle). The last
    tick (no next state to compare against) and any tick immediately followed by an episode
    boundary (done[t]==True, so agent_states[t+1] belongs to a new episode) get 0.

    "blocked" mirrors movement_monitor.MovementMonitor's own definition MINUS the action_norm
    check: every action this codebase's agents ever take is one of MyPolicy.DIRECTION_VECTORS'
    8 unit-length compass directions (no "stay" action exists -- see my_policy.py), so
    action_norm is always exactly 1.0 and that half of the check is vacuous here.

    See diagnose_stopping2.py / diagnose_qvalues.py (docs/offline_pretrain_runs.md, Run4
    findings) for why this exists: 41.66% of the Run4 checkpoint's unit-ticks were "blocked" this
    way, with nothing in the actual game reward (reward_config.json / PotentialRewardCalculator.cs
    / IndividualNavPotentialCalculator.cs) penalizing it directly -- only failing to reward it,
    since nav-potential shaping is a pure function of grid-path distance to the current
    objective and doesn't change while a unit is stuck (no progress, but no penalty either).
    """
    n_ticks = agent_states.shape[0]
    adjustment = np.zeros(n_ticks, dtype=np.float32)
    if n_ticks < 2:
        return adjustment

    pos_t = agent_states[:-1, team_indices, :2]
    pos_t1 = agent_states[1:, team_indices, :2]
    movement = np.linalg.norm(pos_t1 - pos_t, axis=-1)  # [T-1, len(team_indices)]

    # Respawn/teleport and class/cargo transitions delimit a run rather than count as a movement
    # failure -- same exclusion movement_monitor.MovementMonitor.observe() applies.
    class_t = agent_states[:-1, team_indices, 3:]
    class_t1 = agent_states[1:, team_indices, 3:]
    transition = np.any(class_t != class_t1, axis=-1) | (movement > 0.15)

    blocked = (movement <= 2e-4) & ~transition
    blocked &= ~done[:-1, None]

    n_blocked = blocked.sum(axis=-1)  # [T-1]
    adjustment[:-1] = -penalty_per_unit * n_blocked
    return adjustment
