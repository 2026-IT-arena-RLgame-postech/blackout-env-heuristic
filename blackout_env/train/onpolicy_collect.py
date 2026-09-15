"""
On-policy data collection for offline_pretrain.py's periodic eval hook: plays additional
self-vs-heuristic and self-play matches with the CURRENT checkpoint and pushes their raw
transitions into buffer_a/buffer_b (the same SequentialReplayBuffer instances offline_pretrain.py
preloaded from the static heuristic dataset), applying reward_shaping.blocked_penalty_adjustment
as it goes.

Why this exists: diagnose_stopping2.py / diagnose_qvalues.py (docs/offline_pretrain_runs.md, Run4
findings) found the Run4 checkpoint spends 41.66% of unit-ticks "blocked", with overconfident-but-
wrong Q-values and zero self-correction (100% of blocked runs >=12 ticks kept picking the same
wrong direction the whole run) -- and reward_proposal.md / the actual C# reward code confirmed
nothing in the game's reward structure penalizes wall-collision itself. Two complementary fixes:

  1. reward_shaping.blocked_penalty_adjustment gives training data an explicit negative signal
     for this exact failure mode, applied uniformly to both the static dataset and this module's
     freshly-collected matches.
  2. Pure heuristic-vs-heuristic data never visits "stuck against a wall" states at all (the
     heuristic doesn't get stuck), so growing the training distribution with on-policy rollouts
     from the checkpoint itself -- self-vs-heuristic and self-play -- gives the net examples of
     (and reward signal for recovering from) exactly the states it actually reaches at inference
     time. SequentialReplayBuffer is a real ring buffer (FIFO eviction once full, see push()), so
     simply pushing new transitions into the SAME buffer_a/buffer_b the static dataset was loaded
     into naturally displaces the oldest (originally pure-heuristic) entries over time -- no
     separate multi-buffer weighted sampler needed. Injecting ~0.3x and ~0.1x of the dataset's
     original size in self-vs-heuristic / self-play transitions (spread evenly over
     --eval-interval windows) converges to roughly a 60/30/10 final mix by the end of a run sized
     the way the static dataset was.
"""

from __future__ import annotations

import numpy as np

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import TEAM_A_INDICES, TEAM_B_INDICES, team_a_agents, team_b_agents
from blackout_env.model.base import BaseModel
from blackout_env.model.my_policy import direction_vector_to_idx
from blackout_env.train.replay_buffer import SequentialReplayBuffer
from blackout_env.train.reward_shaping import blocked_penalty_adjustment

ABSORPTION_IDX = 3  # matches qmix_trainer.ABSORPTION_IDX -- team_state row layout

_FIELDS = ("graphic", "team_state", "agent_states", "actions", "reward", "done")


def play_and_collect(
    env: BlackOutEnv,
    team_a_policy: BaseModel,
    team_b_policy: BaseModel,
    seed: int,
    penalty_per_unit: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Plays one full match with team_a_policy/team_b_policy controlling their respective teams
    (pass the same object for both for self-play) and returns (transitions_a, transitions_b),
    each a dict of arrays matching SequentialReplayBuffer.push()'s per-field schema (one row per
    tick), reward already including the blocked-tick penalty.
    """
    team_a_names = team_a_agents()
    team_b_names = team_b_agents()

    rows_a: dict[str, list] = {f: [] for f in _FIELDS}
    rows_b: dict[str, list] = {f: [] for f in _FIELDS}

    obs, _ = env.reset(seed=seed)
    prev_absorption_time_left = float(obs[team_a_names[0]]["team_state"][ABSORPTION_IDX])
    if hasattr(team_a_policy, "reset"):
        team_a_policy.reset()
    if team_b_policy is not team_a_policy and hasattr(team_b_policy, "reset"):
        team_b_policy.reset()
    empty_obs_steps = 0

    while env.agents:
        if not obs:
            obs, _, _, _, _ = env.step({})
            empty_obs_steps += 1
            if empty_obs_steps > 200:
                raise RuntimeError("empty obs for 200+ steps")
            continue
        empty_obs_steps = 0

        obs_a = {a: obs[a] for a in team_a_names if a in obs}
        obs_b = {a: obs[a] for a in team_b_names if a in obs}
        actions_a = team_a_policy.act(obs_a) if obs_a else {}
        actions_b = team_b_policy.act(obs_b) if obs_b else {}
        env_actions = {**actions_a, **actions_b}

        if len(obs_a) < len(team_a_names) or len(obs_b) < len(team_b_names):
            # Partial decision window (see periodic_eval.play_eval_match's docstring for this
            # same lockstep edge case) -- advance without recording a training row for this tick
            # rather than guessing the missing units' chosen directions.
            obs, _, _, _, _ = env.step(env_actions)
            continue

        dir_a = direction_vector_to_idx(np.stack([actions_a[a] for a in team_a_names]))
        dir_b = direction_vector_to_idx(np.stack([actions_b[a] for a in team_b_names]))
        full_direction_idx = np.concatenate([dir_a, dir_b]).astype(np.int64)

        shared_a, shared_b = obs[team_a_names[0]], obs[team_b_names[0]]
        next_obs, rewards, terminations, _, _ = env.step(env_actions)
        done = any(terminations.values())

        absorption_fired = False
        if next_obs and team_a_names[0] in next_obs:
            absorption_time_left = float(next_obs[team_a_names[0]]["team_state"][ABSORPTION_IDX])
            absorption_fired = absorption_time_left > prev_absorption_time_left + 1e-6
            prev_absorption_time_left = absorption_time_left
        buffer_done = done or absorption_fired

        reward_a = sum(rewards.get(a, 0.0) for a in team_a_names)
        reward_b = sum(rewards.get(a, 0.0) for a in team_b_names)

        for f, v in zip(_FIELDS, (shared_a["graphic"], shared_a["team_state"], shared_a["agent_states"], full_direction_idx, reward_a, buffer_done)):
            rows_a[f].append(v)
        for f, v in zip(_FIELDS, (shared_b["graphic"], shared_b["team_state"], shared_b["agent_states"], full_direction_idx, reward_b, buffer_done)):
            rows_b[f].append(v)

        obs = next_obs

    def _finalize(rows: dict[str, list], team_indices: tuple[int, ...]) -> dict[str, np.ndarray]:
        agent_states = np.stack(rows["agent_states"])
        done_arr = np.array(rows["done"], dtype=np.bool_)
        reward_arr = np.array(rows["reward"], dtype=np.float32)
        reward_arr = reward_arr + blocked_penalty_adjustment(agent_states, done_arr, team_indices, penalty_per_unit)
        return dict(
            graphic=np.stack(rows["graphic"]),
            team_state=np.stack(rows["team_state"]),
            agent_states=agent_states,
            actions=np.stack(rows["actions"]),
            reward=reward_arr,
            done=done_arr,
        )

    return _finalize(rows_a, TEAM_A_INDICES), _finalize(rows_b, TEAM_B_INDICES)


def _push(buffer: SequentialReplayBuffer, transitions: dict[str, np.ndarray]) -> None:
    n = transitions["graphic"].shape[0]
    for i in range(n):
        buffer.push(
            transitions["graphic"][i],
            transitions["team_state"][i],
            transitions["agent_states"][i],
            transitions["actions"][i],
            float(transitions["reward"][i]),
            bool(transitions["done"][i]),
        )


def collect_onpolicy_data(
    env: BlackOutEnv,
    buffer_a: SequentialReplayBuffer,
    buffer_b: SequentialReplayBuffer,
    candidate: BaseModel,
    heuristic: BaseModel,
    target_ticks_self_vs_heuristic: int,
    target_ticks_self_play: int,
    penalty_per_unit: float,
    seed_start: int,
    max_matches: int = 50,
) -> dict[str, int]:
    """
    Plays self-vs-heuristic matches (candidate vs heuristic, side swapped every match for
    fairness) until >= target_ticks_self_vs_heuristic ticks are collected, then self-play matches
    (candidate vs itself) until >= target_ticks_self_play more, pushing every match's transitions
    into buffer_a/buffer_b as it goes (so a crash mid-window still keeps whatever was pushed so
    far). `seed_start` should differ every call (e.g. the current train step) so successive
    windows don't replay identical episodes. Stops early past `max_matches` per phase with a
    printed warning rather than looping forever if matches turn out shorter than expected.
    """
    stats = {
        "self_vs_heuristic_ticks": 0,
        "self_vs_heuristic_matches": 0,
        "self_play_ticks": 0,
        "self_play_matches": 0,
    }

    seed = seed_start
    swap = False
    while (
        stats["self_vs_heuristic_ticks"] < target_ticks_self_vs_heuristic
        and stats["self_vs_heuristic_matches"] < max_matches
    ):
        team_a_policy, team_b_policy = (heuristic, candidate) if swap else (candidate, heuristic)
        t_a, t_b = play_and_collect(env, team_a_policy, team_b_policy, seed, penalty_per_unit)
        _push(buffer_a, t_a)
        _push(buffer_b, t_b)
        stats["self_vs_heuristic_ticks"] += t_a["graphic"].shape[0]
        stats["self_vs_heuristic_matches"] += 1
        seed += 1
        swap = not swap
    if stats["self_vs_heuristic_ticks"] < target_ticks_self_vs_heuristic:
        print(
            f"[onpolicy] warning: only collected {stats['self_vs_heuristic_ticks']}/"
            f"{target_ticks_self_vs_heuristic} self-vs-heuristic ticks after "
            f"{stats['self_vs_heuristic_matches']} matches (max_matches={max_matches})"
        )

    while (
        stats["self_play_ticks"] < target_ticks_self_play
        and stats["self_play_matches"] < max_matches
    ):
        t_a, t_b = play_and_collect(env, candidate, candidate, seed, penalty_per_unit)
        _push(buffer_a, t_a)
        _push(buffer_b, t_b)
        stats["self_play_ticks"] += t_a["graphic"].shape[0]
        stats["self_play_matches"] += 1
        seed += 1
    if stats["self_play_ticks"] < target_ticks_self_play:
        print(
            f"[onpolicy] warning: only collected {stats['self_play_ticks']}/"
            f"{target_ticks_self_play} self-play ticks after "
            f"{stats['self_play_matches']} matches (max_matches={max_matches})"
        )

    return stats
