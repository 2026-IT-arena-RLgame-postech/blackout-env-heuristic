"""
On-policy data collection for offline_pretrain.py's periodic eval hook: plays additional
self-vs-heuristic and self-play matches with the CURRENT checkpoint and pushes their raw
transitions into that source's own small FIFO buffer pair (QMIXTrainer.onpolicy_buffers -- the static
dataset's buffer_a/buffer_b stay untouched), applying reward_shaping.blocked_penalty_adjustment as it
goes.

Why this exists: the Run 4 diagnosis (docs/offline_pretrain_runs.md; its one-off scripts were not
kept) found the Run4 checkpoint spends 41.66% of unit-ticks "blocked", with overconfident-but-wrong
Q-values and zero self-correction (100% of blocked runs >=12 ticks kept picking the same wrong
direction the whole run) -- and docs/archive/reward_proposal.md / the actual C# reward code
confirmed nothing in the game's reward structure penalizes wall-collision itself. Two complementary fixes:

  1. reward_shaping.blocked_penalty_adjustment gives training data an explicit negative signal
     for this exact failure mode, applied uniformly to both the static dataset and this module's
     freshly-collected matches.
  2. Pure heuristic-vs-heuristic data never visits "stuck against a wall" states at all (the
     heuristic doesn't get stuck), so growing the training distribution with on-policy rollouts
     from the checkpoint itself -- self-vs-heuristic and self-play -- gives the net examples of
     (and reward signal for recovering from) exactly the states it actually reaches at inference
     time. Each on-policy source lives in its own bounded FIFO buffer, and the trainer draws a fixed
     share of every batch from each buffer (QMIXConfig.batch_source_fracs), so the batch mix no
     longer depends on how much of each source happens to be stored, and the dataset is never
     overwritten.
     In self-vs-heuristic matches the heuristic side's stream is also stored as a behavior-cloning
     demonstration: it shows the heuristic playing against an opponent unlike anything in the
     heuristic-vs-heuristic dataset.

How Run 11 uses it (models/run11_step80k/run11_pipeline.sh train; wiring in offline_pretrain.py):
  - --onpolicy-self-vs-heuristic-frac 0.3 --onpolicy-self-play-frac 0 -> batch_source_fracs
    (0.7, 0.3, 0): 30% of every batch comes from the self-vs-heuristic FIFO buffer (262,144 rows
    per stream, --onpolicy-buffer-capacity), 70% from the static dataset, no self-play.
  - Opponent: V4 (RecommendedStrategicHeuristic, --onpolicy-opponent default "v4" -- the same
    instance periodic eval plays). --onpolicy-opponent mixture swaps in HeuristicPolicyMixture.
  - Timing: collect_onpolicy_data runs right after each periodic eval (every --eval-interval =
    10k steps), never at step 0 (an untrained policy's rows would sit in the FIFO as noise; a
    --resume does collect once up front). Each window targets
    frac * dataset_rows / n_windows ~= 0.3 * 1M / 20 ~= 15k ticks.
  - Sides alternate every match (candidate is team A, then B, ...); seeds run from
    10_000 + train_step, one per match.
  - Matches stop at the first absorption with no battery left anywhere (stop_when_exhausted,
    unless --keep-exhausted), mirroring how the static dataset's dead segments are dropped.
  - Rewards: blocked-tick penalty (--blocked-penalty 0.02) on top of reward v2
    (reward_v2.annotate_sequence) when --reward v2-*; Unity's own reward otherwise.
"""

from __future__ import annotations

import numpy as np

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import TEAM_A_INDICES, TEAM_B_INDICES, team_a_agents, team_b_agents
from blackout_env.model.base import BaseModel
from blackout_env.model.my_policy import direction_vector_to_idx
from blackout_env.train.policy_strength import policy_index
from blackout_env.train.objective_monitor import ObjectiveMonitor, aggregate_objectives
from blackout_env.train.replay_buffer import SOURCE_SELF_PLAY, SOURCE_SELF_VS_HEURISTIC, SequentialReplayBuffer
from blackout_env.train.dead_segments import batteries_in_play
from blackout_env.train.reward_shaping import blocked_penalty_adjustment

ABSORPTION_IDX = 3  # matches qmix_trainer.ABSORPTION_IDX -- team_state row layout
# Ψ = tanh(score_diff / potentialScale) with potentialScale=40 (reward_config.json); past
# |diff| ~53 the slope drops below 0.25, i.e. score changes barely move the shaping reward.
PSI_SCALE = 40.0
PSI_SATURATED_SLOPE = 0.25

# V17-V19: the Elo top tier, reported separately when the on-policy opponent is the mixture.
TOP_TIER = ("strategic_v17", "strategic_v18", "strategic_v19")

_FIELDS = ("graphic", "team_state", "agent_states", "actions", "reward", "done")


def play_and_collect(
    env: BlackOutEnv,
    team_a_policy: BaseModel,
    team_b_policy: BaseModel,
    seed: int,
    penalty_per_unit: float,
    reward_v2=None,
    stop_when_exhausted: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """
    Plays one full match with team_a_policy/team_b_policy controlling their respective teams
    (pass the same object for both for self-play) and returns (transitions_a, transitions_b,
    match_info), the first two each a dict of arrays matching SequentialReplayBuffer.push()'s
    per-field schema (one row per tick), reward already including the blocked-tick penalty.
    match_info carries the physical outcome (winner: 0=team A, 1=team B, -1=draw, final scores)
    for diagnostics.

    stop_when_exhausted: end the match at the first absorption after which no battery is left
    anywhere (train/dead_segments.py) -- the result can no longer change, so the rest of a 420 s
    match (most of it) would only be dropped at training time. The outcome is decided by the
    scores at that point and added to the last row like Unity's own +-1 per agent.
    """
    team_a_names = team_a_agents()
    team_b_names = team_b_agents()

    rows_a: dict[str, list] = {f: [] for f in _FIELDS}
    rows_b: dict[str, list] = {f: [] for f in _FIELDS}

    objectives_a, objectives_b = ObjectiveMonitor(), ObjectiveMonitor()

    obs, _ = env.reset(seed=seed)
    prev_absorption_time_left = float(obs[team_a_names[0]]["team_state"][ABSORPTION_IDX])
    if hasattr(team_a_policy, "reset"):
        team_a_policy.reset()
    if team_b_policy is not team_a_policy and hasattr(team_b_policy, "reset"):
        team_b_policy.reset()
    # A HeuristicPolicyMixture picks this match's policy in reset(); record it for BC weighting.
    played = tuple(_played_policy_id(p) for p in (team_a_policy, team_b_policy))
    empty_obs_steps = 0
    final_info: dict = {}
    terminal_reward = (0.0, 0.0)

    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            if infos:
                final_info = next(iter(infos.values()))
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
            obs, _, _, _, infos = env.step(env_actions)
            if infos:
                final_info = next(iter(infos.values()))
            continue

        dir_a = direction_vector_to_idx(np.stack([actions_a[a] for a in team_a_names]))
        dir_b = direction_vector_to_idx(np.stack([actions_b[a] for a in team_b_names]))
        full_direction_idx = np.concatenate([dir_a, dir_b]).astype(np.int64)

        shared_a, shared_b = obs[team_a_names[0]], obs[team_b_names[0]]
        objectives_a.observe(shared_a["graphic"], shared_a["agent_states"], list(TEAM_A_INDICES))
        objectives_b.observe(shared_b["graphic"], shared_b["agent_states"], list(TEAM_B_INDICES))
        next_obs, rewards, terminations, _, infos = env.step(env_actions)
        if infos:
            final_info = next(iter(infos.values()))
        done = any(terminations.values())

        absorption_fired = False
        if next_obs and team_a_names[0] in next_obs:
            absorption_time_left = float(next_obs[team_a_names[0]]["team_state"][ABSORPTION_IDX])
            absorption_fired = absorption_time_left > prev_absorption_time_left + 1e-6
            prev_absorption_time_left = absorption_time_left
        buffer_done = done or absorption_fired

        reward_a = sum(rewards.get(a, 0.0) for a in team_a_names)
        reward_b = sum(rewards.get(a, 0.0) for a in team_b_names)
        if done:
            terminal_reward = (reward_a, reward_b)

        for f, v in zip(_FIELDS, (shared_a["graphic"], shared_a["team_state"], shared_a["agent_states"], full_direction_idx, reward_a, buffer_done)):
            rows_a[f].append(v)
        for f, v in zip(_FIELDS, (shared_b["graphic"], shared_b["team_state"], shared_b["agent_states"], full_direction_idx, reward_b, buffer_done)):
            rows_b[f].append(v)

        if stop_when_exhausted and absorption_fired and not done and team_a_names[0] in next_obs:
            view = next_obs[team_a_names[0]]
            if batteries_in_play(view["graphic"][None], view["agent_states"][None])[0] == 0:
                score_0, score_1 = float(view["team_state"][0]), float(view["team_state"][1])
                sign = 0.0 if abs(score_0 - score_1) < 1e-6 else (1.0 if score_0 > score_1 else -1.0)
                rows_a["reward"][-1] += sign * len(team_a_names)
                rows_b["reward"][-1] -= sign * len(team_b_names)
                terminal_reward = (sign * len(team_a_names), -sign * len(team_b_names))
                final_info = {"winner": -1 if sign == 0 else (0 if sign > 0 else 1), "score_0": score_0, "score_1": score_1}
                break

        obs = next_obs

    def _finalize(rows: dict[str, list], team_indices: tuple[int, ...]) -> tuple[dict[str, np.ndarray], float, int]:
        agent_states = np.stack(rows["agent_states"])
        done_arr = np.array(rows["done"], dtype=np.bool_)
        env_reward = np.array(rows["reward"], dtype=np.float32)
        penalty = blocked_penalty_adjustment(agent_states, done_arr, team_indices, penalty_per_unit)
        # Blocked unit-ticks recovered from the penalty itself so the count matches exactly
        # what was applied (0 when the penalty is disabled).
        blocked_unit_ticks = int(round(-penalty.sum() / penalty_per_unit)) if penalty_per_unit > 0 else 0
        transitions = dict(
            graphic=np.stack(rows["graphic"]),
            team_state=np.stack(rows["team_state"]),
            agent_states=agent_states,
            actions=np.stack(rows["actions"]),
            reward=env_reward + penalty,
            done=done_arr,
        )
        if reward_v2 is not None:
            from blackout_env.train.reward_v2 import annotate_sequence

            # one match per call: its last row is the match end
            annotated = annotate_sequence(transitions["graphic"], agent_states, transitions["team_state"],
                                          env_reward, done_arr, reward_v2)
            transitions.update(reward=annotated["reward"] + penalty, done=annotated["done"],
                               potential=annotated["potential"], terminal=annotated["terminal"])
        return transitions, float(env_reward.sum()), blocked_unit_ticks

    t_a, env_reward_a, blocked_a = _finalize(rows_a, TEAM_A_INDICES)
    t_b, env_reward_b, blocked_b = _finalize(rows_b, TEAM_B_INDICES)

    score_diff = (t_a["team_state"][:, 0] - t_a["team_state"][:, 1]) * 100.0  # team A's perspective
    psi_slope = 1.0 - np.tanh(score_diff / PSI_SCALE) ** 2
    winner = final_info.get("winner")
    match_info = {
        "ticks": int(t_a["graphic"].shape[0]),
        "winner": -1 if winner in (None, -1) else int(winner),
        "score_0": float(final_info.get("score_0", 0.0)) * 100.0,
        "score_1": float(final_info.get("score_1", 0.0)) * 100.0,
        "env_reward": (env_reward_a, env_reward_b),
        "terminal_reward": terminal_reward,
        "blocked_unit_ticks": (blocked_a, blocked_b),
        "abs_score_diff_sum": float(np.abs(score_diff).sum()),
        "psi_saturated_ticks": int((psi_slope < PSI_SATURATED_SLOPE).sum()),
        # Per-side scoring pipeline (pickups/deliveries/approach rates) -- the breakdown that
        # located Run 6's actual bottleneck; see train/objective_monitor.py.
        "objectives": (objectives_a.counts, objectives_b.counts),
        "policy_id": played,
    }
    return t_a, t_b, match_info


def _played_policy_id(policy) -> str | None:
    """policy_id recorded in a policy's ``current_sample`` (HeuristicPolicyMixture, V4PolicyFamily);
    None for a plain heuristic such as V4 or for the candidate network."""
    sample = getattr(policy, "current_sample", None)
    return sample.policy_id if sample is not None else None


def _push(buffer: SequentialReplayBuffer, transitions: dict[str, np.ndarray], source: int, demo: bool, policy_id: str | None = None) -> None:
    """Append one match's rows for one team to its on-policy FIFO buffer, in time order.

    demo=True marks the rows as heuristic demonstrations, so the trainer applies BC to them;
    only then is policy_id kept (as a policy_strength index, for --bc-policy-weighting) --
    candidate rows always get -1. potential/terminal exist only when reward v2 annotated the
    match; otherwise the buffer falls back to 0.0 and terminal = done.
    """
    policy = policy_index(policy_id) if demo else -1
    n = transitions["graphic"].shape[0]
    for i in range(n):
        buffer.push(
            transitions["graphic"][i],
            transitions["team_state"][i],
            transitions["agent_states"][i],
            transitions["actions"][i],
            float(transitions["reward"][i]),
            bool(transitions["done"][i]),
            source,
            demo,
            potential=float(transitions["potential"][i]) if "potential" in transitions else 0.0,
            terminal=bool(transitions["terminal"][i]) if "terminal" in transitions else None,
            policy=policy,
        )


def _new_phase_totals() -> dict[str, float]:
    """Running sums for one collection phase (self-vs-heuristic or self-play); see _phase_metrics."""
    return dict(
        ticks=0, matches=0, wins=0, losses=0, draws=0, margin=0.0,
        candidate_env_reward=0.0, other_env_reward=0.0, candidate_terminal_reward=0.0,
        candidate_blocked=0, other_blocked=0, abs_score_diff=0.0, psi_saturated=0,
        candidate_objectives=[], other_objectives=[],
    )


def _accumulate(totals: dict[str, float], info: dict, candidate_team: int) -> None:
    """Add one play_and_collect match_info to a phase's totals, re-indexed from physical teams
    (0 = A, 1 = B) to candidate vs other so side-swapped matches pool correctly."""
    other = 1 - candidate_team
    totals["ticks"] += info["ticks"]
    totals["matches"] += 1
    if info["winner"] == -1:
        totals["draws"] += 1
    elif info["winner"] == candidate_team:
        totals["wins"] += 1
    else:
        totals["losses"] += 1
    scores = (info["score_0"], info["score_1"])
    totals["margin"] += scores[candidate_team] - scores[other]
    totals["candidate_env_reward"] += info["env_reward"][candidate_team]
    totals["other_env_reward"] += info["env_reward"][other]
    totals["candidate_terminal_reward"] += info["terminal_reward"][candidate_team]
    totals["candidate_blocked"] += info["blocked_unit_ticks"][candidate_team]
    totals["other_blocked"] += info["blocked_unit_ticks"][other]
    totals["abs_score_diff"] += info["abs_score_diff_sum"]
    totals["psi_saturated"] += info["psi_saturated_ticks"]
    totals["candidate_objectives"].append(info["objectives"][candidate_team])
    totals["other_objectives"].append(info["objectives"][other])


def _phase_metrics(totals: dict[str, float], penalty_per_unit: float) -> dict[str, float]:
    """Turn a phase's totals into per-match (outcome, margin) and per-tick (reward, blocked,
    Psi-saturation) rates plus candidate_/opponent_ objective rates; logged under onpolicy/.
    Margins are in game points (scores x 100)."""
    ticks = max(1, totals["ticks"])
    matches = max(1, totals["matches"])
    n_team_units = len(TEAM_A_INDICES)
    objectives = aggregate_objectives(totals["candidate_objectives"], "candidate_")
    objectives.update(aggregate_objectives(totals["other_objectives"], "opponent_"))
    return {
        **objectives,
        "win_rate": totals["wins"] / matches,
        "loss_rate": totals["losses"] / matches,
        "draw_rate": totals["draws"] / matches,
        "mean_margin": totals["margin"] / matches,
        "mean_match_ticks": totals["ticks"] / matches,
        "candidate_env_reward_per_tick": totals["candidate_env_reward"] / ticks,
        "opponent_env_reward_per_tick": totals["other_env_reward"] / ticks,
        "candidate_terminal_reward_mean": totals["candidate_terminal_reward"] / matches,
        "candidate_blocked_penalty_per_tick": -penalty_per_unit * totals["candidate_blocked"] / ticks,
        "candidate_blocked_unit_frac": totals["candidate_blocked"] / (n_team_units * ticks),
        "opponent_blocked_unit_frac": totals["other_blocked"] / (n_team_units * ticks),
        "abs_score_diff_mean": totals["abs_score_diff"] / ticks,
        "psi_saturated_frac": totals["psi_saturated"] / ticks,
    }


def collect_onpolicy_data(
    env: BlackOutEnv,
    buffers: dict[int, tuple[SequentialReplayBuffer, SequentialReplayBuffer]],
    candidate: BaseModel,
    heuristic: BaseModel,
    target_ticks_self_vs_heuristic: int,
    target_ticks_self_play: int,
    penalty_per_unit: float,
    seed_start: int,
    max_matches: int = 50,
    reward_v2=None,
    stop_when_exhausted: bool = True,
) -> dict[str, float]:
    """
    Plays self-vs-heuristic matches (candidate vs heuristic, side swapped every match for
    fairness) until >= target_ticks_self_vs_heuristic ticks are collected, then self-play matches
    (candidate vs itself) until >= target_ticks_self_play more, pushing every match's transitions
    into buffers[SOURCE_SELF_VS_HEURISTIC] / buffers[SOURCE_SELF_PLAY] (team A, team B) as it goes
    (so a crash mid-window still keeps whatever was pushed so far); a phase with a positive target
    needs its buffer pair present. `seed_start` should differ every call (e.g. the current train step) so successive
    windows don't replay identical episodes. Stops early past `max_matches` per phase with a
    printed warning rather than looping forever if matches turn out shorter than expected.

    Returns tick/match counts plus per-phase diagnostics under "self_vs_heuristic/<metric>" and
    "self_play/<metric>" (outcome from the candidate's side, env reward vs blocked penalty vs
    terminal reward, and how much of the window sat in the saturated region of the Ψ tanh).
    In self-play "candidate" is team A.
    """
    svh = _new_phase_totals()
    sp = _new_phase_totals()

    seed = seed_start
    swap = False
    while svh["ticks"] < target_ticks_self_vs_heuristic and svh["matches"] < max_matches:
        team_a_policy, team_b_policy = (heuristic, candidate) if swap else (candidate, heuristic)
        t_a, t_b, info = play_and_collect(env, team_a_policy, team_b_policy, seed, penalty_per_unit, reward_v2, stop_when_exhausted)
        buffer_a, buffer_b = buffers[SOURCE_SELF_VS_HEURISTIC]
        _push(buffer_a, t_a, SOURCE_SELF_VS_HEURISTIC, demo=swap, policy_id=info["policy_id"][0])  # team A is the heuristic when swapped
        _push(buffer_b, t_b, SOURCE_SELF_VS_HEURISTIC, demo=not swap, policy_id=info["policy_id"][1])
        _accumulate(svh, info, candidate_team=1 if swap else 0)
        opponent_id = info["policy_id"][0 if swap else 1]
        if opponent_id is not None:
            scores = (info["score_0"], info["score_1"])
            cand = 1 if swap else 0
            tier = "top" if opponent_id in TOP_TIER else "rest"
            svh[f"margin_vs_{tier}"] = svh.get(f"margin_vs_{tier}", 0.0) + scores[cand] - scores[1 - cand]
            svh[f"matches_vs_{tier}"] = svh.get(f"matches_vs_{tier}", 0) + 1
        seed += 1
        swap = not swap
    if svh["ticks"] < target_ticks_self_vs_heuristic:
        print(
            f"[onpolicy] warning: only collected {svh['ticks']}/{target_ticks_self_vs_heuristic} "
            f"self-vs-heuristic ticks after {svh['matches']} matches (max_matches={max_matches})"
        )

    while sp["ticks"] < target_ticks_self_play and sp["matches"] < max_matches:
        t_a, t_b, info = play_and_collect(env, candidate, candidate, seed, penalty_per_unit, reward_v2, stop_when_exhausted)
        buffer_a, buffer_b = buffers[SOURCE_SELF_PLAY]
        _push(buffer_a, t_a, SOURCE_SELF_PLAY, demo=False)
        _push(buffer_b, t_b, SOURCE_SELF_PLAY, demo=False)
        _accumulate(sp, info, candidate_team=0)
        seed += 1
    if sp["ticks"] < target_ticks_self_play:
        print(
            f"[onpolicy] warning: only collected {sp['ticks']}/{target_ticks_self_play} "
            f"self-play ticks after {sp['matches']} matches (max_matches={max_matches})"
        )

    for tier in ("top", "rest"):
        if svh.get(f"matches_vs_{tier}"):
            svh[f"mean_margin_vs_{tier}"] = svh[f"margin_vs_{tier}"] / svh[f"matches_vs_{tier}"]
    stats: dict[str, float] = {
        "self_vs_heuristic_ticks": svh["ticks"],
        "self_vs_heuristic_matches": svh["matches"],
        "self_play_ticks": sp["ticks"],
        "self_play_matches": sp["matches"],
    }
    for prefix, totals in (("self_vs_heuristic", svh), ("self_play", sp)):
        if totals["matches"]:
            for key, value in _phase_metrics(totals, penalty_per_unit).items():
                stats[f"{prefix}/{key}"] = value
            for key in ("mean_margin_vs_top", "mean_margin_vs_rest", "matches_vs_top", "matches_vs_rest"):
                if key in totals:
                    stats[f"{prefix}/{key}"] = totals[key]
    return stats
