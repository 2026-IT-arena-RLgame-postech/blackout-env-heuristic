"""
Actor process (see package docstring): owns one BlackOutEnv and reproduces
qmix_trainer.QMIXTrainer.collect_step()'s rollout logic verbatim (bootstrap heuristic-vs-
heuristic phase, self-play opponent via ema_net, heuristic-opponent episodes, epsilon-mixed
exploration, absorption-as-episode-boundary) -- the only thing that moved out is WHERE the
net/ema_net forward pass happens (inference_server.py, batched across every actor) and WHERE
transitions land (learner.py's replay buffers, via transition_queue instead of a local push()).

Everything here should read as a straight decomposition of QMIXTrainer.collect_step/_reset_env/
select_actions/select_actions_heuristic -- if this drifts from that method's behavior, that's a
bug, not an intentional difference.
"""

from __future__ import annotations

import numpy as np

from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import N_AGENTS, N_TEAM_A, team_a_agents, team_b_agents
from blackout_env.heuristics import HeuristicPolicyMixture
from blackout_env.model.my_policy import DIRECTION_VECTORS, direction_vector_to_idx
from blackout_env.train.qmix_trainer import QMIXConfig
from blackout_env.train.tb_logger import TBLogger

from .shared import SharedState, compute_epsilon, get_until_stop, put_until_stop

ABSORPTION_IDX = 3  # team_state row: [own_score, opp_score, episode_time_left, absorption_time_left]
N_TEAM = N_TEAM_A


def _heuristic_direction_idx(obs, agents: list[str], heuristic) -> np.ndarray:
    """See QMIXTrainer._heuristic_direction_idx -- identical logic, standalone here since the
    actor has no QMIXTrainer instance to hang it off of."""
    team_obs = {a: obs[a] for a in agents}
    heuristic_actions = heuristic.act(team_obs)
    vectors = np.stack([heuristic_actions[a] for a in agents])
    return direction_vector_to_idx(vectors)


def _pack_direction_idx(direction_idx: np.ndarray, team_a: list[str], team_b: list[str]):
    """See QMIXTrainer._pack_direction_idx -- identical logic."""
    env_actions = {}
    full_direction_idx = np.zeros(N_AGENTS, dtype=np.int64)
    for team_idx, agents in enumerate((team_a, team_b)):
        for slot, agent in enumerate(agents):
            d = int(direction_idx[team_idx, slot])
            env_actions[agent] = DIRECTION_VECTORS[d]
            full_direction_idx[team_idx * N_TEAM + slot] = d
    return env_actions, full_direction_idx


def _reset(env, heuristic_a, heuristic_b, team_a: list[str], cfg: QMIXConfig):
    """See QMIXTrainer._reset_env -- identical logic (re-randomize online/opponent roles,
    re-seed per-episode heuristic state, snapshot the starting absorption timer)."""
    obs, _ = env.reset()
    online_is_team_a = bool(np.random.rand() < 0.5)
    opponent_is_heuristic = bool(np.random.rand() < cfg.heuristic_opponent_frac)
    prev_absorption = float(obs[team_a[0]]["team_state"][ABSORPTION_IDX])
    heuristic_a.reset()
    heuristic_b.reset()
    return obs, online_is_team_a, opponent_is_heuristic, prev_absorption


def run_actor(
    actor_id: int,
    build_path: str | None,
    cfg: QMIXConfig,
    shared: SharedState,
    request_queue,
    response_queue,
    transition_queue,
    time_scale: float,
    unity_log_file: str | None,
    tb_log_dir: str | None,
    smoke_test: bool = False,
) -> None:
    heuristic_a = HeuristicPolicyMixture(seed=cfg.heuristic_seed_a + 1000 * actor_id)
    heuristic_b = HeuristicPolicyMixture(seed=cfg.heuristic_seed_b + 1000 * actor_id)
    team_a = team_a_agents()
    team_b = team_b_agents()

    if smoke_test:
        from .fake_env import FakeBlackOutEnv

        env = FakeBlackOutEnv(seed=actor_id)
    else:
        additional_args = ["-logFile", unity_log_file] if unity_log_file else None
        env = BlackOutEnv(
            env_path=build_path,
            map_w=24,
            map_h=24,
            time_scale=time_scale,
            no_graphics=True,
            additional_args=additional_args,
        )

    tb = TBLogger(f"{tb_log_dir}/actor_{actor_id}" if tb_log_dir else None)

    local_step = 0
    episode_return_a = episode_return_b = 0.0
    episode_count = wins_a = wins_b = draws = 0

    obs, online_is_team_a, opponent_is_heuristic, prev_absorption = _reset(env, heuristic_a, heuristic_b, team_a, cfg)

    try:
        while not shared.stop.is_set():
            bootstrapping = bool(shared.bootstrapping.value)

            if bootstrapping:
                dir_a = _heuristic_direction_idx(obs, team_a, heuristic_a)
                dir_b = _heuristic_direction_idx(obs, team_b, heuristic_b)
                direction_idx = np.stack([dir_a, dir_b])
                noise_frac = cfg.heuristic_bootstrap_noise_frac
                if noise_frac > 0:
                    random_dirs = np.random.randint(0, 8, size=direction_idx.shape)
                    direction_idx = np.where(np.random.rand(*direction_idx.shape) < noise_frac, random_dirs, direction_idx)
            else:
                obs_a, obs_b = obs[team_a[0]], obs[team_b[0]]
                graphic = np.stack([obs_a["graphic"], obs_b["graphic"]])
                team_state = np.stack([obs_a["team_state"], obs_b["team_state"]])
                agent_states = np.stack([obs_a["agent_states"], obs_b["agent_states"]])
                sent = put_until_stop(request_queue, (actor_id, graphic, team_state, agent_states), shared.stop)
                if not sent:
                    continue  # shared.stop fired while waiting for room in request_queue
                # This actor has nothing else useful to do until inference_server.py answers
                # this exact request (see that module -- it batches across actors, not within
                # one actor's own sequential decisions). get_until_stop retries indefinitely
                # (not "give up after N seconds") -- the timeout here only sets how often it
                # re-checks shared.stop, so it stays small regardless of how long a real answer
                # might take.
                response = get_until_stop(response_queue, shared.stop)
                if response is None:
                    continue  # shared.stop fired (or inference server never answered) -- bail
                greedy_online, greedy_ema = response

                opponent_idx = 1 if online_is_team_a else 0
                greedy = greedy_online.copy()
                greedy[opponent_idx] = greedy_ema[opponent_idx]

                dir_a = _heuristic_direction_idx(obs, team_a, heuristic_a)
                dir_b = _heuristic_direction_idx(obs, team_b, heuristic_b)
                heuristic_dirs = np.stack([dir_a, dir_b])

                eps = compute_epsilon(cfg, shared.global_step.value, shared.phase2_anchor_step.value)
                direction_idx = np.where(np.random.rand(2, N_TEAM) < eps, heuristic_dirs, greedy)

                if opponent_is_heuristic:
                    direction_idx[opponent_idx] = heuristic_dirs[opponent_idx]

            env_actions, full_direction_idx = _pack_direction_idx(direction_idx, team_a, team_b)
            obs_a, obs_b = obs[team_a[0]], obs[team_b[0]]
            next_obs, rewards, terminations, _, infos = env.step(env_actions)
            done = any(terminations.values())

            absorption_left = float(next_obs[team_a[0]]["team_state"][ABSORPTION_IDX])
            absorption_fired = prev_absorption is not None and absorption_left > prev_absorption + 1e-6
            prev_absorption = absorption_left
            buffer_done = done or absorption_fired

            reward_a = sum(rewards[a] for a in team_a)
            reward_b = sum(rewards[a] for a in team_b)
            episode_return_a += reward_a
            episode_return_b += reward_b

            sent_a = put_until_stop(transition_queue, ("a", obs_a["graphic"], obs_a["team_state"], obs_a["agent_states"], full_direction_idx, reward_a, buffer_done), shared.stop)
            sent_b = put_until_stop(transition_queue, ("b", obs_b["graphic"], obs_b["team_state"], obs_b["agent_states"], full_direction_idx, reward_b, buffer_done), shared.stop)
            if not (sent_a and sent_b):
                continue  # shared.stop fired while waiting for room in transition_queue
            shared.increment_step()
            local_step += 1

            if local_step % cfg.tb_log_interval == 0:
                tb.scalars("reward/step", {"team_a": reward_a, "team_b": reward_b}, local_step)
                tb.scalar("bootstrap/is_heuristic_fill_phase", float(bootstrapping), local_step)

            if done:
                # NOT episode-return comparison -- see QMIXTrainer.collect_step's comment on why
                # the physical winner has to come from Unity's own score, not summed shaping reward.
                physical_winner = next(iter(infos.values())).get("winner") if infos else None
                episode_count += 1
                team_a_won = physical_winner == 0
                team_b_won = physical_winner == 1
                if team_a_won:
                    wins_a += 1
                elif team_b_won:
                    wins_b += 1
                else:
                    draws += 1
                online_won = (team_a_won and online_is_team_a) or (team_b_won and not online_is_team_a)
                tb.scalars("episode/return", {"team_a": episode_return_a, "team_b": episode_return_b}, local_step)
                tb.scalars("episode/win_rate", {"team_a": wins_a / episode_count, "team_b": wins_b / episode_count}, local_step)
                tb.scalar("episode/draw_rate", draws / episode_count, local_step)
                tb.scalar("episode/win_rate_online", float(online_won), local_step)
                episode_return_a = episode_return_b = 0.0
                obs, online_is_team_a, opponent_is_heuristic, prev_absorption = _reset(env, heuristic_a, heuristic_b, team_a, cfg)
            else:
                obs = next_obs
    finally:
        # This process is a pure producer on both queues -- on a normal exit, Python's
        # multiprocessing atexit machinery otherwise tries to flush every buffered-but-not-yet-
        # written item through each queue's feeder thread before letting the process die
        # (join_thread()). Once the reader on the other end (learner.py / inference_server.py)
        # has stopped draining -- exactly the normal case at shutdown -- that flush can never
        # complete (the OS pipe fills after just a couple of ~30KB graphic arrays) and hangs
        # process exit indefinitely instead of merely slowly. Losing a handful of in-flight
        # transitions at shutdown is fine; hanging every actor process until a hard kill is not.
        transition_queue.cancel_join_thread()
        request_queue.cancel_join_thread()
        env.close()
        tb.close()
