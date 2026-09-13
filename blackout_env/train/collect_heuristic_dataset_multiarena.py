"""
Standalone entry point: collect heuristic-vs-heuristic (+ a little uniform-random exploration
noise) transitions from N concurrent arenas in ONE Unity process (see
blackout_env.env.multi_arena_env.MultiArenaBlackOutEnv and
blackout/Assets/Editor/ArenaDuplicator.cs), merging every arena's transitions into one combined
buffer_a.npz/buffer_b.npz -- the multi-arena analogue of collect_heuristic_dataset.py.

Reimplements that script's essential logic (heuristic action selection with exploration noise,
absorption-boundary buffer_done bookkeeping, per-team SequentialReplayBuffer push) standalone
rather than through QMIXTrainer.collect_step()/._reset_env(), which assume ONE synchronous
self.env.step() call maps 1:1 to "the next decision for the one match this trainer owns" --
that assumption doesn't hold here: a single MultiArenaBlackOutEnv.step() call advances every
arena's simulation together, and different arenas become "ready" (produce a fresh decision) on
different ticks depending on their own decisionPeriod phase and episode-restart timing, so each
arena needs its own independent action/episode bookkeeping driven off the same shared
connection instead of N independent blocking env.step() loops (which would silently desync
every arena but the one being waited on).

Usage:
    python -m blackout_env.train.collect_heuristic_dataset_multiarena \\
        --build build/mac_multiarena/BlackOut.app --num-arenas 4 \\
        --steps 100000 --noise-frac 0.1 --out datasets/run1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from blackout_env.env.constants import N_AGENTS, N_TEAM_A, team_a_agents, team_b_agents
from blackout_env.env.multi_arena_env import ArenaStep, MultiArenaBlackOutEnv
from blackout_env.heuristics import HeuristicPolicyMixture
from blackout_env.model.my_policy import DIRECTION_VECTORS, direction_vector_to_idx
from blackout_env.train.offline_dataset import save_buffer
from blackout_env.train.qmix_trainer import ABSORPTION_IDX
from blackout_env.train.replay_buffer import SequentialReplayBuffer

TEAM_A = team_a_agents()
TEAM_B = team_b_agents()


def _heuristic_direction_idx(obs: dict, agents: list[str], heuristic) -> np.ndarray:
    """See QMIXTrainer._heuristic_direction_idx -- identical logic, reimplemented standalone
    since that method's sibling collect_step()/select_actions_heuristic() aren't reusable here
    (they call self.env directly; see module docstring)."""
    team_obs = {a: obs[a] for a in agents}
    heuristic_actions = heuristic.act(team_obs)
    vectors = np.stack([heuristic_actions[a] for a in agents])
    return direction_vector_to_idx(vectors)


def _pack_direction_idx(direction_idx: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """See QMIXTrainer._pack_direction_idx."""
    env_actions = {}
    full_direction_idx = np.zeros(N_AGENTS, dtype=np.int64)
    for team_idx, agents in enumerate((TEAM_A, TEAM_B)):
        for slot, agent in enumerate(agents):
            d = int(direction_idx[team_idx, slot])
            env_actions[agent] = DIRECTION_VECTORS[d]
            full_direction_idx[team_idx * N_TEAM_A + slot] = d
    return env_actions, full_direction_idx


def select_actions_heuristic(
    obs: dict, heuristic_a, heuristic_b, noise_frac: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """See QMIXTrainer.select_actions_heuristic -- identical logic."""
    dir_a = _heuristic_direction_idx(obs, TEAM_A, heuristic_a)
    dir_b = _heuristic_direction_idx(obs, TEAM_B, heuristic_b)
    direction_idx = np.stack([dir_a, dir_b])
    if noise_frac > 0:
        random_dirs = np.random.randint(0, 8, size=direction_idx.shape)
        direction_idx = np.where(np.random.rand(*direction_idx.shape) < noise_frac, random_dirs, direction_idx)
    return _pack_direction_idx(direction_idx)


class _ArenaState:
    """Per-arena bookkeeping the driver loop needs between step() calls: the observation an
    action was chosen from (to push once its resulting reward/done arrives), that action's
    snapped direction indices (what actually gets written to the replay buffer -- see
    QMIXTrainer.select_actions_heuristic's docstring on why), and absorption-boundary tracking."""

    def __init__(self, obs: dict, action_idx: np.ndarray, absorption_time_left: float):
        self.prev_obs = obs
        self.prev_action_idx = action_idx
        self.prev_absorption = absorption_time_left
        self.episodes = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="Path to a multi-arena Unity build (see ArenaDuplicator.cs)")
    parser.add_argument("--num-arenas", type=int, required=True, help="Must match the build's ARENA_COUNT")
    parser.add_argument("--steps", type=int, required=True, help="Total transitions to collect across all arenas combined (per stream)")
    parser.add_argument(
        "--noise-frac", type=float, default=0.1,
        help="Per-unit probability of a uniformly random compass direction instead of the "
        "heuristic's own choice -- see collect_heuristic_dataset.py's --noise-frac.",
    )
    parser.add_argument("--out", required=True, help="Output directory for buffer_a.npz/buffer_b.npz")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale")
    parser.add_argument("--graphics", action="store_true", help="Show the Unity window instead of headless")
    parser.add_argument(
        "--heuristic-seed-base", type=int, default=0,
        help="Arena i's team A/B heuristics seed with (base + 2*i, base + 2*i + 1).",
    )
    args = parser.parse_args()

    if args.num_arenas < 1:
        raise SystemExit("--num-arenas must be >= 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = MultiArenaBlackOutEnv(
        env_path=args.build,
        num_arenas=args.num_arenas,
        map_w=24,
        map_h=24,
        time_scale=args.time_scale,
        no_graphics=not args.graphics,
    )

    prep = env.preprocessor
    graphic_shape = (24, 24, prep.n_graphic_channels)
    buffer_a = SequentialReplayBuffer(args.steps, graphic_shape, prep.team_state_size, prep.agent_state_size)
    buffer_b = SequentialReplayBuffer(args.steps, graphic_shape, prep.team_state_size, prep.agent_state_size)

    heuristics_a = [HeuristicPolicyMixture(seed=args.heuristic_seed_base + 2 * i) for i in range(args.num_arenas)]
    heuristics_b = [HeuristicPolicyMixture(seed=args.heuristic_seed_base + 2 * i + 1) for i in range(args.num_arenas)]

    win_a = win_b = draws = 0
    total_pushed = 0

    try:
        arena_obs = env.reset()
        pending_actions: list[dict[str, np.ndarray] | None] = [None] * args.num_arenas
        arenas: list[_ArenaState] = []
        for i in range(args.num_arenas):
            actions_i, dir_idx_i = select_actions_heuristic(arena_obs[i], heuristics_a[i], heuristics_b[i], args.noise_frac)
            pending_actions[i] = actions_i
            absorption0 = float(arena_obs[i][TEAM_A[0]]["team_state"][ABSORPTION_IDX])
            arenas.append(_ArenaState(arena_obs[i], dir_idx_i, absorption0))

        t0 = time.time()
        while total_pushed < args.steps:
            results: list[ArenaStep | None] = env.step(pending_actions)
            any_pushed = False
            for i, result in enumerate(results):
                if result is None:
                    continue
                any_pushed = True
                st = arenas[i]

                if result.episode_started:
                    heuristics_a[i].reset()
                    heuristics_b[i].reset()

                done = any(result.terminations.values())
                absorption_time_left = float(result.obs[TEAM_A[0]]["team_state"][ABSORPTION_IDX])
                absorption_fired = absorption_time_left > st.prev_absorption + 1e-6
                st.prev_absorption = absorption_time_left
                buffer_done = done or absorption_fired

                obs_a_prev, obs_b_prev = st.prev_obs[TEAM_A[0]], st.prev_obs[TEAM_B[0]]
                reward_a = sum(result.rewards.get(a, 0.0) for a in TEAM_A)
                reward_b = sum(result.rewards.get(a, 0.0) for a in TEAM_B)

                buffer_a.push(obs_a_prev["graphic"], obs_a_prev["team_state"], obs_a_prev["agent_states"],
                              st.prev_action_idx, reward_a, buffer_done)
                buffer_b.push(obs_b_prev["graphic"], obs_b_prev["team_state"], obs_b_prev["agent_states"],
                              st.prev_action_idx, reward_b, buffer_done)
                total_pushed += 1

                if done:
                    st.episodes += 1
                    winner = result.info.get("winner")
                    if winner == 0:
                        win_a += 1
                    elif winner == 1:
                        win_b += 1
                    else:
                        draws += 1

                # Choose this arena's next action from the fresh state s' this tick produced.
                actions_i, dir_idx_i = select_actions_heuristic(result.obs, heuristics_a[i], heuristics_b[i], args.noise_frac)
                pending_actions[i] = actions_i
                st.prev_obs = result.obs
                st.prev_action_idx = dir_idx_i

            if any_pushed and (total_pushed % 1000 < args.num_arenas or total_pushed >= args.steps):
                elapsed = time.time() - t0
                episodes = sum(a.episodes for a in arenas)
                print(
                    f"[collect-multiarena] step {total_pushed}/{args.steps} "
                    f"({total_pushed / elapsed:.1f} steps/s, buffer_a={len(buffer_a)}, buffer_b={len(buffer_b)}, "
                    f"episodes={episodes}, W/L/D={win_a}/{win_b}/{draws})"
                )
    finally:
        env.close()

    save_buffer(buffer_a, out_dir / "buffer_a.npz")
    save_buffer(buffer_b, out_dir / "buffer_b.npz")
    print(f"[collect-multiarena] saved {len(buffer_a)} transitions/stream to {out_dir}")
    print(
        f"[collect-multiarena] next: python -m blackout_env.train.offline_pretrain "
        f"--dataset-dir {out_dir} --steps <gradient steps> --checkpoint-dir <dir>"
    )


if __name__ == "__main__":
    main()
