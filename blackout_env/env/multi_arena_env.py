"""
Multi-arena BlackOut environment wrapper.

Connects to a PrototypeMultiArena Unity build (see blackout/Assets/Editor/ArenaDuplicator.cs)
running `num_arenas` independent matches in one Unity process. Every arena's units/map agent
share one BehaviorName per team across the whole scene, so ML-Agents' own communicator already
batches every arena's decisions into the same handful of round-trips per tick -- see
docs/perf_experiments_20260913.md for why that round-trip, not payload size or Time.timeScale,
is BlackOutEnv's real throughput ceiling (measured ~4x aggregate throughput at the 4-arena
build vs. single-arena at the same process count).

Each arena's own BlackOutAgent/MapObsAgent observations carry an extra "arenaIndex" float
(see BlackOutAgent.cs/MapObsAgent.cs) so this class can tell arena 0's unit 3 apart from arena
1's unit 3 -- something a legacy single-arena build's observations don't carry at all, which is
exactly why this needs a purpose-built class rather than reusing BlackOutEnv.

Scope: built for blackout_env.train.collect_heuristic_dataset_multiarena's use case --
heuristic-vs-heuristic rollout collection into replay buffers -- not online RL training.
BlackOutEnv itself is untouched and remains the class to use for a single arena or online
training; this is a separate, additive class because a multi-arena scene's episodes each end
and restart independently forever, which doesn't fit PettingZoo's ParallelEnv model (every
agent terminates together, then the whole env needs one reset()) without reworking most of
that class's per-match-scoped bookkeeping (winner, score, time_left all currently single
values, one per BlackOutEnv instance).

Each arena's obs dict uses the exact same "unit_0".."unit_9" keys as BlackOutEnv -- only the
list index (0..num_arenas-1) says which arena it came from -- so existing per-match code
(StrategicHeuristic.act(obs), HeuristicPolicyMixture, ...) needs zero changes to run once per
arena per tick.

Usage
-----
    env = MultiArenaBlackOutEnv(env_path="build/mac_multiarena/BlackOut.app", num_arenas=4)
    arena_obs = env.reset()                        # list[num_arenas] of obs dicts
    pending_actions = [None] * env.num_arenas
    while True:
        results = env.step(pending_actions)         # list[num_arenas], None = not ready yet
        for i, result in enumerate(results):
            if result is None:
                continue
            pending_actions[i] = my_policy.act(result.obs)   # choose this arena's next action
    env.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .blackout_env import find_free_port, _kill_stray_unity_processes
from .constants import BEHAVIOR_NAME, MAP_BEHAVIOR_NAME, N_AGENTS, N_TEAM_A, agent_name
from .obs_preprocessor import load_semantic_config
from .my_obs_preprocessor import MyObsPreprocessor
from .seed_channel import SeedChannel


class ArenaStep(NamedTuple):
    """One arena's result for a tick it actually had a fresh decision or terminal step on."""

    obs: dict[str, dict[str, np.ndarray]]
    rewards: dict[str, float]
    terminations: dict[str, bool]
    info: dict[str, Any]  # score_0, score_1, time_left, absorption_time_left, + winner if done
    episode_started: bool  # True iff this is this arena's first decision since a termination
    # (or since reset()) -- mirrors BlackOutEnv callers re-seeding a heuristic mixture per
    # episode; see HeuristicPolicyMixture's own time_left-jump self-detection for why this is a
    # convenience, not a strict correctness requirement.


class MultiArenaBlackOutEnv:
    """
    Runs `num_arenas` independent BlackOut matches in one Unity process.

    Unlike BlackOutEnv (PettingZoo ParallelEnv, one match per instance), this is NOT a
    PettingZoo env: arenas terminate and restart independently forever, so there's no single
    "the episode is over" moment for the whole environment -- reset() only runs once, right
    after connecting; from then on step() is called in a loop and each arena's own episodes
    cycle on their own schedule (Unity's BlackOutEpisodeCoordinator handles the actual
    episode-restart timing, same as it always has).
    """

    _DEFAULT_CONFIG = Path(__file__).parent.parent / "semantic_map_config.json"

    def __init__(
        self,
        env_path: str | None,
        num_arenas: int,
        semantic_config_path: str | Path | None = None,
        map_w: int = 24,
        map_h: int = 24,
        worker_id: int = 0,
        base_port: int | None = None,
        no_graphics: bool = True,
        time_scale: float = 20.0,
        additional_args: list[str] | None = None,
    ):
        """See BlackOutEnv's constructor docstring for every parameter except num_arenas --
        identical meanings here. num_arenas must match the number of arenas baked into the
        connected build (ArenaDuplicator's ARENA_COUNT); a mismatch isn't detectable up front
        (nothing in the wire protocol states the total arena count), so an arenaIndex outside
        [0, num_arenas) is dropped with a one-time warning rather than raising -- see
        _route_unit/_route_map.
        """
        if num_arenas < 1:
            raise ValueError(f"num_arenas must be >= 1, got {num_arenas}")
        self.num_arenas = num_arenas

        if base_port is None:
            base_port = find_free_port()
            worker_id = 0
        self._port = base_port + worker_id

        if semantic_config_path is None:
            semantic_config_path = self._DEFAULT_CONFIG
        cfg = load_semantic_config(semantic_config_path)
        self._preprocessor = MyObsPreprocessor(cfg, n_items=cfg["n_items"], n_classes=cfg["n_classes"])
        self._map_w = map_w
        self._map_h = map_h

        self._seed_channel = SeedChannel()
        self._engine_channel = EngineConfigurationChannel()
        self._unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
            additional_args=additional_args,
            side_channels=[self._seed_channel, self._engine_channel],
        )
        self._engine_channel.set_configuration_parameters(time_scale=time_scale)

        # (behavior_name, ml_agents_agent_id) -> (arena_index, agent_name) / arena_index.
        # ML-Agents keeps agent_id stable for a unit's whole process lifetime (units are
        # repositioned/respawned as a game mechanic, never Destroy()ed/re-Instantiate()d), so
        # caching this on first sight and never re-parsing it is safe -- same assumption
        # BlackOutEnv._agent_name_cache already relies on for the single-arena case.
        self._unit_route: dict[tuple[str, int], tuple[int, str]] = {}
        self._map_route: dict[tuple[str, int], int] = {}

        # Per-arena cached state, mirroring BlackOutEnv's single-arena fields one-per-arena.
        self._team_graphics: list[dict[int, np.ndarray]] = [{} for _ in range(num_arenas)]
        self._agent_states: list[dict[int, np.ndarray]] = [{} for _ in range(num_arenas)]
        self._team_states: list[dict[int, np.ndarray]] = [{} for _ in range(num_arenas)]
        self._latest_scalars: list[dict[str, float]] = [{} for _ in range(num_arenas)]
        self._latest_winner: list[int | None] = [None] * num_arenas
        self._episode_active: list[bool] = [False] * num_arenas

        self._warned_no_arena_index = False
        self._warned_arena_index_oob = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def preprocessor(self) -> MyObsPreprocessor:
        """Exposes graphic/team_state/agent_state sizing for callers that need to allocate
        their own buffers (e.g. collect_heuristic_dataset_multiarena.py's SequentialReplayBuffer)."""
        return self._preprocessor

    def reset(self, seed: int | None = None) -> list[dict[str, dict[str, np.ndarray]]]:
        """Advances until every arena has produced at least one decision step. Returns a list
        of num_arenas obs dicts (unit_0..unit_9 keys), one per arena, in arena-index order."""
        if seed is not None:
            self._seed_channel.send_seed(seed)
        self._unity_env.reset()

        obs_by_arena: list[dict | None] = [None] * self.num_arenas
        ticks = 0
        while any(o is None for o in obs_by_arena):
            ticks += 1
            if ticks > 16:
                missing = [i for i, o in enumerate(obs_by_arena) if o is None]
                raise RuntimeError(
                    f"MultiArenaBlackOutEnv.reset(): arena(s) {missing} produced no decision "
                    f"after 16 ticks -- decisionPeriod misconfigured, Unity stalled, or "
                    f"num_arenas ({self.num_arenas}) doesn't match this build."
                )
            for i, result in enumerate(self._collect_all()):
                if result is not None and obs_by_arena[i] is None:
                    obs_by_arena[i] = result.obs
            if any(o is None for o in obs_by_arena):
                self._unity_env.step()
        return obs_by_arena

    def step(self, actions: list[dict[str, np.ndarray] | None]) -> list[ArenaStep | None]:
        """`actions[i]` is arena i's action dict for units that had a fresh decision on the
        PREVIOUS call (a None entry -- including every entry on the first call after reset()
        -- sends no action this tick for that arena, matching BlackOutEnv's implicit skip-tick
        passthrough). Returns one ArenaStep per arena that had a fresh decision or terminal
        step THIS tick, else None at that index -- callers should only act on non-None entries
        and simply leave that arena's pending action as-is otherwise (same skip-tick handling,
        generalized to "not every arena is ready every tick" instead of "the only arena is
        ready or it isn't")."""
        if len(actions) != self.num_arenas:
            raise ValueError(f"expected {self.num_arenas} action dicts, got {len(actions)}")
        self._send_actions(actions)
        self._unity_env.step()
        return self._collect_all()

    def close(self) -> None:
        self._unity_env.close()
        _kill_stray_unity_processes(self._port)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_behavior_names(self) -> list[str]:
        names = [n for n in self._unity_env.behavior_specs if n.startswith(BEHAVIOR_NAME)]
        if not names:
            raise RuntimeError(
                f"Behavior '{BEHAVIOR_NAME}' not found. Available: {list(self._unity_env.behavior_specs.keys())}"
            )
        return names

    def _resolve_map_behavior_name(self) -> str | None:
        for name in self._unity_env.behavior_specs:
            if name.startswith(MAP_BEHAVIOR_NAME):
                return name
        return None

    def _route_unit(self, behavior_name: str, agent_id: int, raw_vector: np.ndarray) -> tuple[int, str] | None:
        cached = self._unit_route.get((behavior_name, agent_id))
        if cached is not None:
            return cached
        if len(raw_vector) < 2:
            if not self._warned_no_arena_index:
                print(f"[MultiArenaBlackOutEnv] WARNING: '{behavior_name}' observation has no "
                      f"arenaIndex float (shape {raw_vector.shape}) -- connected to a "
                      f"legacy single-arena build? Dropping this agent's steps.")
                self._warned_no_arena_index = True
            return None
        unit_idx = int(round(float(raw_vector[0])))
        arena_idx = int(round(float(raw_vector[1])))
        if not (0 <= unit_idx < N_AGENTS) or not (0 <= arena_idx < self.num_arenas):
            if not self._warned_arena_index_oob:
                print(f"[MultiArenaBlackOutEnv] WARNING: '{behavior_name}' agent_id={agent_id} "
                      f"reported (unitIndex={unit_idx}, arenaIndex={arena_idx}), outside "
                      f"expected ranges (num_arenas={self.num_arenas}) -- dropping.")
                self._warned_arena_index_oob = True
            return None
        route = (arena_idx, agent_name(unit_idx))
        self._unit_route[(behavior_name, agent_id)] = route
        return route

    def _route_map(self, behavior_name: str, agent_id: int, raw_state: np.ndarray) -> int | None:
        cached = self._map_route.get((behavior_name, agent_id))
        if cached is not None:
            return cached
        if len(raw_state) < self._preprocessor.RAW_VECTOR_SIZE + 1:
            return None  # legacy single-arena build's MapObsAgent -- no arenaIndex appended
        arena_idx = int(round(float(raw_state[-1])))
        if not (0 <= arena_idx < self.num_arenas):
            return None
        self._map_route[(behavior_name, agent_id)] = arena_idx
        return arena_idx

    def _collect_map_obs_all(self) -> None:
        map_behavior = self._resolve_map_behavior_name()
        if map_behavior is None:
            return
        decision_steps, _ = self._unity_env.get_steps(map_behavior)
        for agent_id in decision_steps.agent_id:
            obs_list = decision_steps[agent_id].obs
            visual_obs = next((o for o in obs_list if o.ndim == 3), None)
            raw_state = next((o for o in obs_list if o.ndim == 1), None)
            if raw_state is None:
                continue
            arena_idx = self._route_map(map_behavior, agent_id, raw_state)
            if arena_idx is None:
                continue

            if visual_obs is not None:
                # ML-Agents delivers visual obs as (C, H, W); preprocessor expects (H, W, C).
                team_a, team_b = self._preprocessor.preprocess_team_graphics(
                    np.transpose(visual_obs, (1, 2, 0))
                )
                self._team_graphics[arena_idx][0] = team_a
                self._team_graphics[arena_idx][1] = team_b

            core = raw_state[: self._preprocessor.RAW_VECTOR_SIZE]
            self._agent_states[arena_idx][0], self._agent_states[arena_idx][1] = (
                self._preprocessor.preprocess_agent_states(core)
            )
            self._team_states[arena_idx][0], self._team_states[arena_idx][1] = (
                self._preprocessor.preprocess_team_states(core)
            )
            start = self._preprocessor.RAW_SCALAR_START
            self._latest_scalars[arena_idx] = {
                "score_0": float(core[start]),
                "score_1": float(core[start + 1]),
                "time_left": float(core[start + 2]),
                "absorption_time_left": float(core[start + 3]),
            }

    def _build_obs(self, arena_idx: int, aname: str) -> dict[str, np.ndarray]:
        team_id = 0 if int(aname.split("_")[1]) < N_TEAM_A else 1

        graphic = self._team_graphics[arena_idx].get(team_id)
        if graphic is None:
            graphic = np.zeros(
                (self._map_h, self._map_w, self._preprocessor.n_graphic_channels), dtype=np.float32
            )
        team_state = self._team_states[arena_idx].get(team_id)
        if team_state is None:
            team_state = np.zeros((self._preprocessor.team_state_size,), dtype=np.float32)
        agent_states = self._agent_states[arena_idx].get(team_id)
        if agent_states is None:
            agent_states = np.zeros((N_AGENTS, self._preprocessor.agent_state_size), dtype=np.float32)

        return {"graphic": graphic, "team_state": team_state, "agent_states": agent_states}

    def _send_actions(self, actions: list[dict[str, np.ndarray] | None]) -> None:
        map_behavior = self._resolve_map_behavior_name()
        if map_behavior is not None:
            decision_steps, _ = self._unity_env.get_steps(map_behavior)
            if len(decision_steps) > 0:
                n = len(decision_steps)
                self._unity_env.set_actions(
                    map_behavior, ActionTuple(continuous=np.zeros((n, 0), dtype=np.float32))
                )

        for behavior_name in self._resolve_behavior_names():
            decision_steps, _ = self._unity_env.get_steps(behavior_name)
            n = len(decision_steps)
            continuous = np.zeros((n, 2), dtype=np.float32)

            for i, agent_id in enumerate(decision_steps.agent_id):
                routed = self._unit_route.get((behavior_name, agent_id))
                if routed is None:
                    continue
                arena_idx, aname = routed
                arena_actions = actions[arena_idx]
                if arena_actions is not None and aname in arena_actions:
                    continuous[i] = np.clip(arena_actions[aname], -1.0, 1.0)

            self._unity_env.set_actions(behavior_name, ActionTuple(continuous=continuous))

    def _collect_all(self) -> list[ArenaStep | None]:
        """Mirrors BlackOutEnv._collect_obs/_collect_map_obs, generalized to num_arenas
        independent matches instead of one. See that method's comments for the terminal-tick
        staleness correction and physical-winner-from-score logic this replicates per arena."""
        previous_scalars = [dict(s) for s in self._latest_scalars]
        self._collect_map_obs_all()

        obs: list[dict | None] = [None] * self.num_arenas
        rewards: list[dict[str, float]] = [{} for _ in range(self.num_arenas)]
        terminations: list[dict[str, bool]] = [{} for _ in range(self.num_arenas)]
        touched = [False] * self.num_arenas

        for behavior_name in self._resolve_behavior_names():
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)

            for agent_id in decision_steps.agent_id:
                raw = decision_steps[agent_id].obs[0]
                routed = self._route_unit(behavior_name, agent_id, raw)
                if routed is None:
                    continue
                arena_idx, aname = routed
                touched[arena_idx] = True
                if obs[arena_idx] is None:
                    obs[arena_idx] = {}
                obs[arena_idx][aname] = self._build_obs(arena_idx, aname)
                rewards[arena_idx][aname] = float(decision_steps[agent_id].reward)

            for agent_id in terminal_steps.agent_id:
                raw = terminal_steps[agent_id].obs[0]
                routed = self._route_unit(behavior_name, agent_id, raw)
                if routed is None:
                    continue
                arena_idx, aname = routed
                touched[arena_idx] = True
                if obs[arena_idx] is None:
                    obs[arena_idx] = {}
                obs[arena_idx][aname] = self._build_obs(arena_idx, aname)
                rewards[arena_idx][aname] = float(terminal_steps[agent_id].reward)
                terminations[arena_idx][aname] = True

        results: list[ArenaStep | None] = [None] * self.num_arenas
        for i in range(self.num_arenas):
            if not touched[i]:
                continue

            ended = any(terminations[i].values())
            if ended and previous_scalars[i]:
                prev_time = previous_scalars[i].get("time_left", 0.0)
                cur_time = self._latest_scalars[i].get("time_left", 0.0)
                if cur_time > prev_time + 0.25:
                    self._latest_scalars[i] = previous_scalars[i]

            info = dict(self._latest_scalars[i])
            episode_started = False
            if ended:
                if self._latest_winner[i] is None:
                    score_a = self._latest_scalars[i].get("score_0", 0.0)
                    score_b = self._latest_scalars[i].get("score_1", 0.0)
                    self._latest_winner[i] = 0 if score_a > score_b else 1 if score_b > score_a else -1
                info["winner"] = self._latest_winner[i]
                self._episode_active[i] = False
            elif not self._episode_active[i]:
                self._episode_active[i] = True
                self._latest_winner[i] = None
                episode_started = True

            results[i] = ArenaStep(
                obs=obs[i], rewards=rewards[i], terminations=terminations[i],
                info=info, episode_started=episode_started,
            )
        return results
