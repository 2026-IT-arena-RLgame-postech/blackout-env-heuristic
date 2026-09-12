"""
BlackOut PettingZoo Parallel environment wrapper.

Wraps mlagents_envs.UnityEnvironment as a PettingZoo ParallelEnv.
All 10 agents share behavior name "BlackOutUnit" with TeamId 0 (A) or 1 (B).

Graphic + vector observation optimization
-------------------------------------------
A dedicated MapObsAgent (behavior "BlackOutMap") broadcasts both the semantic map (visual)
and the shared 10-unit state + game-state scalars (vector, 44 floats) ONCE per step, instead
of each of the 10 BlackOutUnit agents redundantly resending its own near-identical copy.
Each BlackOutUnit agent's own observation shrinks to a single float (its unitIndex, used
only for action/reward routing). Python decodes MapObsAgent's broadcast once per step via
MyObsPreprocessor.preprocess_team_graphics()/preprocess_agent_states()/preprocess_team_states()
(each returns both team perspectives directly — team B is a cheap derivation of team A, not
a second independent decode) and shares the results across every agent's obs dict.

Usage
-----
    from training.env.blackout_env import BlackOutEnv

    env = BlackOutEnv(
        env_path="path/to/BlackOut.exe",      # None = connect to running Unity Editor
        semantic_config_path="semantic_map_config.json",
    )
    obs, infos = env.reset()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        # infos[agent] contains: score_0, score_1, time_left, absorption_time_left (every step)
        # and winner (0=Team A, 1=Team B, -1=draw) on the terminal step
    env.close()

Parallel usage (Ray)
--------------------
    # base_port=None (default): each worker calls find_free_port() independently,
    # so no manual port coordination is needed.
    def make_env():
        return BlackOutEnv(env_path="path/to/BlackOut.exe",
                           semantic_config_path="semantic_map_config.json")
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .constants import BEHAVIOR_NAME, MAP_BEHAVIOR_NAME, N_AGENTS, N_TEAM_A, all_agents, agent_name, unit_index
from .obs_preprocessor import load_semantic_config
from .my_obs_preprocessor import MyObsPreprocessor
from .seed_channel import SeedChannel


def find_free_port() -> int:
    """Bind to port 0 and let the OS pick an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class BlackOutEnv(ParallelEnv):
    """
    PettingZoo Parallel API wrapper for the BlackOut Unity environment.

    Observations
    ------------
    Each agent receives a dict observation:
      "graphic"       : float32[H × W × C]              this agent's team's semantic map
      "team_state"    : float32[team_state_size]        this agent's team's game-state summary
                         [own_score, opp_score, episode_time_left, absorption_time_left]
      "agent_states"  : float32[N_AGENTS, agent_state_size]  shared state table for all 10 units
                         (same array reference for every agent — see MyObsPreprocessor)

    The graphic is sourced from a shared MapObsAgent (behavior "BlackOutMap") that
    broadcasts both team textures once per step, instead of each of the 10 agents
    sending their own copy. This halves the number of visual obs transmitted over gRPC.
    team_state / agent_states are likewise computed once per step (not once per agent).

    Actions
    -------
    Each agent outputs a float32[2] continuous action (dx, dy) in [-1, 1].

    Notes
    -----
    - All 10 agents terminate simultaneously (episode ends when one team wins or time runs out).
    - Truncations are never set; all episode ends are terminations.
    - ML-Agents agent IDs are mapped to "unit_0" … "unit_9" by unitIndex order.
    """

    metadata = {"render_modes": [], "name": "blackout_v0"}

    _DEFAULT_CONFIG = Path(__file__).parent.parent / "semantic_map_config.json"

    def __init__(
        self,
        env_path: str | None,
        semantic_config_path: str | Path | None = None,
        map_w: int = 24,
        map_h: int = 24,
        worker_id: int = 0,
        base_port: int | None = None,
        no_graphics: bool = True,
        time_scale: float = 1.0,
        additional_args: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        env_path : str | None
            Path to Unity build executable. Pass None to attach to a running Unity Editor.
        semantic_config_path : str | Path | None
            Path to semantic_map_config.json. Defaults to the config bundled with
            the package. Override if your Unity build uses a different config.
        map_w, map_h : int
            Visual observation texture size. Must match resolution_scale × map_size in Unity.
        worker_id : int
            Offset added to base_port. Ignored when base_port is None (auto mode).
        base_port : int | None
            Base gRPC port for Unity communication. Actual port = base_port + worker_id.
            Pass None (default) to let the OS pick a free port automatically — recommended
            for parallel training (e.g. Ray), where each worker process calls find_free_port()
            independently and gets a unique port without manual coordination.
        no_graphics : bool
            Pass --no-graphics to Unity build. Has no effect when env_path is None.
        time_scale : float
            Unity Time.timeScale. Values > 1 speed up simulation for faster training.
            20~100 is typical for headless builds; keep at 1 when using the Editor.
        additional_args : list[str] | None
            Optional command-line arguments forwarded to a launched Unity player.
        """
        if base_port is None:
            base_port = find_free_port()
            worker_id = 0  # port is already unique; worker_id offset unnecessary

        if semantic_config_path is None:
            semantic_config_path = self._DEFAULT_CONFIG
        cfg = load_semantic_config(semantic_config_path)
        n_items = cfg["n_items"]
        n_classes = cfg["n_classes"]
        self._preprocessor = MyObsPreprocessor(cfg, n_items=n_items, n_classes=n_classes)
        self._map_w = map_w
        self._map_h = map_h

        self.possible_agents = all_agents()
        self.agents: list[str] = []

        # Build observation / action spaces
        graphic_channels = self._preprocessor.n_graphic_channels
        agent_state_size = self._preprocessor.agent_state_size
        team_state_size = self._preprocessor.team_state_size

        self._obs_space = spaces.Dict({
            "graphic": spaces.Box(
                low=0.0, high=1.0, shape=(map_h, map_w, graphic_channels), dtype=np.float32
            ),
            "team_state": spaces.Box(
                low=-1.0, high=1.0, shape=(team_state_size,), dtype=np.float32
            ),
            "agent_states": spaces.Box(
                low=-1.0, high=1.0, shape=(N_AGENTS, agent_state_size), dtype=np.float32
            ),
        })
        self._act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Connect to Unity
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

        # Internal state
        self._latest_obs: dict[str, dict[str, np.ndarray]] = {}
        self._latest_rewards: dict[str, float] = {}
        self._latest_terminations: dict[str, bool] = {}
        self._latest_winner: int | None = None   # 0 (Team A) | 1 (Team B) | -1 (draw) | None
        self._latest_scalars: dict[str, float] = {}   # score_0, score_1, time_left
        # (behavior_name, agent_id) → agent_name, built in _collect_obs, reused in _send_actions
        self._agent_name_cache: dict[tuple[str, int], str] = {}
        # Cached team graphics from MapObsAgent: 0→TeamA, 1→TeamB
        self._team_graphics: dict[int, np.ndarray] = {}
        # Cached team-perspective agent-state tables and game-state summaries, both decoded
        # once per step from MapObsAgent's shared raw state vector (see _collect_map_obs).
        # 0→TeamA perspective, 1→TeamB perspective.
        self._agent_states: dict[int, np.ndarray] = {}
        self._team_states: dict[int, np.ndarray] = {}
        # One-shot warning flags (avoids log spam on repeated calls)
        self._warned_no_map_behavior = False
        self._warned_empty_map_steps = False
        self._warned_empty_map_obs = False
        self._warned_no_visual_obs = False
        self._warned_no_state_obs = False

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict[str, Any]]:
        if seed is not None:
            self._seed_channel.send_seed(seed)
        self._unity_env.reset()
        self.agents = list(self.possible_agents)
        self._latest_rewards = {a: 0.0 for a in self.agents}
        self._latest_terminations = {a: False for a in self.agents}
        self._latest_winner = None
        self._latest_scalars = {}

        self._latest_obs = self._collect_obs()

        infos: dict[str, Any] = {a: {} for a in self.agents}
        return dict(self._latest_obs), infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        self._send_actions(actions)
        self._unity_env.step()
        obs = self._collect_obs()

        # BlackOutAgent requests a fresh decision only every `decisionPeriod` FixedUpdate
        # ticks (Unity repeats the last action via RequestAction() on the ticks in between --
        # see BlackOutAgent.cs); on those in-between ticks, BlackOutUnit's decision/terminal
        # steps are BOTH empty this exchange, so `obs` comes back {} even though the match is
        # still very much running. Keep advancing the sim tick-by-tick (no new actions to
        # send -- Unity ignores whatever's queued on a skip tick anyway, and
        # UnityEnvironment.step() auto-fills an empty action for any behavior not re-sent) until
        # a real decision or terminal round comes back, so callers never observe a skip tick as
        # a distinct env.step() result. A match-ending EndEpisode() call sends its terminal
        # AgentInfo synchronously regardless of decision-tick parity (that's the whole reason
        # BlackOutAgent suppresses its next decision post-reset), so this never has to
        # distinguish "still skipping" from "episode over" -- both end the loop via non-empty obs.
        skip_ticks = 0
        while not obs:
            skip_ticks += 1
            if skip_ticks > 16:
                raise RuntimeError(
                    "BlackOutEnv.step(): 16 consecutive FixedUpdate ticks with no BlackOutUnit "
                    "decision or terminal step -- decisionPeriod misconfigured, or Unity stalled."
                )
            self._unity_env.step()
            obs = self._collect_obs()

        rewards = dict(self._latest_rewards)
        terminations = dict(self._latest_terminations)
        truncations = {a: False for a in self.agents}
        episode_ended = any(terminations.values())
        step_info = dict(self._latest_scalars)
        if episode_ended:
            step_info["winner"] = self._latest_winner
        infos: dict[str, Any] = {a: step_info for a in self.agents}

        self._latest_obs = obs

        # Remove terminated agents from active list
        self.agents = [a for a in self.agents if not terminations.get(a, False)]

        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        pass  # no visual rendering; render_modes = []

    def close(self) -> None:
        self._unity_env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_obs(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Poll DecisionSteps and TerminalSteps from Unity, build preprocessed obs.
        Graphics, agent_states, team_states, and score/time scalars are all read once from
        MapObsAgent (see _collect_map_obs) and shared across every agent's obs dict — the
        per-agent loop below only handles reward/termination bookkeeping and routing.
        """
        obs: dict[str, dict[str, np.ndarray]] = {}
        rewards: dict[str, float] = {a: 0.0 for a in self.agents}
        terminations: dict[str, bool] = {a: False for a in self.agents}

        # Unit agents call EndEpisode immediately on game end.  Depending on ML-Agents
        # scheduling, MapObsAgent can already expose the freshly reset episode (scores=0,
        # time_left=1) in the same Python poll.  Retain the last playing-state scalars so
        # terminal infos report the match that just ended rather than the next reset.
        previous_scalars = dict(self._latest_scalars)
        self._collect_map_obs()

        for behavior_name in self._resolve_behavior_names():
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)

            for agent_id in decision_steps.agent_id:
                aname = self._extract_step(decision_steps, agent_id)
                if aname is None:
                    continue
                self._agent_name_cache[(behavior_name, agent_id)] = aname
                obs[aname] = self._build_obs(aname)
                rewards[aname] = float(decision_steps[agent_id].reward)

            for agent_id in terminal_steps.agent_id:
                aname = self._extract_step(terminal_steps, agent_id)
                if aname is None:
                    continue
                self._agent_name_cache[(behavior_name, agent_id)] = aname
                obs[aname] = self._build_obs(aname)
                r = float(terminal_steps[agent_id].reward)
                rewards[aname] = r
                terminations[aname] = True

                # Derive winner from the first terminal agent's reward.
                # unit 0~4 = Team A, 5~9 = Team B. Winner gets +1, loser -1, draw 0.
                if self._latest_winner is None:
                    is_team_a = unit_index(aname) < N_TEAM_A
                    if r > 0:
                        self._latest_winner = 0 if is_team_a else 1
                    elif r < 0:
                        self._latest_winner = 1 if is_team_a else 0
                    else:
                        self._latest_winner = -1

        self._latest_rewards = rewards
        self._latest_terminations = terminations
        if any(terminations.values()) and previous_scalars:
            previous_time = previous_scalars.get("time_left", 0.0)
            current_time = self._latest_scalars.get("time_left", 0.0)
            if current_time > previous_time + 0.25:
                self._latest_scalars = previous_scalars
        return obs

    def _collect_map_obs(self) -> None:
        """
        Fetch the graphic + shared state broadcast from MapObsAgent once per step and cache
        both team perspectives. MapObsAgent's obs_list holds a 3-D visual observation (the
        packed semantic map) and a 1-D vector observation (the 44-float shared unit/game
        state) — order isn't assumed, each is picked out by ndim.
        If MapObsAgent is not present (e.g. legacy build), cached values remain as before.
        """
        map_behavior = self._resolve_map_behavior_name()
        if map_behavior is None:
            if not self._warned_no_map_behavior:
                print(f"[BlackOutEnv] WARNING: MapObsAgent behavior not found. "
                      f"Available behaviors: {list(self._unity_env.behavior_specs.keys())}")
                self._warned_no_map_behavior = True
            return
        decision_steps, _ = self._unity_env.get_steps(map_behavior)
        if len(decision_steps) == 0:
            if not self._warned_empty_map_steps:
                print(f"[BlackOutEnv] WARNING: MapObsAgent has no decision steps (behavior='{map_behavior}'). "
                      f"Check that MapObsAgent.FixedUpdate calls RequestDecision().")
                self._warned_empty_map_steps = True
            return
        agent_id = decision_steps.agent_id[0]
        obs_list = decision_steps[agent_id].obs
        if len(obs_list) < 1:
            if not self._warned_empty_map_obs:
                print(f"[BlackOutEnv] WARNING: MapObsAgent has no observations in obs_list. "
                      f"Check that a RenderTextureSensorComponent is attached to the MapObsAgent GameObject.")
                self._warned_empty_map_obs = True
            return
        # Find the visual (3-D) and vector (1-D) observations regardless of order.
        visual_obs = None
        raw_state = None
        for o in obs_list:
            if o.ndim == 3:
                visual_obs = o
            elif o.ndim == 1:
                raw_state = o
        if visual_obs is None:
            if not self._warned_no_visual_obs:
                print(f"[BlackOutEnv] WARNING: MapObsAgent has no 3-D visual observation in obs_list. "
                      f"Shapes: {[o.shape for o in obs_list]}. "
                      f"Ensure RenderTextureSensorComponent is attached with Grayscale=true.")
                self._warned_no_visual_obs = True
        else:
            # ML-Agents delivers visual obs as (C, H, W); preprocessor expects (H, W, C)
            team_a, team_b = self._preprocessor.preprocess_team_graphics(
                np.transpose(visual_obs, (1, 2, 0))
            )
            self._team_graphics[0] = team_a
            self._team_graphics[1] = team_b

        if raw_state is None:
            if not self._warned_no_state_obs:
                print(f"[BlackOutEnv] WARNING: MapObsAgent has no 1-D vector observation in obs_list. "
                      f"Shapes: {[o.shape for o in obs_list]}. "
                      f"Check MapObsAgent.CollectObservations / Behavior Parameters Vector Observation size.")
                self._warned_no_state_obs = True
        else:
            self._agent_states[0], self._agent_states[1] = self._preprocessor.preprocess_agent_states(raw_state)
            self._team_states[0], self._team_states[1] = self._preprocessor.preprocess_team_states(raw_state)
            start = self._preprocessor.RAW_SCALAR_START
            self._latest_scalars = {
                "score_0": float(raw_state[start]),
                "score_1": float(raw_state[start + 1]),
                "time_left": float(raw_state[start + 2]),
                "absorption_time_left": float(raw_state[start + 3]),
            }

    def _extract_step(self, steps, agent_id: int) -> str | None:
        """
        Extract this BlackOutUnit's agent_name from its (now 1-float, unitIndex-only) raw
        observation. Position/item/class/graphic all come from MapObsAgent via
        _collect_map_obs() instead — this is routing information only.
        """
        raw_vector: np.ndarray = steps[agent_id].obs[0]

        unit_idx = int(round(float(raw_vector[0])))
        if not (0 <= unit_idx < N_AGENTS):
            return None

        return agent_name(unit_idx)

    def _build_obs(self, aname: str) -> dict[str, np.ndarray]:
        """Assembles this agent's obs dict from the shared, once-per-step precomputed pieces."""
        team_id = 0 if unit_index(aname) < N_TEAM_A else 1

        graphic = self._team_graphics.get(team_id)
        if graphic is None:
            graphic = np.zeros(
                (self._map_h, self._map_w, self._preprocessor.n_graphic_channels),
                dtype=np.float32,
            )

        team_state = self._team_states.get(team_id)
        if team_state is None:
            team_state = np.zeros((self._preprocessor.team_state_size,), dtype=np.float32)

        agent_states = self._agent_states.get(team_id)
        if agent_states is None:
            agent_states = np.zeros(
                (N_AGENTS, self._preprocessor.agent_state_size), dtype=np.float32
            )

        return {"graphic": graphic, "team_state": team_state, "agent_states": agent_states}

    def _send_actions(self, actions: dict[str, np.ndarray]) -> None:
        # Send empty actions to MapObsAgent (it has no real actions, Continuous Actions = 0)
        map_behavior = self._resolve_map_behavior_name()
        if map_behavior is not None:
            decision_steps, _ = self._unity_env.get_steps(map_behavior)
            if len(decision_steps) > 0:
                n = len(decision_steps)
                self._unity_env.set_actions(
                    map_behavior,
                    ActionTuple(continuous=np.zeros((n, 0), dtype=np.float32)),
                )

        for behavior_name in self._resolve_behavior_names():
            decision_steps, _ = self._unity_env.get_steps(behavior_name)

            n = len(decision_steps)
            continuous = np.zeros((n, 2), dtype=np.float32)

            for i, agent_id in enumerate(decision_steps.agent_id):
                aname = self._agent_name_cache.get((behavior_name, agent_id))
                if aname is not None and aname in actions:
                    continuous[i] = np.clip(actions[aname], -1.0, 1.0)

            action_tuple = ActionTuple(continuous=continuous)
            self._unity_env.set_actions(behavior_name, action_tuple)

    def _resolve_behavior_names(self) -> list[str]:
        """
        Returns all registered BlackOutUnit behavior names.
        In Unity Editor, names get a '?team=0' / '?team=1' suffix.
        MapObsAgent behavior is excluded.
        """
        names = [
            n for n in self._unity_env.behavior_specs
            if n.startswith(BEHAVIOR_NAME)
        ]
        if not names:
            raise RuntimeError(
                f"Behavior '{BEHAVIOR_NAME}' not found in environment. "
                f"Available: {list(self._unity_env.behavior_specs.keys())}"
            )
        return names

    def _resolve_map_behavior_name(self) -> str | None:
        """Returns the registered MapObsAgent behavior name, or None if not present."""
        for name in self._unity_env.behavior_specs:
            if name.startswith(MAP_BEHAVIOR_NAME):
                return name
        return None
