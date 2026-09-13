"""
Minimal in-process stand-in for BlackOutEnv, used only by --smoke-test to validate the
actor/inference-server/learner multiprocessing wiring (queues, shared state, weight sync,
checkpointing) without a Unity build, GPU cluster, or even a GPU at all.

Not a gameplay simulator: observations/rewards are random noise, shaped exactly like the real
env's (see MyObsPreprocessor) so tensors flow through the real network code paths, and episodes
end on a fixed step count with a "absorption" pulse partway through (mimicking
team_state[ABSORPTION_IDX] snapping back up -- see qmix_trainer.QMIXTrainer.collect_step's
absorption_fired detection, which actor.py reproduces). This only proves the pipeline moves data
and shuts down correctly end to end, not that it learns anything -- always validate against the
real BlackOutEnv build before trusting a training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackout_env.env.constants import N_AGENTS, all_agents
from blackout_env.env.my_obs_preprocessor import MyObsPreprocessor
from blackout_env.env.obs_preprocessor import load_semantic_config

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "semantic_map_config.json"


class FakeBlackOutEnv:
    def __init__(
        self,
        env_path: str | None = None,  # accepted for call-site parity with BlackOutEnv; ignored
        map_w: int = 24,
        map_h: int = 24,
        n_items: int = 5,
        n_classes: int = 3,
        episode_len: int = 40,
        absorption_every: int = 12,
        seed: int = 0,
        **_ignored,
    ) -> None:
        cfg = load_semantic_config(_DEFAULT_CONFIG)
        preprocessor = MyObsPreprocessor(cfg, n_items=n_items, n_classes=n_classes)
        self._graphic_shape = (map_h, map_w, preprocessor.n_graphic_channels)
        self._agent_state_size = preprocessor.agent_state_size
        self._episode_len = episode_len
        self._absorption_every = absorption_every
        self._rng = np.random.default_rng(seed)
        self.possible_agents = all_agents()
        self.agents: list[str] = []
        self._t = 0

    def _obs(self) -> dict[str, dict[str, np.ndarray]]:
        graphic = self._rng.random(self._graphic_shape, dtype=np.float32)
        steps_to_absorption = self._absorption_every - (self._t % self._absorption_every)
        team_state = np.array(
            [0.0, 0.0, 1.0 - self._t / self._episode_len, steps_to_absorption / self._absorption_every],
            dtype=np.float32,
        )
        agent_states = self._rng.random((N_AGENTS, self._agent_state_size)).astype(np.float32)
        agent_states[:5, 2] = 1.0   # team A sign, see qmix_trainer.TEAM_SIGN_COL
        agent_states[5:, 2] = -1.0  # team B sign
        one = {"graphic": graphic, "team_state": team_state, "agent_states": agent_states}
        return {a: one for a in self.possible_agents}

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._t = 0
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        self._t += 1
        done = self._t >= self._episode_len
        obs = self._obs()
        rewards = {a: float(self._rng.normal()) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        winner = int(self._rng.integers(0, 3)) - 1 if done else None  # -1/0/1, matches "winner" semantics
        infos = {a: ({"winner": winner} if done else {}) for a in self.possible_agents}
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        pass
