"""Terminal score/winner reporting of BlackOutEnv against a scripted fake UnityEnvironment.

A match-ending deposit resets the match inside the same Unity step, so the terminal MapObsAgent
broadcast of a build without MapObsAgent.CaptureTerminalState carries the reset state; builds with
it replay the frozen final state.  Both must report the true winner of an overtaking finish.
"""
import numpy as np
import pytest

import blackout_env.env.blackout_env as env_module
from blackout_env.env.blackout_env import BlackOutEnv
from blackout_env.env.constants import BEHAVIOR_NAME, MAP_BEHAVIOR_NAME, N_AGENTS, N_TEAM_A, all_agents
from blackout_env.env.my_obs_preprocessor import MyObsPreprocessor

MAP_HW = 24


def _raw_state(score_a: float, score_b: float, time_left: float, absorption_left: float = 0.5) -> np.ndarray:
    raw = np.zeros(MyObsPreprocessor.RAW_VECTOR_SIZE, np.float32)
    raw[MyObsPreprocessor.RAW_SCALAR_START:] = (score_a, score_b, time_left, absorption_left)
    return raw


class _Step:
    def __init__(self, obs, reward):
        self.obs, self.reward = obs, reward


class _Steps:
    def __init__(self, steps: dict[int, _Step]):
        self._steps = steps
        self.agent_id = np.array(list(steps), dtype=np.int32)

    def __len__(self):
        return len(self._steps)

    def __getitem__(self, agent_id):
        return self._steps[int(agent_id)]


def _frame(raw: np.ndarray, terminal: bool, rewards=None) -> dict:
    """One Unity exchange: the map broadcast plus all 10 units as decision or terminal steps."""
    rewards = rewards or [0.0] * N_AGENTS
    units = _Steps({i: _Step([np.array([i], np.float32)], rewards[i]) for i in range(N_AGENTS)})
    empty = _Steps({})
    map_steps = _Steps({0: _Step([np.zeros((1, MAP_HW, MAP_HW), np.float32), raw], 0.0)})
    return {MAP_BEHAVIOR_NAME: (map_steps, empty), BEHAVIOR_NAME: (empty, units) if terminal else (units, empty)}


class _FakeUnity:
    frames: list[dict] = []

    def __init__(self, **_):
        self.behavior_specs = {BEHAVIOR_NAME: None, MAP_BEHAVIOR_NAME: None}
        self._i = 0

    def reset(self):
        self._i = 0

    def step(self):
        self._i += 1

    def get_steps(self, behavior_name):
        return self.frames[self._i][behavior_name]

    def set_actions(self, *_):
        pass


def _play(monkeypatch, frames: list[dict]) -> dict:
    monkeypatch.setattr(env_module, "UnityEnvironment", _FakeUnity)
    monkeypatch.setattr(_FakeUnity, "frames", frames)
    env = BlackOutEnv(env_path=None, base_port=5005, unity_shaping=False)
    env.reset()
    actions = {a: np.zeros(2, np.float32) for a in env.agents}
    _, _, terminations, _, infos = env.step(actions)
    assert all(terminations.values()) and not env.agents
    return infos[all_agents()[0]]


# Team A trails 98-99 and wins by depositing 3: Unity's ±1 terminal bonus.
_WIN_A = [1.0] * N_TEAM_A + [-1.0] * (N_AGENTS - N_TEAM_A)


def test_snapshot_build_reports_final_score_of_overtaking_win(monkeypatch):
    info = _play(monkeypatch, [
        _frame(_raw_state(0.98, 0.99, 0.9), terminal=False),
        _frame(_raw_state(1.01, 0.99, 0.8999), terminal=True, rewards=_WIN_A),
    ])
    assert info["winner"] == 0
    assert info["score_0"] == pytest.approx(1.01)
    assert info["score_1"] == pytest.approx(0.99)


def test_legacy_reset_broadcast_still_names_overtaking_winner(monkeypatch):
    # Pre-snapshot build: the terminal broadcast is already the next match's reset.
    info = _play(monkeypatch, [
        _frame(_raw_state(0.98, 0.99, 0.9), terminal=False),
        _frame(_raw_state(0.0, 0.0, 1.0, 1.0), terminal=True, rewards=_WIN_A),
    ])
    assert info["winner"] == 0
    assert info["score_0"] == pytest.approx(0.98)  # last playing-state scores, not the reset
    assert info["time_left"] == pytest.approx(0.9)


def test_legacy_time_expiry_keeps_score_winner(monkeypatch):
    # Time ran out with B ahead; nonzero terminal rewards (shaping residue) must not override it.
    shaped = [0.3] * N_TEAM_A + [-0.3] * (N_AGENTS - N_TEAM_A)
    info = _play(monkeypatch, [
        _frame(_raw_state(0.40, 0.55, 5e-5), terminal=False),
        _frame(_raw_state(0.0, 0.0, 1.0, 1.0), terminal=True, rewards=shaped),
    ])
    assert info["winner"] == 1


def test_legacy_time_expiry_draw(monkeypatch):
    info = _play(monkeypatch, [
        _frame(_raw_state(0.5, 0.5, 5e-5), terminal=False),
        _frame(_raw_state(0.0, 0.0, 1.0, 1.0), terminal=True),
    ])
    assert info["winner"] == -1
