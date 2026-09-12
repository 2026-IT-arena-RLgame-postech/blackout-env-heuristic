import numpy as np

from blackout_env.competition.match import run_match
from blackout_env.model.base import BaseModel


class _Policy(BaseModel):
    def act(self, obs):
        return {name: np.zeros(2, np.float32) for name in obs}


class _FakeEnv:
    possible_agents = [f"unit_{i}" for i in range(10)]

    def reset(self, seed=None):
        self.agents = self.possible_agents.copy()
        self.seed = seed
        return {a: {} for a in self.agents}, {}

    def step(self, actions):
        old_agents = self.agents.copy()
        self.agents = []
        # Team A gets misleadingly worse shaping return but wins the real game 100-20.
        rewards = {a: (-10.0 if int(a.split("_")[1]) < 5 else 10.0) for a in old_agents}
        terms = {a: True for a in old_agents}
        info = {"winner": 0, "score_0": 1.0, "score_1": 0.2}
        return {}, rewards, terms, {a: False for a in old_agents}, {a: info for a in old_agents}


def test_match_uses_terminal_winner_not_shaped_return():
    env = _FakeEnv()
    result = run_match(env, _Policy(), _Policy(), seed=42)
    assert result.winner == 0
    assert result.team_a_total_reward < result.team_b_total_reward
    assert result.model_a_score == 1.0
    assert result.model_b_score == 0.2
    assert result.seed == 42


def test_swapped_winner_and_scores_are_model_relative():
    result = run_match(_FakeEnv(), _Policy(), _Policy(), swap_teams=True)
    assert result.winner == 1
    assert result.model_a_score == 0.2
    assert result.model_b_score == 1.0
