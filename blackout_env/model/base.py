"""
Base interface that every competition model must implement.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Stateless inference interface for a team of 5 agents.

    Implement this class and pass instances to run_match().

    Example
    -------
        class MyPolicy(nn.Module):
            ...

        class MyModel(BaseModel):
            def __init__(self, checkpoint_path: str):
                self._net = load_checkpoint(MyPolicy, checkpoint_path)

            def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
                actions = {}
                for agent, agent_obs in obs.items():
                    graphic = torch.tensor(agent_obs["graphic"]).permute(2, 0, 1).unsqueeze(0)
                    team_state = torch.tensor(agent_obs["team_state"]).unsqueeze(0)
                    agent_states = torch.tensor(agent_obs["agent_states"]).unsqueeze(0)
                    with torch.no_grad():
                        action = self._net(graphic, team_state, agent_states).squeeze(0).numpy()
                    actions[agent] = action
                return actions
    """

    @abstractmethod
    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """
        Decide actions for a team's agents.

        Parameters
        ----------
        obs : dict[agent_name, {"graphic": float32[H, W, C], "team_state": float32[4],
                                 "agent_states": float32[10, 12]}]
            Preprocessed observations for this model's agents only (5 agents). `graphic` is
            this agent's team-perspective semantic map (H, W, C order — permute to (C, H, W)
            before feeding a CNN). `team_state` is
            [own_score, opp_score, episode_time_left, absorption_time_left]. `agent_states`
            covers all 10 units (both teams), one row per unit.

        Returns
        -------
        dict[agent_name, float32[2]]
            Continuous actions (dx, dy) in [-1, 1] per agent.
        """
