import math

import numpy as np
import torch

from blackout_env.env.constants import unit_index
from blackout_env.model.base import BaseModel
from blackout_env.model.my_model import MyModel

# 8 compass directions, evenly spaced starting at 0 rad (+x axis), matching MyModel's
# N_DISCRETE_ACTIONS Q-head. Unit-length so they land exactly on the env's [-1, 1] action
# bounds without needing a separate clamp.
DIRECTION_VECTORS = np.array(
    [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(8)],
    dtype=np.float32,
)


class MyPolicy(BaseModel):
    """
    Wraps MyModel's per-unit Q-values as a BaseModel. graphic/team_state/agent_states are
    identical for every agent on a team (see BlackOutEnv), so the network only needs to run
    once per act() call (batch size 1) — each agent's action is then just its row of the
    resulting 10-unit Q-table, selected by unit index (blackout_env.unit_index), converted
    from an argmax direction index to a continuous (dx, dy) vector.
    """

    def __init__(self, net: MyModel, device: str | torch.device = "cpu") -> None:
        self._net = net
        self._device = torch.device(device)

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        agents = list(obs.keys())
        shared_obs = obs[agents[0]]  # same graphic/team_state/agent_states for every agent here

        graphic = (
            torch.tensor(shared_obs["graphic"], dtype=torch.float32, device=self._device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        team_state = torch.tensor(
            shared_obs["team_state"], dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        agent_states = torch.tensor(
            shared_obs["agent_states"], dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values, _ = self._net(graphic, team_state, agent_states)  # [1, N_UNITS, 8]

        best_direction = q_values.squeeze(0).argmax(dim=-1).cpu().numpy()  # [N_UNITS]

        return {agent: DIRECTION_VECTORS[best_direction[unit_index(agent)]] for agent in agents}
