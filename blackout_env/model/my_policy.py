import math

import numpy as np
import torch

from blackout_env.env.constants import unit_index
from blackout_env.env.team_frame import (
    canonical_obs,
    canonical_unit_row,
    is_team_b_view,
    mirror_direction_idx,
)
from blackout_env.model.action_mask import masked_greedy
from blackout_env.model.base import BaseModel
from blackout_env.model.my_model import MyModel

# 8 compass directions, evenly spaced starting at 0 rad (+x axis), matching MyModel's
# N_DISCRETE_ACTIONS Q-head. Unit-length so they land exactly on the env's [-1, 1] action
# bounds without needing a separate clamp.
DIRECTION_VECTORS = np.array(
    [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(8)],
    dtype=np.float32,
)


def direction_vector_to_idx(vectors: np.ndarray) -> np.ndarray:
    """
    Snaps arbitrary-angle (dx, dy) vectors to the index of the nearest DIRECTION_VECTORS
    compass direction, by max dot product (= cosine similarity, since every DIRECTION_VECTORS
    row is unit-length). Used to convert a heuristic policy's continuous action into the same
    discrete action space MyModel's Q-head uses, so heuristic-generated transitions can be
    stored in and trained on by the replay buffer (see QMIXTrainer's heuristic bootstrap).

    vectors: [..., 2], any leading shape (e.g. [N_TEAM, 2] -> [N_TEAM]).

    A zero vector ties every dot product at 0.0 and argmax deterministically picks index 0 --
    not a new issue this introduces: N_DISCRETE_ACTIONS has no dedicated "stay" action for the
    trained policy either, so both sides of the bootstrap already share this same constraint.
    """
    dots = vectors @ DIRECTION_VECTORS.T  # [..., 8]
    return dots.argmax(axis=-1)


class MyPolicy(BaseModel):
    """
    Wraps MyModel's per-unit Q-values as a BaseModel. graphic/team_state/agent_states are
    identical for every agent on a team (see BlackOutEnv), so the network only needs to run
    once per act() call (batch size 1) — each agent's action is then just its row of the
    resulting 10-unit Q-table, selected by unit index (blackout_env.unit_index), converted
    from an argmax direction index to a continuous (dx, dy) vector.
    """

    def __init__(self, net: MyModel, device: str | torch.device = "cpu", mask_walls: bool = False) -> None:
        self._net = net
        self._device = torch.device(device)
        # Keep in step with QMIXConfig.action_masking: a policy evaluated without the mask its
        # targets were computed under is a different policy (see model/action_mask.py).
        self._mask_walls = mask_walls

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        agents = list(obs.keys())
        shared_obs = obs[agents[0]]  # same graphic/team_state/agent_states for every agent here
        # Team B is trained on -- and must therefore act on -- the mirrored view, where its own
        # units sit in rows 0-4 (see blackout_env.env.team_frame). Team A's view is already
        # canonical and passes through untouched.
        team_b = is_team_b_view(shared_obs["agent_states"])
        shared_obs = canonical_obs(shared_obs)

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
            q_values, *_ = self._net(graphic, team_state, agent_states)  # [1, N_UNITS, 8]

        if self._mask_walls:
            best_direction = masked_greedy(q_values, graphic, agent_states).squeeze(0).cpu().numpy()
        else:
            best_direction = q_values.squeeze(0).argmax(dim=-1).cpu().numpy()  # [N_UNITS]
        if team_b:
            best_direction = mirror_direction_idx(best_direction)  # canonical -> world

        return {
            agent: DIRECTION_VECTORS[best_direction[canonical_unit_row(unit_index(agent), team_b)]]
            for agent in agents
        }
