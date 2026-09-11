import torch
from torch import nn

from blackout_env.model.modules import AttentionLayers, GraphicEncoder, SwiGLUBlock, VectorEncoder

N_VISION_TOKENS = 36  # GraphicEncoder's 6x6 spatial tokens, see graphic_encoder.py
N_UNITS = 10
N_DISCRETE_ACTIONS = 8  # 8 compass directions, see my_policy.py:DIRECTION_VECTORS


class MyModel(nn.Module):
    """
    DQN-style trunk. Encodes the graphic + agent_states/team_state obs into tokens, runs a
    shared self-attention trunk over all of them, and outputs one 8-way (compass direction)
    Q-value row per unit — for both teams, since agent_states covers all 10 units.

    This module only produces Q-values; it doesn't know which unit is "self". graphic /
    team_state / agent_states are identical for every agent on a team (see BlackOutEnv), so
    a single forward call already covers all 5 of a team's agents — MyPolicy.act() picks
    each agent's row out of the 10 by unit index (blackout_env.unit_index) and turns its
    argmax direction into the continuous (dx, dy) action the env expects.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_items: int = 5,
        n_classes: int = 3,
        team_state_size: int = 4,
        n_actions: int = N_DISCRETE_ACTIONS,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # encoders
        self.graphic_encoder = GraphicEncoder(hidden_size)
        self.vector_encoder = VectorEncoder(
            hidden_size, n_items=n_items, n_classes=n_classes, team_state_size=team_state_size
        )

        # main trunk: [36 vision] + [10 units] + [1 team_state] = 47 tokens
        self.attention = AttentionLayers(hidden_size, 8, 12)

        # auxiliary vision representation head (e.g. for a self-predictive/SPR-style aux loss)
        self.spr_head = SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size)

        self.q_head = nn.Sequential(
            SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(
        self,
        graphic: torch.Tensor,
        team_state: torch.Tensor,
        agent_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        graphic      : [B, C, H, W]
        team_state   : [B, team_state_size]
        agent_states : [B, N_UNITS, agent_state_size]

        Returns
        -------
        q_values      : [B, N_UNITS, n_actions]           per-unit Q-value over the 8 directions
        vision_latent : [B, N_VISION_TOKENS, hidden_size]  auxiliary vision representation
        """
        vis_tokens = self.graphic_encoder(graphic)                   # [B, 36, hidden]
        vec_tokens = self.vector_encoder(agent_states, team_state)   # [B, 11, hidden]

        tokens_in = torch.cat((vis_tokens, vec_tokens), dim=1)       # [B, 47, hidden]
        tokens_out = self.attention(tokens_in)

        vis_out = tokens_out[:, :N_VISION_TOKENS, :]
        unit_out = tokens_out[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS, :]

        vision_latent = self.spr_head(vis_out)
        q_values = self.q_head(unit_out)

        return q_values, vision_latent
