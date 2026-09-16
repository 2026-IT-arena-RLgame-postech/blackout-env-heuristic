import torch
from torch import nn

from blackout_env.model.derived_obs import N_PATCH_FEATURES

from .ffn_block import SwiGLUBlock


# How much of a unit token's width each feature group gets, as relative weights rather than in
# proportion to raw column count. The raw-width rule that used to apply here handed the item
# one-hot half the budget (6 mostly-zero columns) and the position pair the smallest share of all
# -- but position, the unit's offset inside its tile and its local walkability are the continuous,
# high-entropy part of the row, and the one the corner-snag failure turns on, while item is a
# 6-way choice plus one amount and class a 3-way choice. Weights sum to 16 so they land exactly on
# multiples of hidden_size/16 (at hidden 128: 72 / 32 / 16 / 8).
SPATIAL_GROUP_WEIGHTS = {"spatial": 9, "item": 4, "class": 2, "team": 1}


def _weighted_group_sizes(
    weights: dict[str, int], hidden_size: int, leftover_to: str, quantum: int = 8, min_size: int = 8
) -> dict[str, int]:
    """
    Split hidden_size across feature groups by the given weights.

    Each share is floored to a multiple of `quantum` (matmul tiles map cleanly onto GPU/tensor-
    core boundaries at multiples of 8, so this avoids paying for oddly-shaped, inefficient Linear
    layers) and floored at `min_size` so a small group still gets a workable width. Rounding
    remainder goes to `leftover_to` so the groups sum to exactly hidden_size -- needed since they
    are concatenated back into one hidden_size-wide vector.
    """
    total = sum(weights.values())
    sizes = {
        name: max(min_size, (hidden_size * weight // total) // quantum * quantum)
        for name, weight in weights.items()
    }
    sizes[leftover_to] += hidden_size - sum(sizes.values())
    assert all(size > 0 for size in sizes.values()) and sum(sizes.values()) == hidden_size
    return sizes


class VectorEncoder(nn.Module):
    """
    Encodes MyObsPreprocessor's agent_states table and team_state summary into attention
    tokens (see blackout_env/env/my_obs_preprocessor.py:preprocess_agent_states).

    Each agent_states row is [pos(2), team(1), item_onehot(n_items+1), class_onehot(n_classes)]
    followed by the local features MyModel derives (a 3x3 walkability patch and the unit's offset
    within its tile — see model/derived_obs.py). These describe unrelated things, so each group
    gets its own small projection before the per-unit token is assembled, rather than one Linear
    over the raw concatenated row. Position and the derived local geometry are projected together
    as one "spatial" group, and the widths come from SPATIAL_GROUP_WEIGHTS.

    Output: one token per unit plus one team_state token, i.e. [B, N_UNITS + 1, hidden_size]
    (team_state token appended last, matching MyModel's trunk token layout).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_items: int = 5,
        n_classes: int = 3,
        team_state_size: int = 4,
        n_patch_features: int = N_PATCH_FEATURES,
    ) -> None:
        super().__init__()

        self.pos_dim = 2
        self.team_dim = 1
        self.item_dim = n_items + 1
        self.class_dim = n_classes
        self.patch_dim = n_patch_features
        self.spatial_dim = self.pos_dim + self.patch_dim

        group_sizes = _weighted_group_sizes(SPATIAL_GROUP_WEIGHTS, hidden_size, leftover_to="spatial")

        self.spatial_proj = nn.Linear(self.spatial_dim, group_sizes["spatial"])
        self.team_proj = nn.Linear(self.team_dim, group_sizes["team"])
        self.item_proj = nn.Linear(self.item_dim, group_sizes["item"])
        self.class_proj = nn.Linear(self.class_dim, group_sizes["class"])
        self.group_act = nn.SiLU()
        self.unit_merge = SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size)

        self.team_state_encoder = nn.Sequential(
            nn.Linear(team_state_size, hidden_size),
            nn.SiLU(),
            SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size),
        )

    def forward(self, agent_states: torch.Tensor, team_state: torch.Tensor) -> torch.Tensor:
        """
        agent_states : [B, N_UNITS, pos(2)+team(1)+item(n_items+1)+class(n_classes)+patch]
        team_state   : [B, team_state_size]

        Returns [B, N_UNITS + 1, hidden_size].
        """
        pos = agent_states[..., 0:2]
        team = agent_states[..., 2:3]
        item = agent_states[..., 3 : 3 + self.item_dim]
        cls = agent_states[..., 3 + self.item_dim : 3 + self.item_dim + self.class_dim]
        patch = agent_states[..., 3 + self.item_dim + self.class_dim :]
        spatial = torch.cat([pos, patch], dim=-1)

        unit_features = torch.cat(
            [
                self.group_act(self.spatial_proj(spatial)),
                self.group_act(self.team_proj(team)),
                self.group_act(self.item_proj(item)),
                self.group_act(self.class_proj(cls)),
            ],
            dim=-1,
        )
        unit_tokens = self.unit_merge(unit_features)  # [B, N_UNITS, hidden_size]

        team_token = self.team_state_encoder(team_state).unsqueeze(1)  # [B, 1, hidden_size]

        return torch.cat([unit_tokens, team_token], dim=1)
