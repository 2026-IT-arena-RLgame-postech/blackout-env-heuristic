import torch
from torch import nn

from .ffn_block import SwiGLUBlock


def _proportional_group_sizes(
    raw_dims: dict[str, int], hidden_size: int, leftover_to: str, quantum: int = 8, min_size: int = 8
) -> dict[str, int]:
    """
    Split hidden_size across feature groups proportionally to their raw input width, instead
    of giving every group an equal share regardless of how much information it actually
    carries (e.g. team's raw 1 float getting the same projection width as item's raw 6).

    Each share is floored to a multiple of `quantum` (matmul tiles map cleanly onto GPU/tensor-
    core boundaries at multiples of 8, so this avoids paying for oddly-shaped, inefficient
    Linear layers) and floored at `min_size` so a 1-float group still gets a workable width.
    Rounding remainder goes to `leftover_to` (rather than whichever group happens to be
    biggest) so the groups sum to exactly hidden_size — needed since they're concatenated
    back into one hidden_size-wide vector.
    """
    total_raw = sum(raw_dims.values())
    sizes = {
        name: max(min_size, (hidden_size * dim // total_raw) // quantum * quantum)
        for name, dim in raw_dims.items()
    }
    leftover = hidden_size - sum(sizes.values())
    sizes[leftover_to] += leftover
    assert all(s > 0 for s in sizes.values()) and sum(sizes.values()) == hidden_size
    return sizes


class VectorEncoder(nn.Module):
    """
    Encodes MyObsPreprocessor's agent_states table and team_state summary into attention
    tokens (see blackout_env/env/my_obs_preprocessor.py:preprocess_agent_states).

    Each agent_states row is [pos(2), team(1), item_onehot(n_items+1), class_onehot(n_classes)]
    — four feature groups that describe unrelated things (continuous position, a +-1 team
    sign, and two categorical one-hots), so each gets its own small projection before the
    per-unit token is assembled, rather than one Linear over the raw concatenated row. Each
    projection's output width is sized proportionally to that group's raw dimensionality (see
    _proportional_group_sizes) rather than an equal 1/4 split, so e.g. the 1-float team sign
    doesn't get the same capacity budget as the 6-wide item one-hot.

    Output: one token per unit plus one team_state token, i.e. [B, N_UNITS + 1, hidden_size]
    (team_state token appended last, matching MyModel's trunk token layout).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_items: int = 5,
        n_classes: int = 3,
        team_state_size: int = 4,
    ) -> None:
        super().__init__()

        self.pos_dim = 2
        self.team_dim = 1
        self.item_dim = n_items + 1
        self.class_dim = n_classes

        group_sizes = _proportional_group_sizes(
            {"pos": self.pos_dim, "team": self.team_dim, "item": self.item_dim, "class": self.class_dim},
            hidden_size,
            leftover_to="team",
        )

        self.pos_proj = nn.Linear(self.pos_dim, group_sizes["pos"])
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
        agent_states : [B, N_UNITS, pos(2)+team(1)+item(n_items+1)+class(n_classes)]
        team_state   : [B, team_state_size]

        Returns [B, N_UNITS + 1, hidden_size].
        """
        pos = agent_states[..., 0:2]
        team = agent_states[..., 2:3]
        item = agent_states[..., 3 : 3 + self.item_dim]
        cls = agent_states[..., 3 + self.item_dim : 3 + self.item_dim + self.class_dim]

        unit_features = torch.cat(
            [
                self.group_act(self.pos_proj(pos)),
                self.group_act(self.team_proj(team)),
                self.group_act(self.item_proj(item)),
                self.group_act(self.class_proj(cls)),
            ],
            dim=-1,
        )
        unit_tokens = self.unit_merge(unit_features)  # [B, N_UNITS, hidden_size]

        team_token = self.team_state_encoder(team_state).unsqueeze(1)  # [B, 1, hidden_size]

        return torch.cat([unit_tokens, team_token], dim=1)
