import torch
from torch import nn

from blackout_env.model.modules import (
    AttentionLayers,
    GraphicEncoder,
    RotaryEmbedding2D,
    SwiGLUBlock,
    VectorEncoder,
    build_grid_position_ids,
)

GRID_H = GRID_W = 6  # GraphicEncoder's final spatial size (24x24 input, two stride-2 convs)
N_VISION_TOKENS = GRID_H * GRID_W  # 36
N_UNITS = 10
N_TEAM_STATE_TOKENS = 1
N_DISCRETE_ACTIONS = 8  # 8 compass directions, see my_policy.py:DIRECTION_VECTORS
N_ATTENTION_HEADS = 8
ATTENTION_DEPTH = 12

# Token-type ids for the trunk sequence: which tokens are "graphic", "unit", or "global"
# (team_state) — see MyModel's token_type_emb.
TYPE_GRAPHIC, TYPE_UNIT, TYPE_GLOBAL = 0, 1, 2


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

    Token identity
    --------------
    The 47 trunk tokens (36 vision + 10 unit + 1 team_state) come from three different
    encoders and have no positional signal of their own once concatenated, so the trunk adds
    two things before attention:
      - a learned token-type embedding (graphic / unit / global), added to every token, so
        attention can tell which "kind" of token it's looking at;
      - 2D RoPE applied only to the 36 graphic tokens (unit/global tokens get row=col=0,
        which is a no-op rotation — see RotaryEmbedding2D), so attention can additionally
        tell *where in the 6x6 map* a graphic token came from, and reason about relative
        spatial offsets between them.

    The trunk's attention layers use Exclusive Self Attention (XSA, exclusive_attention=True
    by default — see GroupedQueryAttention) instead of standard SA: each position's attention
    output has its component along that position's own value vector removed, so attention
    can't just re-derive point-wise (FFN-like) features and is pushed to spend its capacity on
    context aggregation instead.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_items: int = 5,
        n_classes: int = 3,
        team_state_size: int = 4,
        n_actions: int = N_DISCRETE_ACTIONS,
        exclusive_attention: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # encoders
        self.graphic_encoder = GraphicEncoder(hidden_size)
        self.vector_encoder = VectorEncoder(
            hidden_size, n_items=n_items, n_classes=n_classes, team_state_size=team_state_size
        )

        # main trunk: [36 vision] + [10 units] + [1 team_state] = 47 tokens
        self.attention = AttentionLayers(
            hidden_size, N_ATTENTION_HEADS, ATTENTION_DEPTH, exclusive=exclusive_attention
        )

        # token-type embedding: one row per type, gathered by a fixed per-position type id.
        n_vector_tokens = N_UNITS + N_TEAM_STATE_TOKENS
        token_type_ids = torch.tensor(
            [TYPE_GRAPHIC] * N_VISION_TOKENS + [TYPE_UNIT] * N_UNITS + [TYPE_GLOBAL] * N_TEAM_STATE_TOKENS,
            dtype=torch.long,
        )
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)
        self.token_type_emb = nn.Embedding(3, hidden_size)

        # 2D RoPE cos/sin for the trunk sequence, precomputed once (deterministic given the
        # fixed grid size + token layout above, so there's no need to recompute it per call).
        head_dim = hidden_size // N_ATTENTION_HEADS
        rope = RotaryEmbedding2D(head_dim)
        row_ids, col_ids = build_grid_position_ids(GRID_H, GRID_W, n_vector_tokens)
        rope_cos, rope_sin = rope(row_ids, col_ids)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

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
        tokens_in = tokens_in + self.token_type_emb(self.token_type_ids)[None, :, :]

        tokens_out = self.attention(tokens_in, rope=(self.rope_cos, self.rope_sin))

        vis_out = tokens_out[:, :N_VISION_TOKENS, :]
        unit_out = tokens_out[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS, :]

        vision_latent = self.spr_head(vis_out)
        q_values = self.q_head(unit_out)

        return q_values, vision_latent
