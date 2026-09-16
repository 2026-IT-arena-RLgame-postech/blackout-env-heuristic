import torch
from torch import nn

from blackout_env.model.modules import (
    AttentionLayers,
    GraphicEncoder,
    IQNHead,
    RotaryEmbedding2D,
    SwiGLUBlock,
    VectorEncoder,
    build_grid_position_ids,
)
from blackout_env.model.unit_channels import unit_channels

N_QUANTILES_DEFAULT = 32  # quantile samples used when the caller doesn't ask for a specific count

GRID_H = GRID_W = 6  # GraphicEncoder's final spatial size (24x24 input, two stride-2 convs)
N_VISION_TOKENS = GRID_H * GRID_W  # 36
N_UNITS = 10
N_TEAM_STATE_TOKENS = 1
N_SPR_CLS_TOKENS = 1  # dedicated learned query token for the SPR aux loss -- see class docstring
N_DISCRETE_ACTIONS = 8  # 8 compass directions, see my_policy.py:DIRECTION_VECTORS
N_ATTENTION_HEADS = 8
ATTENTION_DEPTH = 4

# Token-type ids for the trunk sequence: which tokens are "graphic", "unit", "global"
# (team_state), or the dedicated SPR CLS token — see MyModel's token_type_emb.
TYPE_GRAPHIC, TYPE_UNIT, TYPE_GLOBAL, TYPE_SPR_CLS = 0, 1, 2, 3


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
    three things before attention:
      - a learned token-type embedding (graphic / unit / global), added to every token, so
        attention can tell which "kind" of token it's looking at;
      - 2D RoPE applied only to the 36 graphic tokens (unit/global tokens get row=col=0,
        which is a no-op rotation — see RotaryEmbedding2D), so attention can additionally
        tell *where in the 6x6 map* a graphic token came from, and reason about relative
        spatial offsets between them;
      - a learned per-slot identity embedding (vec_slot_emb) over the 10 unit tokens + 1
        team_state token, added to vec_tokens before the type embedding. Without this, two
        units with identical agent_states (e.g. stacked at the same spawn point) are
        genuinely indistinguishable to a permutation-equivariant attention trunk -- same
        input, same output, same action, every step, with no way for training to ever break
        the tie (observed in practice: a whole team moving in perfect lockstep the entire
        episode). The slot embedding is keyed on fixed unit index (0-9), not on content, so
        it always tells two units apart even when their observed state coincides.

    The trunk's attention layers use Exclusive Self Attention (XSA, exclusive_attention=True
    by default — see GroupedQueryAttention) instead of standard SA: each position's attention
    output has its component along that position's own value vector removed, so attention
    can't just re-derive point-wise (FFN-like) features and is pushed to spend its capacity on
    context aggregation instead.

    SPR CLS token
    -------------
    The SPR aux loss (see spr_head/QMIXTrainer._forward_and_loss) used to run off a plain
    mean-pool over the 36 vision tokens' post-attention output. Two problems with that: (1) a
    mean is a fixed, non-learned aggregation -- every graphic token contributes an identical
    1/36 weight regardless of whether it's actually informative, unlike attention's learned,
    input-dependent weighting; (2) backprop through a mean divides each individual token's
    gradient contribution by 36, diluting whatever signal reaches graphic_encoder before it
    even gets there (observed in practice: grad_norm/graphic_encoder decayed ~7 orders of
    magnitude over a 200k-step offline pretrain run while every other component's grad_norm
    stayed flat -- see docs/offline_pretrain_runs.md). A dedicated learned query token (in the
    ViT/BERT [CLS] sense) added to the trunk sequence lets attention itself decide, per
    example, which graphic tokens matter and pull from them directly with a learned weight
    instead of a fixed uniform average -- spr_head now runs on this one token's post-attention
    output instead of a mean over vis_out.
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

        # main trunk: [36 vision] + [10 units] + [1 team_state] + [1 spr_cls] = 48 tokens
        self.attention = AttentionLayers(
            hidden_size, N_ATTENTION_HEADS, ATTENTION_DEPTH, exclusive=exclusive_attention
        )

        # token-type embedding: one row per type, gathered by a fixed per-position type id.
        n_vector_tokens = N_UNITS + N_TEAM_STATE_TOKENS
        token_type_ids = torch.tensor(
            [TYPE_GRAPHIC] * N_VISION_TOKENS
            + [TYPE_UNIT] * N_UNITS
            + [TYPE_GLOBAL] * N_TEAM_STATE_TOKENS
            + [TYPE_SPR_CLS] * N_SPR_CLS_TOKENS,
            dtype=torch.long,
        )
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)
        self.token_type_emb = nn.Embedding(4, hidden_size)

        # Dedicated learned query token for the SPR aux loss (ViT/BERT [CLS]-style) -- pure
        # nn.Embedding lookup, no data-dependent content of its own (unlike vis_tokens/
        # vec_tokens), so its only job is to let attention pull whatever graphic info is
        # actually useful into one slot. See class docstring "SPR CLS token".
        self.spr_cls_emb = nn.Embedding(N_SPR_CLS_TOKENS, hidden_size)
        spr_cls_ids = torch.arange(N_SPR_CLS_TOKENS, dtype=torch.long)
        self.register_buffer("spr_cls_ids", spr_cls_ids, persistent=False)

        # Per-slot identity embedding for the 10 unit tokens + 1 team_state token: unit tokens
        # get no positional signal from RoPE (row=col=0, a no-op -- see class docstring) and
        # token_type_emb only tells attention "this is *a* unit token", identical for all 10.
        # Two units with identical agent_states rows (e.g. stacked at the same spawn point)
        # therefore produced byte-identical Q-values and thus always chose the same action --
        # a self-reinforcing lockstep with no way to break symmetry, since self-attention is
        # permutation-equivariant over tokens with identical content. This embedding is added
        # per fixed slot (unit index 0-9, independent of that unit's state) so two units are
        # always distinguishable even when their observed state coincides exactly.
        self.vec_slot_emb = nn.Embedding(n_vector_tokens, hidden_size)
        vec_slot_ids = torch.arange(n_vector_tokens, dtype=torch.long)
        self.register_buffer("vec_slot_ids", vec_slot_ids, persistent=False)

        # 2D RoPE cos/sin for the trunk sequence, precomputed once (deterministic given the
        # fixed grid size + token layout above, so there's no need to recompute it per call).
        # n_extra_tokens covers every non-spatial token (unit + team_state + spr_cls) -- all
        # get row=col=0 (no-op rotation, see build_grid_position_ids), spr_cls included since
        # it has no spatial position of its own either.
        head_dim = hidden_size // N_ATTENTION_HEADS
        rope = RotaryEmbedding2D(head_dim)
        n_extra_tokens = n_vector_tokens + N_SPR_CLS_TOKENS
        row_ids, col_ids = build_grid_position_ids(GRID_H, GRID_W, n_extra_tokens)
        rope_cos, rope_sin = rope(row_ids, col_ids)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # auxiliary vision representation head (e.g. for a self-predictive/SPR-style aux loss)
        self.spr_head = SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size)

        # IQN (arXiv:1806.06923) distributional Q-head, chosen over C51 so the value range
        # doesn't need a hand-tuned [Vmin, Vmax] that would need re-tuning whenever the
        # reward balance changes (see reward_config.json) — see modules/iqn_head.py.
        self.q_head = IQNHead(hidden_size, n_actions, n_cos=64)

    def forward(
        self,
        graphic: torch.Tensor,
        team_state: torch.Tensor,
        agent_states: torch.Tensor,
        n_quantiles: int = N_QUANTILES_DEFAULT,
        tau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        graphic      : [B, C, H, W]
        team_state   : [B, team_state_size]
        agent_states : [B, N_UNITS, agent_state_size]
        n_quantiles  : number of IQN quantile samples to draw (ignored if `tau` is given)
        tau          : [B, n_quantiles] fixed quantile fractions to use instead of sampling
                       (e.g. so an online/target pair can share draws where that matters)

        Returns
        -------
        q_values        : [B, N_UNITS, n_actions]                  mean-over-quantiles Q value
                           per unit/action — what action selection (argmax) and QMIX's chosen-
                           action gather use.
        quantile_values : [B, N_UNITS, n_quantiles, n_actions]      raw IQN quantile values,
                           for the distributional (quantile Huber) training loss.
        tau             : [B, n_quantiles]                          the quantile fractions
                           quantile_values was evaluated at (needed for the loss's asymmetric
                           weighting).
        vision_latent   : [B, hidden_size]                          SPR CLS token's post-
                           attention output through spr_head -- a learned (not mean-pooled)
                           summary of whatever graphic info attention found relevant, used by
                           the SPR aux loss (see class docstring "SPR CLS token").
        global_latent   : [B, hidden_size]                          attention-refined team_state
                           token, used as QMixer's hypernetwork input.
        """
        B = graphic.shape[0]
        # Units are absent from the env's graphic channels, so paint them in here (ally/enemy
        # counts + carried battery) -- derived from agent_states, which every caller already
        # passes, so no stored observation has to change. See model/unit_channels.py.
        graphic = torch.cat((graphic, unit_channels(agent_states, graphic.shape[-2], graphic.shape[-1])), dim=1)
        vis_tokens = self.graphic_encoder(graphic)                   # [B, 36, hidden]
        vec_tokens = self.vector_encoder(agent_states, team_state)   # [B, 11, hidden]
        vec_tokens = vec_tokens + self.vec_slot_emb(self.vec_slot_ids)[None, :, :]
        spr_cls_tok = self.spr_cls_emb(self.spr_cls_ids)[None, :, :].expand(B, -1, -1)  # [B, 1, hidden]

        tokens_in = torch.cat((vis_tokens, vec_tokens, spr_cls_tok), dim=1)  # [B, 48, hidden]
        tokens_in = tokens_in + self.token_type_emb(self.token_type_ids)[None, :, :]

        tokens_out = self.attention(tokens_in, rope=(self.rope_cos, self.rope_sin))

        unit_out = tokens_out[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS, :]
        global_out = tokens_out[:, N_VISION_TOKENS + N_UNITS, :]  # the single TYPE_GLOBAL (team_state) token
        spr_cls_out = tokens_out[:, -1, :]  # the single TYPE_SPR_CLS token, always last

        vision_latent = self.spr_head(spr_cls_out)
        quantile_values, tau = self.q_head(unit_out, n_quantiles=n_quantiles, tau=tau)
        q_values = quantile_values.mean(dim=2)  # [B, N_UNITS, n_actions]

        return q_values, quantile_values, tau, vision_latent, global_out
