import torch
from torch import nn

from blackout_env.model.modules import (
    AttentionLayers,
    GraphicEncoder,
    IQNHead,
    LogitsHead,
    MapCrossAttention,
    RotaryEmbedding2D,
    SwiGLUBlock,
    VectorEncoder,
    build_grid_position_ids,
)
from blackout_env.model.derived_obs import N_PATCH_FEATURES, derived_map_channels, local_wall_features
from blackout_env.model.modules.logits_head import arc_bins_of_scale, state_from_iqn
from blackout_env.model.modules.map_cross_attention import source_of_channels
from blackout_env.model.modules.temporal_memory import TemporalMemory

N_QUANTILES_DEFAULT = 32  # quantile samples used when the caller doesn't ask for a specific count

GRID_H = GRID_W = 6  # GraphicEncoder's final spatial size (24x24 input, two stride-2 convs)
N_VISION_TOKENS = GRID_H * GRID_W  # 36
N_UNITS = 10
N_TEAM_STATE_TOKENS = 1
N_SPR_CLS_TOKENS = 1  # dedicated learned query token for the SPR aux loss -- see class docstring
N_DISCRETE_ACTIONS = 8  # 8 compass directions, see my_policy.py:DIRECTION_VECTORS
N_ATTENTION_HEADS = 8
ATTENTION_DEPTH = 4
# RoPE frequency base, sized for a 6x6 grid rather than a text sequence -- see RotaryEmbedding2D.
ROPE_BASE = 10.0

# Token-type ids for the trunk sequence: which tokens are "graphic", "unit", "global"
# (team_state), or the dedicated SPR CLS token — see MyModel's token_type_emb.
TYPE_GRAPHIC, TYPE_UNIT, TYPE_GLOBAL, TYPE_SPR_CLS = 0, 1, 2, 3

TEMPORAL_PREFIX = "temporal."  # every temporal-memory parameter's state_dict key starts with this

POLICY_HEADS = ("iqn", "logits")
_IQN_KEY = "q_head.tau_proj.0.weight"  # only an IQN head has it (LogitsHead: q_head.in_scale instead)
ANGLE_KEY_PREFIX = "q_head.angle_"  # only a logits head with an angle head (P11) has them
_ANGLE_SCALE_KEY = "q_head.angle_scale"
MAP_XATTN_PREFIX = "map_xattn."  # only a net with the map-token cross-attention path (§25) has them


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
      - 2D RoPE over half of each head's dims (the other half is left unrotated for
        content-based matching — see apply_rope), so attention can tell *where in the 6x6 map*
        a token sits and reason about relative spatial offsets. Vision tokens get their grid
        cell, unit tokens their real (fractional) position, and team_state/spr_cls the grid
        centre. Unit tokens used to get row=col=0 like the other non-grid tokens, which left
        attention no positional way to connect a unit to the vision tokens covering the ground
        it stands on -- it had to decode raw coordinates out of token content instead;
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

    Temporal memory (``temporal=True``)
    -----------------------------------
    Off by default: the stateless model, its forward and its state_dict keys are unchanged.  On,
    a ``TemporalMemory`` block (modules/temporal_memory.py: per-slot KDA over the 10 unit tokens
    + the team_state token, then attention across those 11 slots, added back residually through
    a zero-initialised projection) sits between the trunk and the Q head.  At init the block is
    the identity, so ``MyModel.from_state_dict(stateless_state, temporal=True)`` reproduces the
    stateless net's Q-values exactly.  Its keys all start with ``temporal.``.

    * ``forward(..., memory=None)``: one frame per batch row.  A temporal model returns the same
      five outputs plus the new ``memory`` as a sixth, last element (``q_values, *_ = net(...)``
      still works); ``memory=None`` = start of a match.
    * ``forward_sequence(...)``: [B, T, ...] frames; trunk on all B*T frames at once, KDA chunk-
      parallel over T, ``reset`` [B, T] zeroes the memory before chosen frames.  Equals T
      successive ``forward`` calls carrying ``memory`` (tests/test_temporal_model.py).

    Memory: ``(S [B, 11, H, dk, dv], conv [B, 11, K-1, 2*H*dk + H*dv])`` fp32 tensors, in the
    canonical (own team first) frame -- a Team B caller mirrors the obs, never the memory.

    Policy head (``policy_head``)
    -----------------------------
    ``"iqn"`` (default): the IQN quantile head above.  ``"logits"`` (P5): ``LogitsHead``, the same
    value MLP without the tau embedding -- one evaluation instead of one per quantile, for a
    distilled student that never needs the return distribution.  Its forward ignores
    ``n_quantiles``/``tau`` and returns ``q_values`` = the logits, ``quantile_values`` = the
    logits with a size-1 quantile axis, and ``tau`` = None, so ``q_values, *_ = net(...)`` callers
    work unchanged.  ``from_state_dict(iqn_state, policy_head="logits")`` converts an IQN state
    (``iqn_to_logits_state``).

    Angle head (``angle_head``, P11, logits head and K bins only)
    --------------------------------------------------------------
    ``LogitsHead``'s per-bin offsets (modules/logits_head.py): bin k executes ``2 pi k / K +
    offset_k``.  Only computed when asked: ``forward(..., angle=True)`` returns the offsets
    [B, q_units, K] (radians, canonical frame) as an extra element after the five outputs and
    before the memory; ``forward_sequence(..., angle=True)`` likewise, before the memory.
    Without ``angle=True`` every output is exactly the net without an angle head, and at init
    (zero ``angle_out``) the offsets are 0, so ``from_state_dict(state, angle_head=True)`` on a
    state without one moves exactly like it.  ``angle_arc_bins`` N: each offset spans +-(N + 1/2)
    bins, so a bin next to the teacher's (an arc label's arg max) can still reach its angle
    (recorded in the state as ``q_head.angle_scale``).

    Map-token cross-attention (``map_xattn``, docs/policy_v2_design_ko.md §25)
    ---------------------------------------------------------------------------
    Off by default (no keys, the same forward).  ``"f3"`` / ``"f2"``: after ``map_xattn_layers``
    trunk layers (spread over the depth) the 11 vector tokens cross-attend to map tokens made from
    the GraphicEncoder's 6x6 x 128 (f3) or 12x12 x 64 (f2) feature map -- keys/values only, learned
    2-D position embeddings, a unit's query adding the embedding of the cell it stands in
    (modules/map_cross_attention.py).  The blocks' output projections start at zero, so
    ``from_state_dict(state, map_xattn="f3")`` on a state without the path is exactly that net.
    Keys under ``map_xattn.``; ``config_from_state_dict`` reads the source and depth back.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_items: int = 5,
        n_classes: int = 3,
        team_state_size: int = 4,
        n_actions: int = N_DISCRETE_ACTIONS,
        exclusive_attention: bool = True,
        temporal: bool = False,
        temporal_heads: int = 4,
        temporal_head_dim: int = 64,
        temporal_chunk: int = 64,
        policy_head: str = "iqn",
        angle_head: bool = False,
        angle_arc_bins: int = 0,
        map_xattn: str | None = None,
        map_xattn_layers: int = 2,
    ) -> None:
        super().__init__()
        if policy_head not in POLICY_HEADS:
            raise ValueError(f"policy_head must be one of {POLICY_HEADS}, got {policy_head!r}")
        if angle_head and policy_head != "logits":
            raise ValueError("angle_head needs policy_head='logits'")
        self.angle_head = bool(angle_head)

        self.hidden_size = hidden_size
        self.policy_head = policy_head

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

        # Per-slot identity embedding for the 10 unit tokens + 1 team_state token. RoPE now
        # gives unit tokens their map position, but two units standing on the same spot still
        # share it, and token_type_emb only says "this is *a* unit token", identical for all 10.
        # Two units with identical agent_states rows (e.g. stacked at the same spawn point)
        # therefore produced byte-identical Q-values and thus always chose the same action --
        # a self-reinforcing lockstep with no way to break symmetry, since self-attention is
        # permutation-equivariant over tokens with identical content. This embedding is added
        # per fixed slot (unit index 0-9, independent of that unit's state) so two units are
        # always distinguishable even when their observed state coincides exactly.
        self.vec_slot_emb = nn.Embedding(n_vector_tokens, hidden_size)
        vec_slot_ids = torch.arange(n_vector_tokens, dtype=torch.long)
        self.register_buffer("vec_slot_ids", vec_slot_ids, persistent=False)

        # 2D RoPE grid coordinates for the trunk sequence. The vision tokens' coordinates are
        # fixed, and team_state/spr_cls sit at the grid centre (see build_grid_position_ids);
        # the 10 unit tokens' rows get overwritten per sample in forward() with where those
        # units actually stand, so attention can match a unit against the vision tokens around
        # it by position instead of having to decode raw coordinates out of token content.
        head_dim = hidden_size // N_ATTENTION_HEADS
        self.rope = RotaryEmbedding2D(head_dim, base=ROPE_BASE)
        n_extra_tokens = n_vector_tokens + N_SPR_CLS_TOKENS
        row_ids, col_ids = build_grid_position_ids(GRID_H, GRID_W, n_extra_tokens)
        self.register_buffer("rope_row_ids", row_ids, persistent=False)
        self.register_buffer("rope_col_ids", col_ids, persistent=False)

        # auxiliary vision representation head (e.g. for a self-predictive/SPR-style aux loss)
        self.spr_head = SwiGLUBlock(hidden_size, hidden_size * 3, hidden_size)

        # IQN (arXiv:1806.06923) distributional Q-head, chosen over C51 so the value range
        # doesn't need a hand-tuned [Vmin, Vmax] that would need re-tuning whenever the
        # reward balance changes (see reward_config.json) — see modules/iqn_head.py.
        # policy_head="logits": the same value MLP without tau (see class docstring "Policy head").
        if policy_head == "iqn":
            self.q_head = IQNHead(hidden_size, n_actions, n_cos=64)
        else:  # the angle head's offsets: one per heading bin (STAND, if any, has none)
            angle_bins = (n_actions - n_actions % 2) if angle_head else 0
            self.q_head = LogitsHead(hidden_size, n_actions, angle_bins=angle_bins, angle_arc_bins=angle_arc_bins)
        self.angle_arc_bins = int(angle_arc_bins) if angle_head else 0

        # Created last so that a stateless model's parameter order (and optimiser state layout)
        # is unchanged; absent entirely when off, so its state_dict has no temporal keys.
        self.temporal = (
            TemporalMemory(hidden_size, n_vector_tokens, temporal_heads, temporal_head_dim, temporal_chunk)
            if temporal else None
        )
        # After the temporal block, and absent when off: an old net's parameter order is unchanged.
        self.map_xattn = (MapCrossAttention(hidden_size, N_ATTENTION_HEADS, map_xattn_layers, map_xattn)
                          if map_xattn else None)
        self._xattn_after = (MapCrossAttention.after_layers(ATTENTION_DEPTH, map_xattn_layers)
                             if map_xattn else ())

    @property
    def is_temporal(self) -> bool:
        return self.temporal is not None

    def initial_memory(self, batch: int = 1, device=None):
        """Zero memory (the start of a match) for ``batch`` rows; None for a stateless model."""
        if self.temporal is None:
            return None
        return self.temporal.initial_memory(batch, device=device)

    @staticmethod
    def config_from_state_dict(state: dict) -> dict:
        """Constructor kwargs that match ``state``: width, the head's outputs (8 or K bins, +
        STAND), the policy head (iqn iff it has tau_proj keys) and -- when it has ``temporal.``
        keys -- the temporal block's heads and head size."""
        config = dict(hidden_size=state["token_type_emb.weight"].shape[1],
                      n_actions=state["q_head.value_head.1.weight"].shape[0],
                      policy_head="iqn" if _IQN_KEY in state else "logits")
        if any(key.startswith(ANGLE_KEY_PREFIX) for key in state):
            config["angle_head"] = True
            if _ANGLE_SCALE_KEY in state:  # the offsets' range, in bins around their own
                config["angle_arc_bins"] = arc_bins_of_scale(state[_ANGLE_SCALE_KEY], config["n_actions"] - config["n_actions"] % 2)
        if MAP_XATTN_PREFIX + "in_proj.weight" in state:
            config["map_xattn"] = source_of_channels(state[MAP_XATTN_PREFIX + "in_proj.weight"].shape[1],
                                                     state[MAP_XATTN_PREFIX + "pos_row"].shape[0])
            config["map_xattn_layers"] = len({k.split(".")[2] for k in state if k.startswith(MAP_XATTN_PREFIX + "blocks.")})
        if any(key.startswith(TEMPORAL_PREFIX) for key in state):
            heads = state["temporal.kda.b_proj.weight"].shape[0]
            config.update(temporal=True, temporal_heads=heads,
                          temporal_head_dim=state["temporal.kda.q_proj.weight"].shape[0] // heads)
        return config

    @staticmethod
    def iqn_to_logits_state(state: dict, n_quantiles: int = N_QUANTILES_DEFAULT) -> dict:
        """An IQN-head MyModel state -> the same model with a LogitsHead (``policy_head="logits"``):
        every other key unchanged, the head folded at the fixed taus ``(i + 0.5) / n_quantiles``
        that ``inference.fixed_tau`` uses (``modules/logits_head.state_from_iqn``: exact at
        n_quantiles=1, the mean tau embedding otherwise).  A logits state is returned as is."""
        if _IQN_KEY not in state:
            return state
        tau = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        return state_from_iqn(state, tau, prefix="q_head.")

    @classmethod
    def from_state_dict(cls, state: dict, temporal: bool | None = None, policy_head: str | None = None,
                        angle_head: bool | None = None, map_xattn: str | bool | None = None,
                        map_xattn_layers: int | None = None, **temporal_kwargs) -> "MyModel":
        """A MyModel shaped like ``state`` with ``state`` loaded.

        ``temporal=None`` follows the state (temporal iff it has ``temporal.`` keys).
        ``temporal=True`` on a stateless state is the warm start: every stateless key is loaded
        strictly, the temporal block keeps its (identity) init -- see ``load_warm_start``.
        ``temporal=False`` on a temporal state raises (dropping the memory changes the policy).
        ``policy_head=None`` follows the state; ``"logits"`` on an IQN state converts it
        (``iqn_to_logits_state``); ``"iqn"`` on a logits state raises (no tau embedding to restore).
        ``angle_head=None`` follows the state; True on a state without one adds it zero-initialised
        (the same moves); False on a state with one raises.
        ``map_xattn=None`` follows the state; a source name on a state without the path adds it
        zero-initialised (the same outputs); False on a state with it raises."""
        config = cls.config_from_state_dict(state)
        if map_xattn is False and config.get("map_xattn"):
            raise ValueError("state has a map-token cross-attention path; load it with map_xattn=None")
        if map_xattn and not config.get("map_xattn"):
            config["map_xattn"] = map_xattn
            if map_xattn_layers is not None:
                config["map_xattn_layers"] = map_xattn_layers
        elif map_xattn and map_xattn != config["map_xattn"]:
            raise ValueError(f"state's map tokens are {config['map_xattn']!r}, not {map_xattn!r}")
        if policy_head == "iqn" and config["policy_head"] == "logits":
            raise ValueError("state has a logits head; it cannot be loaded as an IQN head")
        if policy_head == "logits" and config["policy_head"] == "iqn":
            state = cls.iqn_to_logits_state(state)
            config["policy_head"] = "logits"
        if angle_head is False and config.get("angle_head"):
            raise ValueError("state has an angle head; load it with angle_head=None or True")
        if angle_head:
            config["angle_head"] = True
        has_temporal = config.get("temporal", False)
        if temporal is False and has_temporal:
            raise ValueError("state has temporal keys; load it with temporal=None or True")
        if temporal and not has_temporal:
            config.update(temporal=True, **temporal_kwargs)
        elif temporal_kwargs:
            config.update(temporal_kwargs)
        net = cls(**config)
        net.load_warm_start(state)
        return net

    def with_init_angle_keys(self, state: dict) -> dict:
        """``state`` plus this net's angle-head keys it lacks: an angle-head net warm-started from a
        state without one keeps its zero init (and a state without ``angle_scale`` its default,
        the half-bin range every angle head had before that buffer).  Likewise the map-token
        cross-attention path, when ``state`` has none of it (its zero-initialised output = no change)."""
        prefixes = ((ANGLE_KEY_PREFIX,) if self.angle_head else ()) + (
            (MAP_XATTN_PREFIX,) if self.map_xattn is not None and not any(k.startswith(MAP_XATTN_PREFIX) for k in state)
            else ())
        if not prefixes:
            return state
        missing = [k for k in self.state_dict() if k.startswith(prefixes) and k not in state]
        if not missing:
            return state
        own = self.state_dict()
        return {**state, **{k: own[k].clone() for k in missing}}

    def load_warm_start(self, state: dict) -> list[str]:
        """Load ``state`` -- a stateless or a temporal state dict -- into this model.

        Every non-temporal key must match exactly (missing or unexpected ones raise, like
        ``load_state_dict(strict=True)``); temporal keys absent from ``state`` keep their init,
        which for a fresh model is the identity.  Returns those initialised temporal keys.  An
        angle head absent from ``state`` keeps its zero init too."""
        state = self.with_init_angle_keys(state)
        if any(key.startswith(TEMPORAL_PREFIX) for key in state):  # a temporal state: all of it
            self.load_state_dict(state)
            return []
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if not key.startswith(TEMPORAL_PREFIX)]
        if bad_missing or unexpected:
            raise RuntimeError(f"state does not fit this MyModel: missing {bad_missing}, unexpected {list(unexpected)}")
        return list(missing)

    def _token_grid_positions(self, agent_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-sample (row, col) RoPE coordinates for the whole trunk sequence, [B, T] each.

        Everything but the unit tokens is fixed. Unit tokens take their real position, mapped
        from the env's normalized [-1, 1] coordinates into the 6x6 *token* grid and left
        fractional -- a unit three quarters of the way across a token's footprint gets an angle
        three quarters of the way to the next token's. Mapping into token coordinates rather
        than the 24x24 map grid also keeps the fastest RoPE pair under one full turn across the
        map; feeding it tile indices (0-23) would wrap it ~3.7 times and alias distinct
        positions onto the same angle.
        """
        batch = agent_states.shape[0]
        row_ids = self.rope_row_ids.unsqueeze(0).expand(batch, -1).clone()
        col_ids = self.rope_col_ids.unsqueeze(0).expand(batch, -1).clone()

        x, y = agent_states[..., 0], agent_states[..., 1]  # [B, N_UNITS], normalized [-1, 1]
        unit_rows = ((1.0 - y) * 0.5 * GRID_H - 0.5).clamp(0.0, GRID_H - 1.0)
        unit_cols = ((x + 1.0) * 0.5 * GRID_W - 0.5).clamp(0.0, GRID_W - 1.0)
        row_ids[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS] = unit_rows
        col_ids[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS] = unit_cols
        return row_ids, col_ids

    def forward(
        self,
        graphic: torch.Tensor,
        team_state: torch.Tensor,
        agent_states: torch.Tensor,
        n_quantiles: int = N_QUANTILES_DEFAULT,
        tau: torch.Tensor | None = None,
        q_units: int = N_UNITS,
        memory=None,
        angle: bool = False,
    ) -> tuple:
        """
        graphic      : [B, C, H, W]
        team_state   : [B, team_state_size]
        agent_states : [B, N_UNITS, agent_state_size]
        n_quantiles  : number of IQN quantile samples to draw (ignored if `tau` is given)
        tau          : [B, n_quantiles] fixed quantile fractions to use instead of sampling
                       (e.g. so an online/target pair can share draws where that matters)
        q_units      : Q-values only for the first q_units unit rows (the others' rows never
                       reach the head, which is half the forward at 32 quantiles) -- e.g. 5 for a
                       caller that only needs the own team of a canonical-frame batch. The
                       outputs below then have q_units rows where they say N_UNITS.

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
                           token, used as QMixer's hypernetwork input (after the temporal block
                           when there is one).
        memory          : temporal model only, a sixth element: the memory after this frame
                           (pass it back as ``memory`` next tick; the ``memory`` argument is the
                           one before it, None = start of a match).  Stateless: must be None.
        """
        tokens_out = self._trunk(graphic, team_state, agent_states)
        vec_out = tokens_out[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS + N_TEAM_STATE_TOKENS]
        new_memory = None
        if self.temporal is not None:
            vec_out, new_memory = self.temporal(vec_out.unsqueeze(1), memory)
            vec_out = vec_out.squeeze(1)
        elif memory is not None:
            raise ValueError("memory was given to a stateless MyModel (temporal=False)")
        outputs = self._heads(vec_out, tokens_out[:, -1], n_quantiles, tau, q_units, angle)
        return outputs if self.temporal is None else (*outputs, new_memory)

    def _trunk(self, graphic, team_state, agent_states) -> torch.Tensor:
        """The 48 trunk tokens' post-attention output, [B, 48, hidden]."""
        B = graphic.shape[0]
        # Features the env does not send, but that measured failures traced back to -- unit
        # occupancy, storage free capacity, fetchable batteries, and each unit's local
        # walkability. Derived here, the one point every caller flows through, so training,
        # collection and evaluation cannot drift apart. See model/derived_obs.py.
        unit_features = local_wall_features(graphic, agent_states)  # before graphic grows
        graphic = torch.cat((graphic, derived_map_channels(graphic, agent_states)), dim=1)
        agent_states = torch.cat((agent_states, unit_features), dim=-1)

        if self.map_xattn is not None:
            vis_tokens, feats = self.graphic_encoder(graphic, features=True)
        else:
            vis_tokens = self.graphic_encoder(graphic)               # [B, 36, hidden]
        vec_tokens = self.vector_encoder(agent_states, team_state)   # [B, 11, hidden]
        vec_tokens = vec_tokens + self.vec_slot_emb(self.vec_slot_ids)[None, :, :]
        spr_cls_tok = self.spr_cls_emb(self.spr_cls_ids)[None, :, :].expand(B, -1, -1)  # [B, 1, hidden]

        tokens_in = torch.cat((vis_tokens, vec_tokens, spr_cls_tok), dim=1)  # [B, 48, hidden]
        tokens_in = tokens_in + self.token_type_emb(self.token_type_ids)[None, :, :]

        row_ids, col_ids = self._token_grid_positions(agent_states)  # [B, T] each
        rope_cos, rope_sin = self.rope(row_ids, col_ids)
        if self.map_xattn is None:
            return self.attention(tokens_in, rope=(rope_cos, rope_sin))
        return self._trunk_xattn(tokens_in, (rope_cos, rope_sin), feats[self.map_xattn.source], agent_states)

    def _trunk_xattn(self, x, rope, feat, agent_states) -> torch.Tensor:
        """The trunk layers with the map-token cross blocks after ``_xattn_after`` (class docstring)."""
        lo, hi = N_VISION_TOKENS, N_VISION_TOKENS + N_UNITS + N_TEAM_STATE_TOKENS
        mem, q_pos = self.map_xattn.memory(feat, agent_states[..., :2], hi - lo)
        blocks = iter(self.map_xattn.blocks)
        for i, layer in enumerate(self.attention.layers):
            x = layer(x, None, False, rope)
            if i in self._xattn_after:
                vec = x[:, lo:hi]
                x = torch.cat((x[:, :lo], vec + next(blocks)(vec, q_pos, mem), x[:, hi:]), dim=1)
        return x

    def _heads(self, vec_out, spr_cls_out, n_quantiles, tau, q_units, angle=False):
        """vec_out [B, 11, hidden] (10 units + team_state), spr_cls_out [B, hidden] -> forward's
        five outputs (+ the angle offsets with ``angle``)."""
        unit_out = vec_out[:, :N_UNITS]
        global_out = vec_out[:, N_UNITS]  # the single TYPE_GLOBAL (team_state) token
        vision_latent = self.spr_head(spr_cls_out)
        if angle and not self.angle_head:
            raise ValueError("angle=True needs a MyModel with angle_head=True")
        if self.policy_head == "logits":  # no quantile axis: size 1, tau None (class docstring)
            if angle:
                q_values, offsets = self.q_head(unit_out[:, :q_units], angle=True)
                return q_values, q_values.unsqueeze(2), None, vision_latent, global_out, offsets
            q_values = self.q_head(unit_out[:, :q_units])
            return q_values, q_values.unsqueeze(2), None, vision_latent, global_out
        quantile_values, tau = self.q_head(unit_out[:, :q_units], n_quantiles=n_quantiles, tau=tau)
        q_values = quantile_values.mean(dim=2)  # [B, N_UNITS, n_actions]
        return q_values, quantile_values, tau, vision_latent, global_out

    def forward_sequence(
        self,
        graphic: torch.Tensor,
        team_state: torch.Tensor,
        agent_states: torch.Tensor,
        memory=None,
        reset: torch.Tensor | None = None,
        n_quantiles: int = N_QUANTILES_DEFAULT,
        tau: torch.Tensor | None = None,
        q_units: int = N_UNITS,
        burn_in: int = 0,
        angle: bool = False,
    ):
        """
        A temporal model over whole sequences (training / prefill).

        graphic [B, T, C, H, W], team_state [B, T, team_state_size], agent_states
        [B, T, N_UNITS, agent_state_size]; memory: the state before frame 0 (None = zeros);
        reset [B, T] bool: zero the memory before those frames (a match starting mid-window);
        tau: None (sampled per frame), [B, n_quantiles] (shared over T) or [B, T, n_quantiles];
        burn_in: the first ``burn_in`` frames only warm the memory -- trunk and KDA under
        no_grad, no Q outputs for them (R2D2-style); the memory entering frame ``burn_in`` is
        detached.

        Returns (q_values [B, T', q_units, n_actions], quantile_values [B, T', q_units,
        n_quantiles, n_actions], tau [B, T', n_quantiles], memory after the last frame), with
        T' = T - burn_in.  A logits head: n_quantiles = 1 and tau None.  ``angle``: the angle offsets
        [B, T', q_units, K] as a fourth element, before the memory.
        """
        if self.temporal is None:
            raise ValueError("forward_sequence needs a temporal MyModel (temporal=True)")
        B, T = graphic.shape[:2]
        if not 0 <= burn_in < T:
            raise ValueError(f"burn_in must be in [0, T={T}), got {burn_in}")
        if tau is not None and tau.dim() == 3:
            tau = tau[:, burn_in:]
        if burn_in:
            with torch.no_grad():
                memory = self._temporal_frames(graphic[:, :burn_in], team_state[:, :burn_in], agent_states[:, :burn_in],
                                               memory, None if reset is None else reset[:, :burn_in])[1]
            memory = tuple(m.detach() for m in memory)
            graphic, team_state, agent_states = graphic[:, burn_in:], team_state[:, burn_in:], agent_states[:, burn_in:]
            reset = None if reset is None else reset[:, burn_in:]
            T -= burn_in
        (vec_out, spr_cls_out), memory = self._temporal_frames(graphic, team_state, agent_states, memory, reset)
        if tau is not None:
            tau = tau.unsqueeze(1).expand(B, T, tau.shape[-1]) if tau.dim() == 2 else tau
            tau = tau.reshape(B * T, -1)
        q_values, quantile_values, tau, _, _, *offsets = self._heads(vec_out, spr_cls_out, n_quantiles, tau, q_units, angle)
        offsets = tuple(o.reshape(B, T, *o.shape[1:]) for o in offsets)
        return (q_values.reshape(B, T, *q_values.shape[1:]), quantile_values.reshape(B, T, *quantile_values.shape[1:]),
                None if tau is None else tau.reshape(B, T, -1), *offsets, memory)

    def _temporal_frames(self, graphic, team_state, agent_states, memory, reset):
        """Trunk on B*T frames, temporal block over T: ((vec_out [B*T, 11, D], spr_cls [B*T, D]), memory)."""
        B, T = graphic.shape[:2]
        tokens_out = self._trunk(graphic.flatten(0, 1), team_state.flatten(0, 1), agent_states.flatten(0, 1))
        vec_out = tokens_out[:, N_VISION_TOKENS : N_VISION_TOKENS + N_UNITS + N_TEAM_STATE_TOKENS]
        vec_out, memory = self.temporal(vec_out.reshape(B, T, *vec_out.shape[1:]), memory, reset)
        return (vec_out.flatten(0, 1), tokens_out[:, -1]), memory
