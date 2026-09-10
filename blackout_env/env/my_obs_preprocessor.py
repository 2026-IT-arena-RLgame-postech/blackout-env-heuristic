"""
Preprocessor for the redesigned BlackOut observations: bit-packed per-team graphics,
a shared per-agent state table, and a per-team game-state summary.

Graphic observation
--------------------
Unity side (SemanticMapRenderer.cs) renders a single-channel R16 texture where each pixel
is one packed ushort instead of an 8-bit grayscale semantic-id map:

    bit 0-2 : base tile category (static per episode)
              0=void 1=wall 2=site_hunter 3=site_carrier
              4=spawn_ally 5=spawn_enemy 6=storage_ally 7=storage_enemy
    bit 3-6 : battery stack count on this tile, saturated to [0,15] (covers the current
              MaxItemAmount of 10 exactly, with headroom to 15 if it's raised later)
    bit 7-9 : index of the non-battery item on this tile (0=none, 1=BuffSpeed,
              2=DebuffSpeed, 3=BuffSize, 4=DebuffSize; 5-7 reserved for future items)

The packed value (0-1023, all 10 bits used) is written to the texture as a raw ushort and
normalized by PACK_DIVISOR (must match SemanticMapRenderer.PACK_DIVISOR in Unity) before
being sent as a float visual observation. preprocess_team_graphics() multiplies back by
PACK_DIVISOR, bit-unpacks with mask/shift, and expands into 13 float channels per team:

    ch 0-7  : base category one-hot
    ch 8    : battery count, scalar in [0, 1] (count / 15)
    ch 9-12 : item_1..item_4 one-hot (BuffSpeed, DebuffSpeed, BuffSize, DebuffSize)

Units are intentionally NOT part of the graphic observation — unit positions are carried
entirely by the agent-state table below.

Why CPU/numpy and not GPU
--------------------------
This runs once per env.step() on a single (H=24, W=24) array (576 pixels). At that size,
a GPU dispatch (host->device transfer + kernel launch + device->host readback) costs far
more than the vectorized bit-unpack itself, which is a handful of numpy ops over ~600
elements (microseconds). GPU acceleration would only start to pay off if many envs' raw
graphics were batched into one array before this runs (e.g. shape (N, H, W)) — the
bit/mask logic here is written with plain numpy broadcasting so swapping in torch tensors
for a batched call site later is a near-drop-in change if that ever becomes the bottleneck.

Vector observation: MapObsAgent-sourced agent-state table + team-state summary
---------------------------------------------------------------------------------
Vector data is now minimized on the wire too, mirroring the graphic optimization: instead
of all 10 BlackOutUnit agents each redundantly sending a full state vector, MapObsAgent
broadcasts ONE shared float32[44] raw state per step (see MapObsAgent.cs):

    [0~39] : 10 unit blocks x 4 floats (pos_x, pos_y normalized to [-1, 1]; holdingItemId, classId)
    [40]   : score_A   (absolute — MapObsAgent isn't owned by either team)
    [41]   : score_B
    [42]   : episode_time_left
    [43]   : absorption_time_left

There's no team_sign or unitIndex in this vector: team is derived from block index alone
(blocks 0-4 are always Team A, 5-9 always Team B — the same fixed convention `constants.py`
already relies on), and unitIndex isn't needed here since this isn't any one unit's
observation. Each BlackOutUnit agent separately sends a 1-float vector containing only its
own unitIndex, used purely for action/reward routing (see blackout_env.py).

preprocess_agent_states() returns BOTH team perspectives directly, since building one from
the other is just a sign flip on the "team" column (+1/-1) — position/item/class are
identical either way. preprocess_team_states() similarly returns both, reordering the two
absolute scores per team so index 0 is always "my score".

Held-battery count is folded into the existing item one-hot rather than added as a separate
dimension: index 1 (the battery slot) holds count/BATTERY_MAX instead of a flat 1.0, so the
one-hot's shape doesn't change size depending on whether a battery is held.
"""

from __future__ import annotations

import numpy as np

from .constants import N_TEAM_A
from .obs_preprocessor import ObsPreprocessor


class MyObsPreprocessor(ObsPreprocessor):
    """
    ObsPreprocessor variant returning bit-packed per-team graphics, a per-team-perspective
    agent-state table pair, and a per-team game-state summary pair, instead of a single flat
    per-agent vector. preprocess_vector is disabled (raises) — use preprocess_agent_states /
    preprocess_team_states instead, both computed once per step from MapObsAgent's shared
    raw state vector (not from any individual BlackOutUnit agent's observation).
    """

    # ------------------------------------------------------------------
    # Graphic bit-packing (must match SemanticMapRenderer.cs)
    # ------------------------------------------------------------------

    PACK_DIVISOR: float = 1024.0

    BASE_MASK     = 0b111
    BATTERY_SHIFT = 3
    BATTERY_MASK  = 0b1111
    BATTERY_MAX   = 0b1111  # 15 — current MaxItemAmount(10) fits; Unity saturates before packing.
    ITEM_SHIFT    = BATTERY_SHIFT + 4  # 7
    ITEM_MASK     = 0b111

    N_BASE_CHANNELS = 8

    # Base category channel indices
    VOID          = 0
    WALL          = 1
    SITE_HUNTER   = 2
    SITE_CARRIER  = 3
    SPAWN_ALLY    = 4
    SPAWN_ENEMY   = 5
    STORAGE_ALLY  = 6
    STORAGE_ENEMY = 7
    BATTERY       = 8  # scalar channel, not one-hot

    # ------------------------------------------------------------------
    # Vector raw layout (must match MapObsAgent.cs's shared state broadcast)
    # ------------------------------------------------------------------

    N_UNITS         = 10
    UNIT_BLOCK_SIZE = 4  # pos_x, pos_y, holdingItemId, classId (no team_sign — derived from index)
    SCALAR_COUNT    = 4  # score_A, score_B, episode_time_left, absorption_time_left

    RAW_SCALAR_START = N_UNITS * UNIT_BLOCK_SIZE       # 40
    RAW_VECTOR_SIZE  = RAW_SCALAR_START + SCALAR_COUNT  # 44

    # Each BlackOutUnit agent's own (separate, tiny) observation is just its unitIndex.
    UNIT_INDEX_VECTOR_SIZE = 1

    def __init__(self, semantic_config: dict, n_items: int = 5, n_classes: int = 3):
        super().__init__(semantic_config, n_items=n_items, n_classes=n_classes)

        # Item 0 is always the stackable battery (count-encoded); items 1..n_items-1 are
        # presence-only and share the 3-bit item-index field (max 7 representable).
        self.n_nonbattery_items = n_items - 1
        if not (0 <= self.n_nonbattery_items <= self.ITEM_MASK):
            raise ValueError(
                f"n_nonbattery_items must be 0-{self.ITEM_MASK} (got {self.n_nonbattery_items} "
                f"from n_items={n_items}); the 3-bit item-index field can't address more types."
            )

        self.n_graphic_channels = self.N_BASE_CHANNELS + 1 + self.n_nonbattery_items

        # Precomputed comparison arrays for vectorized one-hot separation.
        self._base_channel_ids = np.arange(self.N_BASE_CHANNELS, dtype=np.uint16)
        self._item_channel_ids = np.arange(1, self.n_nonbattery_items + 1, dtype=np.uint16)

        # Per-agent state = pos(2) + team(1) + item one-hot(n_items+1) + class one-hot(n_classes)
        self.agent_state_size = 2 + 1 + (n_items + 1) + n_classes
        self.team_state_size = self.SCALAR_COUNT

    # ------------------------------------------------------------------
    # Vector preprocessing
    # ------------------------------------------------------------------

    def preprocess_vector(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MyObsPreprocessor replaces preprocess_vector with preprocess_agent_states() "
            "(called once per step, shared across all agents) and preprocess_team_states()."
        )

    def preprocess_agent_states(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds the shared 10-agent state table from MapObsAgent's raw state vector, in both
        team perspectives. Position/item/class are objective (identical either way); only the
        "team" column differs, so team B's table is a cheap sign-flip of team A's rather than
        a second independent pass over the raw data.

        Parameters
        ----------
        raw : float32[44]  MapObsAgent's shared per-step state vector.

        Returns
        -------
        (agent_states_a, agent_states_b) : float32[N_UNITS, agent_state_size] each
          per row: [pos_x, pos_y, team, *item_onehot(n_items+1), *class_onehot(n_classes)]
          pos_x, pos_y (indices 0-1): normalized to [-1, 1], zero-centered on the map's
            bottom-left-to-top-right diagonal (see MapObsAgent.cs). Passed through unchanged
            from the raw broadcast — this docstring is the only place the range is asserted.
          team (index 2): +1.0 if this unit is on "my" team from that perspective, else -1.0
            (block index < N_TEAM_A is Team A; agent_states_a keeps that sign as-is,
            agent_states_b flips it)
          item_onehot: index 0 = holding nothing, index 1 = holding battery (value =
            count / BATTERY_MAX instead of a flat 1.0), index 2+ = other item types (flat 1.0)
        """
        assert raw.shape == (self.RAW_VECTOR_SIZE,), f"Expected float32[{self.RAW_VECTOR_SIZE}], got {raw.shape}"

        rows = []
        for i in range(self.N_UNITS):
            base = i * self.UNIT_BLOCK_SIZE
            pos = raw[base : base + 2]
            team_a_sign = np.float32(1.0 if i < N_TEAM_A else -1.0)
            raw_item = raw[base + 2]
            class_id = int(round(raw[base + 3]))

            is_battery = raw_item < 0
            slot = 1 if is_battery else int(round(raw_item))
            item_onehot = self._one_hot(slot, self.n_items + 1)
            if is_battery:
                item_onehot[1] = -raw_item / self.BATTERY_MAX

            class_onehot = self._one_hot(class_id, self.n_classes)

            rows.append(np.concatenate([pos, [team_a_sign], item_onehot, class_onehot], dtype=np.float32))

        agent_states_a = np.stack(rows).astype(np.float32)
        agent_states_b = agent_states_a.copy()
        agent_states_b[:, 2] *= -1.0  # flip the team column only
        return agent_states_a, agent_states_b

    def preprocess_team_states(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds both teams' game-state summaries from MapObsAgent's raw state vector.

        Parameters
        ----------
        raw : float32[44]  MapObsAgent's shared per-step state vector.

        Returns
        -------
        (team_state_a, team_state_b) : float32[4] each
          [own_score, opp_score, episode_time_left, absorption_time_left] — the two absolute
          scores are reordered per team so index 0 is always "my score"; the two time values
          are global and identical in both.
        """
        assert raw.shape == (self.RAW_VECTOR_SIZE,), f"Expected float32[{self.RAW_VECTOR_SIZE}], got {raw.shape}"

        score_a = raw[self.RAW_SCALAR_START]
        score_b = raw[self.RAW_SCALAR_START + 1]
        times = raw[self.RAW_SCALAR_START + 2 : self.RAW_SCALAR_START + 4]

        team_state_a = np.concatenate([[score_a, score_b], times]).astype(np.float32)
        team_state_b = np.concatenate([[score_b, score_a], times]).astype(np.float32)
        return team_state_a, team_state_b

    # ------------------------------------------------------------------
    # Visual preprocessing
    # ------------------------------------------------------------------

    def preprocess_team_graphics(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper: decode once, derive both team perspectives."""
        team_a = self.preprocess_graphic(raw)
        team_b = self.flip_team_perspective(team_a)
        return team_a, team_b

    def preprocess_graphic(self, raw: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        raw : float32[H × W × 1]  packed value from the R16 RenderTextureSensor, normalized
              to [0, 1] by Unity (packed_ushort / PACK_DIVISOR).

        Returns
        -------
        float32[H × W × n_graphic_channels]
          ch 0-7            : base category one-hot (void/wall/site_hunter/site_carrier/
                               spawn_ally/spawn_enemy/storage_ally/storage_enemy)
          ch 8               : battery count, scalar in [0, 1] (count / BATTERY_MAX)
          ch 9 ~ 8+N_ITEMS   : item_i one-hot (i = 1..n_nonbattery_items)
        """
        assert raw.ndim == 3 and raw.shape[2] == 1, (
            f"Expected float32[H×W×1], got {raw.shape}"
        )
        packed = np.round(raw[:, :, 0] * self.PACK_DIVISOR).astype(np.uint16)  # H × W

        base = packed & self.BASE_MASK
        battery = (packed >> self.BATTERY_SHIFT) & self.BATTERY_MASK
        item_idx = (packed >> self.ITEM_SHIFT) & self.ITEM_MASK

        base_channels = (base[:, :, np.newaxis] == self._base_channel_ids).astype(np.float32)
        battery_channel = (battery.astype(np.float32) / self.BATTERY_MAX)[:, :, np.newaxis]
        item_channels = (item_idx[:, :, np.newaxis] == self._item_channel_ids).astype(np.float32)

        return np.concatenate([base_channels, battery_channel, item_channels], axis=-1)

    # ------------------------------------------------------------------
    # Team perspective flip
    # ------------------------------------------------------------------

    def flip_team_perspective(self, graphic: np.ndarray) -> np.ndarray:
        """
        Convert a TeamA graphic to a TeamB graphic by swapping ally/enemy channels.
        Battery and item channels are team-agnostic and left untouched.

        Parameters
        ----------
        graphic : float32[H × W × n_graphic_channels]  TeamA perspective

        Returns
        -------
        float32[H × W × n_graphic_channels]  TeamB perspective
        """
        result = graphic.copy()
        result[..., self.SPAWN_ALLY]    = graphic[..., self.SPAWN_ENEMY]
        result[..., self.SPAWN_ENEMY]   = graphic[..., self.SPAWN_ALLY]
        result[..., self.STORAGE_ALLY]  = graphic[..., self.STORAGE_ENEMY]
        result[..., self.STORAGE_ENEMY] = graphic[..., self.STORAGE_ALLY]
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def item_channel(self, item_index: int) -> int:
        """Return the graphic channel index for a non-battery item (1-based, matches KnownItems)."""
        if not (1 <= item_index <= self.n_nonbattery_items):
            raise ValueError(f"item_index must be 1-{self.n_nonbattery_items}, got {item_index}")
        return self.N_BASE_CHANNELS + item_index

    def channel_name(self, channel: int) -> str:
        """Human-readable name for a graphic channel index. Useful for logging and visualization."""
        names = {
            self.VOID: "void", self.WALL: "wall",
            self.SITE_HUNTER: "site_hunter", self.SITE_CARRIER: "site_carrier",
            self.SPAWN_ALLY: "spawn_ally", self.SPAWN_ENEMY: "spawn_enemy",
            self.STORAGE_ALLY: "storage_ally", self.STORAGE_ENEMY: "storage_enemy",
            self.BATTERY: "battery",
        }
        if channel in names:
            return names[channel]
        if self.N_BASE_CHANNELS < channel <= self.N_BASE_CHANNELS + self.n_nonbattery_items:
            return f"item_{channel - self.N_BASE_CHANNELS}"
        return f"unknown_{channel}"
