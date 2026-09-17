"""
Features the environment does not hand over, derived from the observation it does.

Everything here is a pure function of `graphic` + `agent_states`, computed inside
MyModel.forward, so nothing has to be re-collected and the same values reach training,
collection and evaluation through one code path. Derivation happens after the team-frame
mirroring (blackout_env.env.team_frame), so these come out in whatever frame the batch is in.

Each entry exists because a measured failure traced back to information the network could not
practically recover for itself (docs/run6_diagnosis_20260916.md, and the Run 7 stall taxonomy):

unit occupancy      The 13 env channels contain no units at all; positions reached the network
                    only as 10 coordinate rows, leaving the convolutional path unable to relate
                    "who is standing where" to the map. Counts, not flags: units pass through
                    each other, so two on a cell is a real state.

storage capacity    Unity rejects an over-capacity deposit *whole* and silently (Storage.cs:
                    "storage is full; nop.") -- no event, no reward, no observable change. The
                    final Run 7 checkpoint waited at a storage that could not take its cargo
                    31.5 times per 1000 unit-ticks, and in 100% of those another storage had
                    room. Computing the free capacity of a storage component from the raw
                    channels means segmenting the ally-storage mask and summing (10 - amount)
                    over its tiles, skipping tiles a special item has blocked -- exactly what
                    StrategicHeuristic._storage_target does explicitly, and not something a few
                    convolutions can express.

fetchable battery   Channel 8 conflates three different things: a battery to pick up, one
                    already banked in own storage (which Unity refuses to let you pick up), and
                    one in the enemy's storage (which is worth stealing -- it swings the score
                    twice). Separating them is a product of two channels, which a 1x1 convolution
                    cannot represent.

wall patch          97.5% of the final checkpoint's failed moves were into a wall a sidestep
                    would have cleared: the snag is a convex corner, and escaping it needs the
                    unit's position *within* its tile against the immediate wall geometry. The
                    map path only ever sees tile-quantized positions, and pools 24x24 to 6x6
                    before the unit tokens meet it. Walkability only -- tile identity is already
                    in the map channels at the same resolution; what the patch adds over them is
                    sub-tile relative geometry, and only walkability needs it.

                    Both parts must be CONTINUOUS in the unit's position. The first version read
                    a 3x3 patch around the rounded cell plus the offset within it (in [-0.5,
                    0.5]); crossing a tile boundary shifted the patch by a whole tile and flipped
                    the offset from +0.5 to -0.5. On the Run 9 40k checkpoint the greedy action
                    pointed back at the boundary from both sides for 19.3% of Hunters placed
                    there (0.2% at a tile centre) -- the two-cell oscillation seen in GUI -- and
                    computing these features from the unmoved position removed it (0.2%). The
                    offset alone accounted for most of it (19.3% -> 4.1% when frozen).
                    So walkability is now sampled bilinearly at fixed offsets around the true
                    position.

                    There is deliberately no in-tile position feature. A sin/cos tile phase (the
                    first continuous replacement for the offset) turns a full cycle every tile,
                    so a Hunter moving 0.24 tiles per decision swung it by 1.10 on average per
                    tick (range [-1, 1]) -- three times the old offset's churn -- while in open
                    ground where it swings most, position within a tile decides nothing. Where it
                    does matter, near walls, the bilinear samples already carry it (at most 0.31
                    change per Hunter tick), and the raw pos columns stay on the token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_UNIT_CHANNELS = 4
N_DERIVED_MAP_CHANNELS = N_UNIT_CHANNELS + 2  # + storage free capacity + fetchable battery
PATCH = 5  # wall samples per side, PATCH_SPACING tiles apart, centred on the unit's true position
PATCH_SPACING = 0.5
N_PATCH_FEATURES = PATCH * PATCH

VOID, WALL = 0, 1
STORAGE_ALLY, STORAGE_ENEMY, BATTERY = 6, 7, 8
FIRST_SPECIAL = 9
TEAM_SIGN_COL = 2
CARGO_COL = 4  # agent_states: [pos_x, pos_y, team, item_none, item_battery, ...]

MAX_ITEM_AMOUNT = 10.0  # Unity ItemData.MaxItemAmount
BATTERY_SCALE = 15.0  # channel 8 stores amount / 15
LABEL_PROPAGATION_STEPS = 8  # > the largest storage component's diameter (measured: 5)


def unit_cells(agent_states: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) grid cell of every unit in a [B, N_UNITS, D] agent_states tensor."""
    x, y = agent_states[..., 0], agent_states[..., 1]
    row = torch.round((1.0 - y) * 0.5 * height - 0.5).long().clamp(0, height - 1)
    col = torch.round((x + 1.0) * 0.5 * width - 0.5).long().clamp(0, width - 1)
    return row, col


def unit_channels(agent_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, 4, H, W]: ally count, enemy count, ally carried battery, enemy carried battery."""
    batch = agent_states.shape[0]
    row, col = unit_cells(agent_states, height, width)
    flat_cell = row * width + col  # [B, N_UNITS]

    ally = (agent_states[..., TEAM_SIGN_COL] > 0).to(agent_states.dtype)
    cargo = agent_states[..., CARGO_COL]
    weights = torch.stack([ally, 1.0 - ally, ally * cargo, (1.0 - ally) * cargo], dim=1)  # [B, 4, N]

    channels = torch.zeros(batch, N_UNIT_CHANNELS, height * width, dtype=agent_states.dtype, device=agent_states.device)
    channels.scatter_add_(2, flat_cell.unsqueeze(1).expand(-1, N_UNIT_CHANNELS, -1), weights)
    return channels.view(batch, N_UNIT_CHANNELS, height, width)


def _component_labels(mask: torch.Tensor) -> torch.Tensor:
    """
    [B, 1, H, W] int labels: every cell of one 4-connected blob carries the same value.

    Each cell starts labelled with its own flat index and repeatedly takes the maximum label
    among its 4-neighbours inside the mask, which converges once the iteration count reaches the
    blob's diameter. Storage blobs here are 4 tiles with a diameter of at most 5, so a fixed
    LABEL_PROPAGATION_STEPS is exact rather than a cutoff -- and it stays on the GPU, unlike a
    flood fill.
    """
    batch, _, height, width = mask.shape
    index = torch.arange(height * width, device=mask.device, dtype=mask.dtype).view(1, 1, height, width)
    labels = index * mask
    for _ in range(LABEL_PROPAGATION_STEPS):
        spread = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
        # 4-connectivity, matching StrategicHeuristic._components: a diagonal neighbour must not
        # join two blobs that only touch at a corner.
        orthogonal = torch.maximum(
            F.max_pool2d(labels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            F.max_pool2d(labels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
        )
        labels = torch.minimum(spread, orthogonal) * mask
    return labels.long()


def storage_free_capacity(graphic: torch.Tensor) -> torch.Tensor:
    """
    [B, 1, H, W]: free battery slots of the whole ally-storage component a tile belongs to,
    normalized by MAX_ITEM_AMOUNT, and zero off storage.

    Per-tile capacity follows Unity's rule (Storage.MergeCalculator): a tile holding a special
    item takes nothing, otherwise it accepts MAX_ITEM_AMOUNT minus what it already holds. The
    component total is what matters because a deposit is all-or-nothing.
    """
    batch, _, height, width = graphic[:, :1].shape
    storage = (graphic[:, STORAGE_ALLY : STORAGE_ALLY + 1] > 0.5).to(graphic.dtype)
    blocked = (graphic[:, FIRST_SPECIAL:] > 0.5).any(dim=1, keepdim=True).to(graphic.dtype)
    amount = graphic[:, BATTERY : BATTERY + 1] * BATTERY_SCALE
    per_tile = (MAX_ITEM_AMOUNT - amount).clamp(min=0.0) * storage * (1.0 - blocked)

    labels = _component_labels(storage)
    flat_labels = labels.view(batch, -1)
    totals = torch.zeros(batch, height * width, dtype=graphic.dtype, device=graphic.device)
    totals.scatter_add_(1, flat_labels, per_tile.view(batch, -1))
    component = totals.gather(1, flat_labels).view(batch, 1, height, width)
    return component * storage / MAX_ITEM_AMOUNT


def fetchable_battery(graphic: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W]: battery amount a unit could actually pick up -- loose, or in enemy storage.

    Unity's ItemObject.IsInteractable refuses a pickup from the unit's own region, so a battery
    banked in own storage is not a target; one in the enemy's storage is, and taking it moves the
    score twice (the victim loses it, the thief banks it).
    """
    own_storage = (graphic[:, STORAGE_ALLY : STORAGE_ALLY + 1] > 0.5).to(graphic.dtype)
    return graphic[:, BATTERY : BATTERY + 1] * (1.0 - own_storage)


def derived_map_channels(graphic: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
    """[B, 6, H, W] appended to the env's own channels before the encoder."""
    height, width = graphic.shape[-2:]
    return torch.cat(
        [
            unit_channels(agent_states, height, width),
            storage_free_capacity(graphic),
            fetchable_battery(graphic),
        ],
        dim=1,
    )


def local_wall_features(graphic: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
    """
    [B, N_UNITS, N_PATCH_FEATURES] appended to each unit's agent_states row: PATCH x PATCH
    walkability samples (1 = blocked, off-map counts as blocked), taken PATCH_SPACING tiles apart
    around the unit's true position, row-major from the north-west, each bilinearly interpolated
    between the four surrounding tile centres -- continuous in the unit's position (see the module
    docstring for why that matters, and why there is no in-tile position feature).

    Computed for all ten units, not just own team: the frame is symmetric that way, and where an
    enemy is pinned against geometry is as informative as where you are.
    """
    batch, n_units = agent_states.shape[:2]
    height, width = graphic.shape[-2:]
    dtype = agent_states.dtype
    pad = int(math.ceil(PATCH // 2 * PATCH_SPACING)) + 1
    blocked = (graphic[:, WALL] >= 0.5).to(dtype)
    padded = F.pad(blocked, (pad, pad, pad, pad), value=1.0)  # [B, H + 2pad, W + 2pad]
    padded_w = width + 2 * pad

    # Tile-centre coordinates: tile (r, c) sits at exactly (r, c).
    x, y = agent_states[..., 0], agent_states[..., 1]
    row = (1.0 - y) * 0.5 * height - 0.5  # [B, N]
    col = (x + 1.0) * 0.5 * width - 0.5

    steps = (torch.arange(PATCH, device=agent_states.device, dtype=dtype) - PATCH // 2) * PATCH_SPACING
    sample_row = (row.unsqueeze(-1) + steps).unsqueeze(-1).expand(-1, -1, PATCH, PATCH)  # [B, N, P, P]
    sample_col = (col.unsqueeze(-1) + steps).unsqueeze(-2).expand(-1, -1, PATCH, PATCH)

    r = (sample_row + pad).clamp(0, height + 2 * pad - 1)
    c = (sample_col + pad).clamp(0, padded_w - 1)
    r0 = r.floor().clamp(max=height + 2 * pad - 2)
    c0 = c.floor().clamp(max=padded_w - 2)
    fr, fc = r - r0, c - c0
    r0, c0 = r0.long(), c0.long()

    flat = padded.view(batch, -1)

    def at(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return flat.gather(1, (rows * padded_w + cols).reshape(batch, -1)).view_as(fr)

    return (
        at(r0, c0) * (1 - fr) * (1 - fc)
        + at(r0, c0 + 1) * (1 - fr) * fc
        + at(r0 + 1, c0) * fr * (1 - fc)
        + at(r0 + 1, c0 + 1) * fr * fc
    ).reshape(batch, n_units, PATCH * PATCH)
