"""
Unit occupancy channels, rasterized from agent_states into the graphic grid.

The 13-channel graphic observation carries no units at all (8 terrain one-hots, a battery
scalar, 4 item one-hots); unit positions reach the network only as 10 coordinate rows in
agent_states. That leaves the convolutional encoder unable to relate "who is standing where"
to the map it does see, and the vision tokens and unit tokens only meet in the attention trunk,
already pooled to 6x6 tokens.

Run 6's checkpoint lost cargo to enemies 28 times per 10 matches against the heuristics' 10
(docs/run6_diagnosis_20260916.md §4), which is exactly the judgement -- "is an enemy near my
carrier", "where is the enemy carrier" -- that wants unit positions laid out on the map rather
than as a separate list of coordinates.

Four channels: ally count, enemy count, ally carried battery, enemy carried battery. Counts,
not presence flags: units can occupy the same cell (they do not collide -- measured over 11.5k
unit-ticks, a unit with another unit in its target cell moved every time), so two units on one
cell is a real state worth distinguishing. Ally/enemy comes from agent_states' team-sign column,
so this works unchanged in the canonical team frame (blackout_env.env.team_frame).

Computed on the fly from agent_states rather than stored: they are a pure function of data the
replay buffer already holds, so nothing has to be re-collected and the 30GB dataset does not
grow. Because they are derived after the team-frame mirroring, they are automatically in
whatever frame the rest of the batch is in.
"""

from __future__ import annotations

import torch

N_UNIT_CHANNELS = 4
TEAM_SIGN_COL = 2
BATTERY_COL = 4  # agent_states: [pos_x, pos_y, team, item_none, item_battery, ...]


def unit_channels(agent_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    [B, 4, H, W] occupancy maps for a [B, N_UNITS, D] agent_states table:
    ally count, enemy count, ally carried battery, enemy carried battery.

    Cell indexing matches MyObsPreprocessor/the heuristics: row counts down from the top while
    world y grows upward, so row = (1 - y) * H / 2 - 0.5 and col = (x + 1) * W / 2 - 0.5.
    """
    batch = agent_states.shape[0]
    x, y = agent_states[..., 0], agent_states[..., 1]
    row = torch.round((1.0 - y) * 0.5 * height - 0.5).long().clamp(0, height - 1)
    col = torch.round((x + 1.0) * 0.5 * width - 0.5).long().clamp(0, width - 1)
    flat_cell = row * width + col  # [B, N_UNITS]

    ally = (agent_states[..., TEAM_SIGN_COL] > 0).to(agent_states.dtype)  # [B, N_UNITS]
    cargo = agent_states[..., BATTERY_COL]
    weights = torch.stack([ally, 1.0 - ally, ally * cargo, (1.0 - ally) * cargo], dim=1)  # [B, 4, N]

    channels = torch.zeros(batch, N_UNIT_CHANNELS, height * width, dtype=agent_states.dtype, device=agent_states.device)
    channels.scatter_add_(2, flat_cell.unsqueeze(1).expand(-1, N_UNIT_CHANNELS, -1), weights)
    return channels.view(batch, N_UNIT_CHANNELS, height, width)
