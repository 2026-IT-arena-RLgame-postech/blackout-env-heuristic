"""
Walkability mask over the 8 compass actions.

A unit that commands a direction into a wall does not move, so the observation barely changes
and a deterministic greedy policy re-picks the same direction forever -- measured on Run 6's
final checkpoint, a team spent stretches of up to 381 ticks with 40%+ of its units in that
state, and ~25% of all unit-ticks overall (heuristic opponents: ~1.5%). Masking those
directions out of the argmax removes the loop outright (measured 24.7% -> 0.6% blocked
unit-ticks) and, with it, the blocked-penalty term that otherwise dominates the return.

The mask is applied wherever a *greedy choice* is made -- acting, collecting, and the Double-DQN
bootstrap action -- but never to the Q-value of an action that was actually taken and stored: a
heuristic in the dataset does occasionally walk into a wall, and those transitions still have to
train the value of the action they really executed.

Geometry matches the heuristics' navigation rules (blackout_env/heuristics/strategic.py): a cell
is walkable iff its WALL channel is below 0.5, and a diagonal step additionally requires both
flanking orthogonal cells to be walkable (no cutting a blocked corner).
"""

from __future__ import annotations

import math

import numpy as np
import torch

WALL_CHANNEL = 1
WALL_THRESHOLD = 0.5

# (drow, dcol) per compass index, for the same 8 evenly spaced directions starting at +x that
# my_policy.DIRECTION_VECTORS defines (recomputed here rather than imported, to keep this module
# importable from my_policy itself). Graphic rows are top-down while world y grows upward, so a
# +y action is -1 in rows (see MyObsPreprocessor / strategic.py's waypoint math).
DIR_CELL = np.array(
    [
        [-int(np.sign(round(math.sin(i * math.pi / 4), 3))), int(np.sign(round(math.cos(i * math.pi / 4), 3)))]
        for i in range(8)
    ],
    dtype=np.int64,
)


def unit_cells(agent_states: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) grid cell of every unit in a [B, N_UNITS, D] agent_states tensor."""
    x, y = agent_states[..., 0], agent_states[..., 1]
    row = torch.round((1.0 - y) * 0.5 * height - 0.5).long().clamp(0, height - 1)
    col = torch.round((x + 1.0) * 0.5 * width - 0.5).long().clamp(0, width - 1)
    return row, col


def legal_direction_mask(graphic: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
    """
    Bool [B, N_UNITS, 8]: True where that unit's step lands on a walkable cell.

    graphic: [B, C, H, W] (the layout MyModel takes, channels first).
    A unit with no legal direction at all (fully walled in) gets an all-True row, so callers can
    apply the mask unconditionally without producing an all -inf Q row.
    """
    walkable = graphic[:, WALL_CHANNEL] < WALL_THRESHOLD  # [B, H, W]
    batch, height, width = walkable.shape
    row, col = unit_cells(agent_states, height, width)  # [B, N_UNITS]
    deltas = torch.as_tensor(DIR_CELL, device=graphic.device)  # [8, 2]

    def at(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        flat = r.clamp(0, height - 1) * width + c.clamp(0, width - 1)
        gathered = walkable.reshape(batch, -1).gather(1, flat.reshape(batch, -1)).reshape(r.shape)
        return gathered & inside

    target_row = row.unsqueeze(-1) + deltas[:, 0]  # [B, N_UNITS, 8]
    target_col = col.unsqueeze(-1) + deltas[:, 1]
    mask = at(target_row, target_col)

    diagonal = (deltas[:, 0] != 0) & (deltas[:, 1] != 0)  # [8]
    flank_row = at(target_row, col.unsqueeze(-1).expand_as(target_col))
    flank_col = at(row.unsqueeze(-1).expand_as(target_row), target_col)
    mask = mask & torch.where(diagonal, flank_row & flank_col, torch.ones_like(mask))

    return torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))


def world_legal_mask(graphic: np.ndarray, agent_states: np.ndarray) -> np.ndarray:
    """Bool [N_UNITS, 8] walkability mask for one raw observation (numpy, channels-last).

    Same rule as legal_direction_mask, for the single-observation inference path; the mask is
    purely geometric, so it is identical in the world and canonical frames.
    """
    mask = legal_direction_mask(
        torch.as_tensor(graphic, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
        torch.as_tensor(agent_states, dtype=torch.float32).unsqueeze(0),
    )
    return mask.squeeze(0).numpy()


def masked_greedy(q_values: torch.Tensor, graphic: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
    """argmax over q_values [B, N, 8] restricted to walkable directions of those N unit rows.

    `q_values` may already be narrowed to one team (N = N_TEAM); the mask is then narrowed the
    same way, which only works because own-team rows are the leading block in the canonical
    frame (see blackout_env.env.team_frame).
    """
    mask = legal_direction_mask(graphic, agent_states)[:, : q_values.shape[1]]
    return q_values.masked_fill(~mask, float("-inf")).argmax(dim=-1)
