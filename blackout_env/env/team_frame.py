"""
Team-canonical observation frame.

The map is built by reflecting Team A's layout across the y = x diagonal (see the Unity
`RegionData.CreateSymmetricRegion` / `ProceduralMapGenerator` pair: `posB = (posA.y, posA.x)`),
so Team B's world is Team A's world seen through that reflection. The observation pipeline,
however, only relabels channels for Team B (`MyObsPreprocessor.flip_team_perspective` swaps
SPAWN_ALLY/ENEMY and STORAGE_ALLY/ENEMY) and leaves the grid, the agent_states row order and
the action frame in raw world coordinates. A Team B agent therefore sees its own spawn in the
opposite corner, finds its own units in rows 5-9 instead of 0-4 (and MyModel keys a learned
identity embedding on that fixed slot index), and has to map "toward my storage" onto the
opposite compass direction. Net effect: the network learns Team A's task and Team B's task
separately, each from half the data.

These helpers apply the missing reflection, mapping a Team B view to the frame Team A sees:

  grid          (row, col) -> (H-1-col, W-1-row)   -- anti-transpose; the grid's row axis is
                                                      flipped relative to world y, so the world
                                                      x <-> y swap is the ANTI-diagonal here
  position      (x, y)     -> (y, x)
  unit rows     k          -> (k + 5) % 10         -- own team moves to rows 0-4
  action index  i          -> (2 - i) % 8          -- DIRECTION_VECTORS angle t -> 90deg - t

Every one of those is an involution (the grid is square, and shifting 10 unit rows by 5 twice
is the identity), so a single `mirror_*` family serves both directions: applying it to a Team B
view canonicalizes it, and applying it to a canonical-frame action gives the world-frame action
back. Team A views are already canonical and must not be touched.

The map symmetry these rely on is verified against real observations in
tests/test_team_frame.py rather than assumed.
"""

from __future__ import annotations

import numpy as np

N_UNITS = 10
N_TEAM = 5
POS_COLS = (0, 1)
TEAM_SIGN_COL = 2

# DIRECTION_VECTORS[i] points at angle i * 45deg; reflecting across the world y = x diagonal
# maps angle t to 90deg - t, i.e. index i to (2 - i) % 8.
DIR_MIRROR = np.array([(2 - i) % 8 for i in range(8)], dtype=np.int64)


def mirror_graphic(graphic: np.ndarray) -> np.ndarray:
    """Anti-transpose the spatial axes of a [..., H, W, C] graphic observation."""
    return np.flip(np.swapaxes(graphic, -3, -2), axis=(-3, -2)).copy()


def mirror_agent_states(agent_states: np.ndarray) -> np.ndarray:
    """Swap the x/y position columns and roll the unit axis of a [..., N_UNITS, D] table."""
    out = np.roll(agent_states, -N_TEAM, axis=-2).copy()
    out[..., POS_COLS[0]], out[..., POS_COLS[1]] = (
        out[..., POS_COLS[1]].copy(),
        out[..., POS_COLS[0]].copy(),
    )
    return out


def mirror_action_idx(action_idx: np.ndarray) -> np.ndarray:
    """Reflect discrete compass indices in a per-unit [..., N_UNITS] table and roll the units."""
    return DIR_MIRROR[np.roll(action_idx, -N_TEAM, axis=-1)]


def mirror_direction_idx(action_idx: np.ndarray | int) -> np.ndarray | int:
    """Reflect a bare compass index (or array of them), with no unit-axis reordering."""
    if isinstance(action_idx, (int, np.integer)):
        return int(DIR_MIRROR[action_idx])
    return DIR_MIRROR[action_idx]


def is_team_b_view(agent_states: np.ndarray) -> bool:
    """True when this observation is Team B's view, i.e. row 0 (physical unit 0) is an enemy.

    Reads the same team-sign column `qmix_trainer._own_team_rows` uses, so a caller never has
    to thread the team identity through separately -- the observation itself carries it.
    """
    return bool(np.asarray(agent_states)[..., 0, TEAM_SIGN_COL] < 0)


def canonical_obs(view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One agent's observation dict mapped into the canonical frame.

    A team-A view is returned unchanged; a team-B view comes back mirrored. `team_state` holds
    no spatial information (scores are already reordered per perspective by
    MyObsPreprocessor.preprocess_team_states), so it passes through either way.
    """
    if not is_team_b_view(view["agent_states"]):
        return view
    return {
        **view,
        "graphic": mirror_graphic(view["graphic"]),
        "agent_states": mirror_agent_states(view["agent_states"]),
    }


def canonical_unit_row(unit_index: int, team_b: bool) -> int:
    """Row of `unit_index` (physical 0-9) in the canonical frame."""
    return (unit_index + N_TEAM) % N_UNITS if team_b else unit_index
