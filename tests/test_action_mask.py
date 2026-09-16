"""Wall mask: geometry, the no-corner-cutting rule, and agreement with the heuristics' rules."""

from __future__ import annotations

import numpy as np
import torch

from blackout_env.model.action_mask import DIR_CELL, legal_direction_mask, masked_greedy, unit_cells
from blackout_env.model.my_policy import DIRECTION_VECTORS

H = W = 24
WALL = 1
N_UNITS, N_CHANNELS, N_COLS = 10, 13, 12


def _state(cells: list[tuple[int, int]]) -> torch.Tensor:
    """agent_states whose units sit at the centre of the given (row, col) cells."""
    states = torch.zeros(1, N_UNITS, N_COLS)
    for i, (row, col) in enumerate(cells):
        states[0, i, 0] = (col + 0.5) * 2.0 / W - 1.0
        states[0, i, 1] = 1.0 - (row + 0.5) * 2.0 / H
    states[0, :5, 2] = 1.0
    states[0, 5:, 2] = -1.0
    return states


def _graphic(walls: list[tuple[int, int]]) -> torch.Tensor:
    graphic = torch.zeros(1, N_CHANNELS, H, W)
    for row, col in walls:
        graphic[0, WALL, row, col] = 1.0
    return graphic


def test_dir_cell_matches_direction_vectors():
    for i, vector in enumerate(DIRECTION_VECTORS):
        drow, dcol = DIR_CELL[i]
        assert dcol == int(np.sign(round(float(vector[0]), 3)))
        assert drow == -int(np.sign(round(float(vector[1]), 3)))


def test_unit_cells_round_trip():
    cells = [(3, 20), (0, 0), (23, 23), (12, 7), (5, 5)] * 2
    row, col = unit_cells(_state(cells), H, W)
    assert [(int(r), int(c)) for r, c in zip(row[0], col[0])] == cells


def test_wall_to_the_east_is_masked_out():
    cells = [(10, 10)] * N_UNITS
    mask = legal_direction_mask(_graphic([(10, 11)]), _state(cells))[0, 0]
    assert not mask[0]  # east
    assert mask[2] and mask[4] and mask[6]  # north, west, south


def test_diagonal_needs_both_flanks():
    cells = [(10, 10)] * N_UNITS
    # north-east diagonal with the east cell walled: the corner may not be cut
    mask = legal_direction_mask(_graphic([(10, 11)]), _state(cells))[0, 0]
    assert not mask[1]
    # with both flanks open the same diagonal is legal
    assert legal_direction_mask(_graphic([]), _state(cells))[0, 0][1]


def test_map_edge_is_masked_out():
    mask = legal_direction_mask(_graphic([]), _state([(0, 0)] * N_UNITS))[0, 0]
    assert not mask[2] and not mask[4]  # north and west leave the grid
    assert mask[0] and mask[6]  # east and south stay inside


def test_fully_walled_unit_keeps_every_direction():
    """An all -inf Q row would break argmax, so a boxed-in unit falls back to no mask."""
    walls = [(9 + dr, 9 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    mask = legal_direction_mask(_graphic(walls), _state([(9, 9)] * N_UNITS))[0, 0]
    assert bool(mask.all())


def test_masked_greedy_skips_the_best_but_blocked_direction():
    cells = [(10, 10)] * N_UNITS
    q = torch.zeros(1, N_UNITS, 8)
    q[0, 0, 0] = 5.0  # east: highest, but walled
    q[0, 0, 6] = 1.0  # south: best legal
    graphic, states = _graphic([(10, 11)]), _state(cells)

    assert int(q[0, 0].argmax()) == 0
    assert int(masked_greedy(q, graphic, states)[0, 0]) == 6


def test_masked_greedy_accepts_own_team_slice():
    cells = [(10, 10)] * N_UNITS
    q = torch.zeros(1, 5, 8)
    q[0, 0, 0] = 5.0
    q[0, 0, 2] = 1.0
    chosen = masked_greedy(q, _graphic([(10, 11)]), _state(cells))
    assert chosen.shape == (1, 5)
    assert int(chosen[0, 0]) == 2


def test_agrees_with_the_heuristic_walkability_rule():
    """Same wall rule the heuristics navigate by, checked over random maps."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        wall_grid = rng.random((H, W)) < 0.25
        graphic = torch.zeros(1, N_CHANNELS, H, W)
        graphic[0, WALL] = torch.tensor(wall_grid, dtype=torch.float32)
        cells = [(int(r), int(c)) for r, c in rng.integers(1, H - 1, size=(N_UNITS, 2))]
        mask = legal_direction_mask(graphic, _state(cells))[0]

        for unit, (row, col) in enumerate(cells):
            walkable = ~wall_grid
            expected = []
            for drow, dcol in DIR_CELL:
                nr, nc = row + drow, col + dcol
                legal = 0 <= nr < H and 0 <= nc < W and walkable[nr, nc]
                if legal and drow != 0 and dcol != 0:
                    legal = walkable[row + drow, col] and walkable[row, col + dcol]
                expected.append(legal)
            if any(expected):
                assert list(mask[unit].numpy()) == expected
