"""Unit occupancy channels: cell placement, ally/enemy split, stacking, and frame consistency."""

from __future__ import annotations

import numpy as np
import torch

from blackout_env.env.team_frame import mirror_agent_states, mirror_graphic
from blackout_env.model.unit_channels import N_UNIT_CHANNELS, unit_channels

H = W = 24
N_UNITS, N_COLS = 10, 12
ALLY, ENEMY, ALLY_CARGO, ENEMY_CARGO = 0, 1, 2, 3


def _states(cells: list[tuple[int, int]], cargo: list[float] | None = None) -> torch.Tensor:
    states = torch.zeros(1, N_UNITS, N_COLS)
    for i, (row, col) in enumerate(cells):
        states[0, i, 0] = (col + 0.5) * 2.0 / W - 1.0
        states[0, i, 1] = 1.0 - (row + 0.5) * 2.0 / H
    states[0, :5, 2] = 1.0
    states[0, 5:, 2] = -1.0
    if cargo is not None:
        states[0, :, 4] = torch.tensor(cargo)
    return states


def test_units_land_on_their_own_cells():
    cells = [(2, 3), (10, 10), (23, 0), (0, 23), (5, 5), (7, 7), (8, 8), (9, 9), (11, 11), (12, 12)]
    channels = unit_channels(_states(cells), H, W)[0]

    assert channels.shape == (N_UNIT_CHANNELS, H, W)
    for unit, (row, col) in enumerate(cells):
        channel = ALLY if unit < 5 else ENEMY
        assert channels[channel, row, col] >= 1.0
    assert channels[ALLY].sum() == 5.0
    assert channels[ENEMY].sum() == 5.0


def test_stacked_units_count_rather_than_saturate():
    """Units pass through each other, so two on one cell is a real state, not a flag."""
    channels = unit_channels(_states([(4, 4)] * N_UNITS), H, W)[0]
    assert channels[ALLY, 4, 4] == 5.0
    assert channels[ENEMY, 4, 4] == 5.0


def test_cargo_channels_follow_the_battery_column():
    cargo = [0.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0]
    cells = [(6, 6)] * N_UNITS
    channels = unit_channels(_states(cells, cargo), H, W)[0]

    assert channels[ALLY_CARGO, 6, 6] == 0.5
    assert channels[ENEMY_CARGO, 6, 6] == 0.25
    assert channels[ALLY_CARGO].sum() == 0.5 and channels[ENEMY_CARGO].sum() == 0.25


def test_mirroring_the_state_mirrors_the_channels():
    """Derived after the team-frame mirror, so the two must commute (see team_frame.py)."""
    rng = np.random.default_rng(0)
    cells = [(int(r), int(c)) for r, c in rng.integers(0, H, size=(N_UNITS, 2))]
    states = _states(cells, list(rng.random(N_UNITS)))

    direct = unit_channels(torch.tensor(mirror_agent_states(states.numpy())), H, W)[0]
    # mirror_graphic works on [..., H, W, C]; channels here are [C, H, W]
    via_graphic = torch.tensor(
        mirror_graphic(unit_channels(states, H, W)[0].permute(1, 2, 0).numpy())
    ).permute(2, 0, 1)

    # the mirror swaps ally/enemy rows but not their team sign, so compare the summed occupancy
    assert torch.allclose(direct[ALLY] + direct[ENEMY], via_graphic[ALLY] + via_graphic[ENEMY])
    assert torch.allclose(direct[ALLY_CARGO] + direct[ENEMY_CARGO], via_graphic[ALLY_CARGO] + via_graphic[ENEMY_CARGO])


def test_sub_tile_positions_stay_distinguishable_at_4x():
    """The whole point of the 4x grid: two units in one map tile must not collapse."""
    from blackout_env.model.unit_channels import UNIT_GRID_SCALE

    fine = H * UNIT_GRID_SCALE
    # both inside map tile (5, 5), at opposite corners of it
    near = torch.zeros(1, N_UNITS, N_COLS)
    near[0, :, 2] = 1.0
    near[0, 0, 0] = (5 * UNIT_GRID_SCALE + 0.5) * 2.0 / fine - 1.0
    near[0, 0, 1] = 1.0 - (5 * UNIT_GRID_SCALE + 0.5) * 2.0 / fine
    far = near.clone()
    far[0, 0, 0] = (5 * UNIT_GRID_SCALE + 3.5) * 2.0 / fine - 1.0
    far[0, 0, 1] = 1.0 - (5 * UNIT_GRID_SCALE + 3.5) * 2.0 / fine

    fine_a = unit_channels(near, fine, fine)[0, ALLY]
    fine_b = unit_channels(far, fine, fine)[0, ALLY]
    assert not torch.equal(fine_a, fine_b)

    coarse_a = unit_channels(near, H, W)[0, ALLY]
    coarse_b = unit_channels(far, H, W)[0, ALLY]
    assert torch.equal(coarse_a, coarse_b)  # what painting straight onto 24x24 would lose


def test_graphic_encoder_folds_the_fine_grid_back_to_map_cells():
    from blackout_env.model.modules import GraphicEncoder
    from blackout_env.model.unit_channels import UNIT_GRID_SCALE

    encoder = GraphicEncoder(hidden_size=64)
    unit_map = unit_channels(_states([(3, 3)] * N_UNITS), H * UNIT_GRID_SCALE, W * UNIT_GRID_SCALE)
    assert unit_map.shape == (1, N_UNIT_CHANNELS, 96, 96)
    assert encoder.unit_stem(unit_map).shape == (1, 8, H, W)
    assert encoder(torch.zeros(1, 13, H, W), unit_map).shape == (1, 36, 64)


def test_model_forward_accepts_the_env_channel_count():
    from blackout_env.model.my_model import MyModel

    net = MyModel()
    graphic = torch.zeros(2, 13, H, W)  # what MyObsPreprocessor produces, before unit channels
    q_values, quantiles, tau, vision_latent, global_latent = net(
        graphic, torch.zeros(2, 4), _states([(3, 3)] * N_UNITS).expand(2, -1, -1), n_quantiles=4
    )
    assert q_values.shape == (2, N_UNITS, 8)
    assert quantiles.shape == (2, N_UNITS, 4, 8)
