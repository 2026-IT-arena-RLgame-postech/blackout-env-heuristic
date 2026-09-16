"""Derived observation features: unit occupancy, storage capacity, fetchable batteries, wall patch."""

from __future__ import annotations

import numpy as np
import torch

from blackout_env.env.team_frame import mirror_agent_states, mirror_graphic
from blackout_env.heuristics.strategic import StrategicHeuristic
from blackout_env.model.derived_obs import (
    BATTERY,
    MAX_ITEM_AMOUNT,
    N_DERIVED_MAP_CHANNELS,
    N_PATCH_FEATURES,
    STORAGE_ALLY,
    STORAGE_ENEMY,
    WALL,
    derived_map_channels,
    fetchable_battery,
    local_wall_features,
    storage_free_capacity,
    unit_channels,
)

H = W = 24
N_UNITS, N_COLS = 10, 12
ALLY, ENEMY, ALLY_CARGO, ENEMY_CARGO = 0, 1, 2, 3
FIRST_SPECIAL = 9


def _graphic() -> torch.Tensor:
    return torch.zeros(1, 13, H, W)


def _states(cells: list[tuple[float, float]]) -> torch.Tensor:
    """agent_states placing each unit at a (row, col) that may be fractional."""
    states = torch.zeros(1, N_UNITS, N_COLS)
    for i, (row, col) in enumerate(cells):
        states[0, i, 0] = (col + 0.5) * 2.0 / W - 1.0
        states[0, i, 1] = 1.0 - (row + 0.5) * 2.0 / H
    states[0, :5, 2] = 1.0
    states[0, 5:, 2] = -1.0
    return states


# ---------------------------------------------------------------- unit occupancy


def test_units_land_on_their_own_cells_and_stack():
    cells = [(2, 3), (10, 10), (23, 0), (0, 23), (5, 5)] * 2
    channels = unit_channels(_states(cells), H, W)[0]
    assert channels.shape == (4, H, W)
    for unit, (row, col) in enumerate(cells):
        assert channels[ALLY if unit < 5 else ENEMY, row, col] >= 1.0

    stacked = unit_channels(_states([(4, 4)] * N_UNITS), H, W)[0]
    assert stacked[ALLY, 4, 4] == 5.0 and stacked[ENEMY, 4, 4] == 5.0


def test_cargo_channels_follow_the_battery_column():
    states = _states([(6, 6)] * N_UNITS)
    states[0, 1, 4] = 0.5
    states[0, 6, 4] = 0.25
    channels = unit_channels(states, H, W)[0]
    assert channels[ALLY_CARGO, 6, 6] == 0.5
    assert channels[ENEMY_CARGO, 6, 6] == 0.25


# ---------------------------------------------------------------- storage capacity


def test_component_capacity_is_summed_over_the_whole_blob():
    graphic = _graphic()
    for col in (5, 6, 7):  # one 3-tile component
        graphic[0, STORAGE_ALLY, 5, col] = 1.0
    graphic[0, BATTERY, 5, 5] = 10 / 15  # full
    graphic[0, BATTERY, 5, 6] = 4 / 15   # 6 free

    capacity = storage_free_capacity(graphic)[0, 0]
    expected = (0 + 6 + 10) / MAX_ITEM_AMOUNT
    for col in (5, 6, 7):
        assert abs(float(capacity[5, col]) - expected) < 1e-5
    assert float(capacity[5, 8]) == 0.0  # off storage


def test_separate_components_do_not_pool_their_capacity():
    graphic = _graphic()
    graphic[0, STORAGE_ALLY, 5, 5] = 1.0
    graphic[0, BATTERY, 5, 5] = 10 / 15  # full component
    graphic[0, STORAGE_ALLY, 18, 18] = 1.0  # a separate, empty one

    capacity = storage_free_capacity(graphic)[0, 0]
    assert float(capacity[5, 5]) == 0.0
    assert abs(float(capacity[18, 18]) - 1.0) < 1e-5


def test_components_touching_only_at_a_corner_stay_separate():
    """4-connectivity, matching StrategicHeuristic._components."""
    graphic = _graphic()
    graphic[0, STORAGE_ALLY, 5, 5] = 1.0
    graphic[0, STORAGE_ALLY, 6, 6] = 1.0
    graphic[0, BATTERY, 5, 5] = 10 / 15
    capacity = storage_free_capacity(graphic)[0, 0]
    assert float(capacity[5, 5]) == 0.0
    assert abs(float(capacity[6, 6]) - 1.0) < 1e-5


def test_a_special_item_blocks_its_tile_for_batteries():
    graphic = _graphic()
    for col in (5, 6):
        graphic[0, STORAGE_ALLY, 5, col] = 1.0
    graphic[0, FIRST_SPECIAL, 5, 6] = 1.0  # tile blocked by a special
    assert abs(float(storage_free_capacity(graphic)[0, 0, 5, 5]) - 1.0) < 1e-5


def test_capacity_matches_the_heuristic_on_random_maps():
    """The network should get exactly the number StrategicHeuristic._storage_target computes."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        graphic = _graphic()
        mask = rng.random((H, W)) < 0.06
        graphic[0, STORAGE_ALLY] = torch.tensor(mask, dtype=torch.float32)
        amounts = np.where(mask & (rng.random((H, W)) < 0.5), rng.integers(1, 11, (H, W)), 0)
        graphic[0, BATTERY] = torch.tensor(amounts / 15.0, dtype=torch.float32)

        capacity = storage_free_capacity(graphic)[0, 0].numpy()
        numpy_graphic = graphic[0].permute(1, 2, 0).numpy()
        for component in StrategicHeuristic._components(mask):
            expected = sum(max(0, 10 - int(round(numpy_graphic[y, x, BATTERY] * 15))) for y, x in component)
            for y, x in component:
                assert abs(capacity[y, x] * MAX_ITEM_AMOUNT - expected) < 1e-3


# ---------------------------------------------------------------- fetchable battery


def test_own_storage_batteries_are_not_fetchable_but_enemy_ones_are():
    graphic = _graphic()
    graphic[0, BATTERY, 5, 5] = 0.4   # loose
    graphic[0, BATTERY, 6, 6] = 0.4
    graphic[0, STORAGE_ALLY, 6, 6] = 1.0   # banked: Unity refuses a pickup from your own region
    graphic[0, BATTERY, 7, 7] = 0.4
    graphic[0, STORAGE_ENEMY, 7, 7] = 1.0  # stealable

    fetchable = fetchable_battery(graphic)[0, 0]
    assert abs(float(fetchable[5, 5]) - 0.4) < 1e-6
    assert float(fetchable[6, 6]) == 0.0
    assert abs(float(fetchable[7, 7]) - 0.4) < 1e-6


# ---------------------------------------------------------------- wall patch


PATCH_CENTRE = 2  # index of the unit's own position along each side of the 5x5 sample grid


def _patch(features: torch.Tensor) -> torch.Tensor:
    return features[..., :25].reshape(*features.shape[:-1], 5, 5)


def test_wall_samples_at_tile_centres_read_the_tiles():
    graphic = _graphic()
    graphic[0, WALL, 10, 11] = 1.0  # east of the unit
    features = local_wall_features(graphic, _states([(10, 10)] * N_UNITS))[0, 0]
    assert features.shape == (N_PATCH_FEATURES,)
    patch = _patch(features)
    assert abs(float(patch[2, 4]) - 1.0) < 1e-4  # one tile east: the wall itself
    assert abs(float(patch[2, 3]) - 0.5) < 1e-4  # half a tile east: halfway to it
    assert abs(float(patch[2, 2])) < 1e-4        # the unit's own position
    assert abs(float(patch[0, 2])) < 1e-4        # one tile north: open


def test_outside_the_map_counts_as_blocked():
    patch = _patch(local_wall_features(_graphic(), _states([(0, 0)] * N_UNITS))[0, 0])
    # positions round-trip through normalized coordinates, so allow float error
    assert abs(float(patch[0, 2]) - 1.0) < 1e-4 and abs(float(patch[0, 0]) - 1.0) < 1e-4  # north is off-map
    assert abs(float(patch[2, 0]) - 1.0) < 1e-4  # one tile west is off-map
    assert abs(float(patch[4, 4])) < 1e-4        # one tile south-east is inside


def test_in_tile_phase_is_periodic():
    at_centre = local_wall_features(_graphic(), _states([(10.0, 7.0)] * N_UNITS))[0, 0, 25:]
    assert torch.allclose(at_centre, torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-5)
    at_boundary = local_wall_features(_graphic(), _states([(10.5, 7.0)] * N_UNITS))[0, 0, 25:]
    assert abs(float(at_boundary[1]) + 1.0) < 1e-5  # cos = -1 at the tile edge
    row = local_wall_features(_graphic(), _states([(10.4, 7.0)] * N_UNITS))[0, 0, 25:27]
    assert torch.allclose(row, torch.tensor([np.sin(2 * np.pi * 0.4), np.cos(2 * np.pi * 0.4)], dtype=torch.float32), atol=1e-4)


def test_features_are_continuous_across_a_tile_boundary():
    """The whole point: a unit stepping over a tile edge must not see its features jump.

    The previous 3x3-around-the-rounded-cell patch plus [-0.5, 0.5] offset jumped by a whole tile
    here, and the Run 9 policy chattered between the two cells because of it."""
    rng = np.random.default_rng(3)
    graphic = _graphic()
    graphic[0, WALL] = torch.tensor(rng.random((H, W)) < 0.35, dtype=torch.float32)
    for row, col in [(10.5, 7.0), (4.0, 12.5), (15.5, 15.5), (0.5, 3.0)]:
        for d in (1e-3, 1e-2):
            before = local_wall_features(graphic, _states([(row - d, col - d)] * N_UNITS))[0, 0]
            after = local_wall_features(graphic, _states([(row + d, col + d)] * N_UNITS))[0, 0]
            # bilinear: at most 4 * slope(1 per tile) * 2d per sample; phase: 2*pi*2d
            assert float((after - before).abs().max()) <= 2 * np.pi * 2 * d + 1e-4, (row, col, d)


def test_every_unit_gets_a_patch_including_the_enemy():
    graphic = _graphic()
    graphic[0, WALL, 3, 3] = 1.0
    features = local_wall_features(graphic, _states([(3, 2)] * N_UNITS))
    assert features.shape == (1, N_UNITS, N_PATCH_FEATURES)
    for unit in range(N_UNITS):
        assert abs(float(_patch(features[0, unit])[2, 4]) - 1.0) < 1e-4


# ---------------------------------------------------------------- frame consistency / wiring


def test_derived_channels_commute_with_the_team_mirror():
    rng = np.random.default_rng(1)
    graphic = _graphic()
    graphic[0, STORAGE_ALLY, 5, 5] = 1.0
    graphic[0, STORAGE_ENEMY, 18, 18] = 1.0
    graphic[0, BATTERY, 5, 5] = 0.3
    graphic[0, BATTERY, 12, 4] = 0.5
    states = _states([(int(r), int(c)) for r, c in rng.integers(0, H, size=(N_UNITS, 2))])

    numpy_graphic = graphic[0].permute(1, 2, 0).numpy()
    mirrored_graphic = torch.tensor(mirror_graphic(numpy_graphic)).permute(2, 0, 1).unsqueeze(0)
    mirrored_states = torch.tensor(mirror_agent_states(states[0].numpy())).unsqueeze(0)

    direct = derived_map_channels(mirrored_graphic, mirrored_states)[0]
    via_mirror = torch.tensor(
        mirror_graphic(derived_map_channels(graphic, states)[0].permute(1, 2, 0).numpy())
    ).permute(2, 0, 1)
    # the mirror swaps which team is "ally", so compare the frame-invariant sums
    assert torch.allclose(direct[ALLY] + direct[ENEMY], via_mirror[ALLY] + via_mirror[ENEMY])


def test_model_forward_takes_the_env_channel_count():
    from blackout_env.model.my_model import MyModel

    net = MyModel(hidden_size=128)
    q_values, quantiles, _, _, _ = net(
        torch.zeros(2, 13, H, W), torch.zeros(2, 4), _states([(3, 3)] * N_UNITS).expand(2, -1, -1),
        n_quantiles=4,
    )
    assert q_values.shape == (2, N_UNITS, 8)
    assert quantiles.shape == (2, N_UNITS, 4, 8)
    assert net.graphic_encoder.pre_conv[0].in_channels == 13 + N_DERIVED_MAP_CHANNELS
