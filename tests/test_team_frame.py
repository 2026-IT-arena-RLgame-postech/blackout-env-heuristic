"""Canonical-frame helpers: involution, position/grid agreement, and the map symmetry they assume."""

from __future__ import annotations

import numpy as np
import pytest

from blackout_env.env.team_frame import (
    N_TEAM,
    canonical_unit_row,
    is_team_b_view,
    mirror_action_idx,
    mirror_agent_states,
    mirror_direction_idx,
    mirror_graphic,
)
from blackout_env.heuristics.strategic import StrategicHeuristic
from blackout_env.model.my_policy import DIRECTION_VECTORS

H = W = 24
WALL, SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY = 1, 4, 5, 6, 7


def _cell(position: np.ndarray) -> tuple[int, int]:
    return StrategicHeuristic._to_pixel(position, (H, W))


def _random_graphic(rng: np.random.Generator) -> np.ndarray:
    return rng.random((H, W, 13)).astype(np.float32)


def _random_agent_states(rng: np.random.Generator) -> np.ndarray:
    states = rng.uniform(-1.0, 1.0, size=(10, 12)).astype(np.float32)
    states[:5, 2] = 1.0
    states[5:, 2] = -1.0
    return states


def test_mirror_is_an_involution():
    rng = np.random.default_rng(0)
    graphic, states = _random_graphic(rng), _random_agent_states(rng)
    actions = rng.integers(0, 8, size=(10,))

    assert np.array_equal(mirror_graphic(mirror_graphic(graphic)), graphic)
    assert np.allclose(mirror_agent_states(mirror_agent_states(states)), states)
    assert np.array_equal(mirror_action_idx(mirror_action_idx(actions)), actions)


def test_mirror_graphic_moves_a_cell_to_the_anti_diagonal_image():
    graphic = np.zeros((H, W, 13), dtype=np.float32)
    graphic[3, 20, SPAWN_ALLY] = 1.0
    mirrored = mirror_graphic(graphic)
    assert mirrored[W - 1 - 20, H - 1 - 3, SPAWN_ALLY] == 1.0
    assert mirrored[..., SPAWN_ALLY].sum() == 1.0


def test_unit_position_and_grid_stay_aligned():
    """A unit standing on a marked cell must still stand on it after mirroring both."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        states = _random_agent_states(rng)
        row, col = _cell(states[7, :2])
        graphic = np.zeros((H, W, 13), dtype=np.float32)
        graphic[row, col, STORAGE_ALLY] = 1.0

        m_graphic, m_states = mirror_graphic(graphic), mirror_agent_states(states)
        m_row, m_col = _cell(m_states[canonical_unit_row(7, team_b=True), :2])
        assert m_graphic[m_row, m_col, STORAGE_ALLY] == 1.0


def test_action_mirror_matches_the_position_mirror():
    """Stepping then mirroring == mirroring then stepping (up to the grid's discretization)."""
    rng = np.random.default_rng(2)
    for idx in range(8):
        for _ in range(20):
            pos = rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
            step = DIRECTION_VECTORS[idx] * 0.1
            moved_then_mirrored = (pos + step)[::-1]
            mirrored_step = DIRECTION_VECTORS[mirror_direction_idx(idx)] * 0.1
            mirrored_then_moved = pos[::-1] + mirrored_step
            assert np.allclose(moved_then_mirrored, mirrored_then_moved, atol=1e-6)


def test_own_team_moves_to_rows_0_to_4():
    rng = np.random.default_rng(3)
    states = _random_agent_states(rng)  # team-B view: rows 5-9 are "own" (sign flipped)
    states[:, 2] *= -1.0
    assert is_team_b_view(states)

    mirrored = mirror_agent_states(states)
    assert not is_team_b_view(mirrored)
    assert np.all(mirrored[:N_TEAM, 2] > 0)
    assert np.allclose(mirrored[0, 2:], states[N_TEAM, 2:])


def test_action_table_follows_the_same_row_roll():
    actions = np.arange(10) % 8
    mirrored = mirror_action_idx(actions)
    for unit in range(10):
        assert mirrored[canonical_unit_row(unit, team_b=True)] == mirror_direction_idx(actions[unit])


def test_mirror_batch_keeps_every_field_in_one_frame():
    """The team-B training batch must come out with obs, next-obs and actions all mirrored."""
    from blackout_env.train.qmix_trainer import QMIXTrainer

    rng = np.random.default_rng(4)
    b, k = 3, 2
    batch = {
        "graphic": rng.random((b, H, W, 13)).astype(np.float32),
        "boot_graphic": rng.random((b, H, W, 13)).astype(np.float32),
        "future_graphic": rng.random((b, k, H, W, 13)).astype(np.float32),
        "agent_states": np.stack([_random_agent_states(rng) for _ in range(b)]),
        "boot_agent_states": np.stack([_random_agent_states(rng) for _ in range(b)]),
        "future_agent_states": np.stack([[_random_agent_states(rng) for _ in range(k)] for _ in range(b)]),
        "actions": rng.integers(0, 8, size=(b, 10)),
        "action_window": rng.integers(0, 8, size=(b, k, 10)),
        "is_weights": rng.random(b),  # untouched field, must survive
    }
    mirrored = QMIXTrainer._mirror_batch(batch)

    assert np.array_equal(mirrored["is_weights"], batch["is_weights"])
    for unit in range(10):
        row = canonical_unit_row(unit, team_b=True)
        assert mirrored["actions"][0, row] == mirror_direction_idx(batch["actions"][0, unit])
        assert mirrored["action_window"][0, 1, row] == mirror_direction_idx(batch["action_window"][0, 1, unit])
        assert np.allclose(mirrored["agent_states"][0, row, :2], batch["agent_states"][0, unit, :2][::-1])
    assert np.array_equal(mirrored["future_graphic"][0, 1], mirror_graphic(batch["future_graphic"][0, 1]))
    assert np.array_equal(mirrored["boot_graphic"][2], mirror_graphic(batch["boot_graphic"][2]))


@pytest.mark.parametrize("row_index", [0, 123_456, 654_321])
def test_real_map_is_symmetric_under_the_mirror(row_index):
    """The whole canonicalization rests on the map being the anti-transpose of itself."""
    dataset = pytest.importorskip("pathlib").Path("datasets/heuristic_mixv3_20260915/buffer_a.npz")
    if not dataset.exists():
        pytest.skip("heuristic dataset not present")

    import ast
    import zipfile

    with zipfile.ZipFile(dataset) as archive, archive.open("graphic.npy") as handle:
        header = handle.read(10)
        header_len = int.from_bytes(header[8:10], "little")
        meta = ast.literal_eval(handle.read(header_len).decode().strip())
        dtype, shape = np.dtype(meta["descr"]), meta["shape"]
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        handle.seek(10 + header_len + row_index * row_bytes)
        graphic = np.frombuffer(handle.read(row_bytes), dtype=dtype).reshape(shape[1:])

    mirrored = mirror_graphic(graphic)
    assert np.array_equal(mirrored[..., WALL] > 0.5, graphic[..., WALL] > 0.5)
    assert np.array_equal(mirrored[..., SPAWN_ALLY] > 0.5, graphic[..., SPAWN_ENEMY] > 0.5)
    assert np.array_equal(mirrored[..., STORAGE_ALLY] > 0.5, graphic[..., STORAGE_ENEMY] > 0.5)
