"""Stall taxonomy: idling on storage, waiting at a full storage, snagging on a corner."""

from __future__ import annotations

import numpy as np

from blackout_env.model.my_policy import DIRECTION_VECTORS
from blackout_env.train.stall_monitor import (
    BATTERY,
    CARGO_COL,
    SPAWN_ALLY,
    STORAGE_ALLY,
    STORAGE_ENEMY,
    WALL,
    StallMonitor,
    aggregate_stalls,
)

H = W = 24
AGENT = {"unit_0": 0}


def _graphic() -> np.ndarray:
    return np.zeros((H, W, 13), dtype=np.float32)


def _at(cell: tuple[int, int]) -> np.ndarray:
    states = np.zeros((10, 12), dtype=np.float32)
    states[:, 0] = (cell[1] + 0.5) * 2.0 / W - 1.0
    states[:, 1] = 1.0 - (cell[0] + 0.5) * 2.0 / H
    states[:5, 2] = 1.0
    states[5:, 2] = -1.0
    return states


def _observe(graphic, cell, *, cargo=0.0, action=(0.0, 0.0), moved=False):
    monitor = StallMonitor()
    before = _at(cell)
    before[0, CARGO_COL] = cargo
    after = before.copy()
    if moved:
        after[0, 0] += 0.05
    monitor.observe(graphic, before, after, {"unit_0": np.array(action, dtype=np.float32)}, AGENT)
    return monitor.counts


def test_idling_empty_handed_on_a_storage_tile():
    graphic = _graphic()
    graphic[5, 5, STORAGE_ALLY] = 1.0
    assert _observe(graphic, (5, 5)).idle_on_storage == 1
    assert _observe(graphic, (5, 6)).idle_on_storage == 0           # off the storage
    assert _observe(graphic, (5, 5), moved=True).idle_on_storage == 0  # walking across it
    assert _observe(graphic, (5, 5), cargo=0.2).idle_on_storage == 0   # carrying: a deposit attempt

    enemy = _graphic()
    enemy[5, 5, STORAGE_ENEMY] = 1.0
    assert _observe(enemy, (5, 5)).idle_on_storage == 1  # loitering on either team's storage counts


def test_waiting_at_a_storage_that_cannot_take_the_cargo():
    graphic = _graphic()
    for col in (5, 6):
        graphic[5, col, STORAGE_ALLY] = 1.0
        graphic[5, col, BATTERY] = 10 / 15  # both tiles maxed out
    graphic[5, 5, SPAWN_ALLY] = 0.0

    counts = _observe(graphic, (5, 4), cargo=3 / 15)
    assert counts.cargo_wait_full == 1
    assert counts.summary()["cargo_wait_avoidable_frac"] == 0.0  # nowhere else to go

    graphic[5, 6, BATTERY] = 0.0  # same component now has room
    assert _observe(graphic, (5, 4), cargo=3 / 15).cargo_wait_full == 0


def test_a_second_storage_with_room_makes_the_wait_avoidable():
    graphic = _graphic()
    graphic[5, 5, STORAGE_ALLY] = 1.0
    graphic[5, 5, BATTERY] = 10 / 15          # full, and the one the unit is standing at
    graphic[18, 18, STORAGE_ALLY] = 1.0       # a separate component, empty
    counts = _observe(graphic, (5, 4), cargo=4 / 15)
    assert counts.cargo_wait_full == 1
    assert counts.summary()["cargo_wait_avoidable_frac"] == 1.0


def test_corner_block_counts_only_escapable_snags():
    graphic = _graphic()
    graphic[10, 11, WALL] = 1.0  # wall directly east
    east = DIRECTION_VECTORS[0]
    counts = _observe(graphic, (10, 10), action=east)
    assert counts.blocked == 1
    assert counts.corner_block == 1  # north/south are open: sidestepping clears it

    boxed = _graphic()
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if (drow, dcol) != (0, 0):
                boxed[10 + drow, 10 + dcol, WALL] = 1.0
    counts = _observe(boxed, (10, 10), action=east)
    assert counts.blocked == 1
    assert counts.corner_block == 0  # genuinely boxed in, not a corner snag


def test_a_move_that_succeeds_is_not_blocked():
    graphic = _graphic()
    graphic[10, 11, WALL] = 1.0
    counts = _observe(graphic, (10, 10), action=DIRECTION_VECTORS[0], moved=True)
    assert counts.blocked == 0


def test_aggregate_pools_before_dividing():
    graphic = _graphic()
    graphic[5, 5, STORAGE_ALLY] = 1.0
    counts = [_observe(graphic, (5, 5)) for _ in range(3)]
    pooled = aggregate_stalls(counts, "candidate_")
    assert pooled["candidate_idle_on_storage_per_1000"] == 1000.0
