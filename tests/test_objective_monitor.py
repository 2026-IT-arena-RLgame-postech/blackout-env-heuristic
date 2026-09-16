"""Scoring-pipeline counters: pickup vs delivery vs loss, and the approach-rate sign."""

from __future__ import annotations

import numpy as np

from blackout_env.train.objective_monitor import (
    BATTERY_CHANNEL,
    STORAGE_ALLY_CHANNEL,
    CARGO_COL,
    ObjectiveMonitor,
    aggregate_objectives,
)

H = W = 24
ROWS = [0, 1, 2, 3, 4]


def _graphic(batteries: list[tuple[int, int]], storages: list[tuple[int, int]]) -> np.ndarray:
    graphic = np.zeros((H, W, 13), dtype=np.float32)
    for row, col in batteries:
        graphic[row, col, BATTERY_CHANNEL] = 0.5
    for row, col in storages:
        graphic[row, col, STORAGE_ALLY_CHANNEL] = 1.0
    return graphic


def _states(cells: list[tuple[int, int]], cargo: list[float]) -> np.ndarray:
    states = np.zeros((10, 12), dtype=np.float32)
    for i, (row, col) in enumerate(cells):
        states[i, 0] = (col + 0.5) * 2.0 / W - 1.0
        states[i, 1] = 1.0 - (row + 0.5) * 2.0 / H
    states[: len(cargo), CARGO_COL] = cargo
    states[:5, 2] = 1.0
    states[5:, 2] = -1.0
    return states


def _run(frames: list[tuple[list[tuple[int, int]], list[float]]], graphic: np.ndarray) -> ObjectiveMonitor:
    monitor = ObjectiveMonitor()
    for cells, cargo in frames:
        monitor.observe(graphic, _states(cells + [(0, 0)] * (10 - len(cells)), cargo), ROWS)
    return monitor


def test_pickup_then_delivery():
    graphic = _graphic(batteries=[(5, 10)], storages=[(5, 5)])
    frames = [
        ([(5, 8)] * 5, [0.0] * 5),   # approaching the battery
        ([(5, 9)] * 5, [0.0] * 5),
        ([(5, 10)] * 5, [0.5] * 5),  # picked up
        ([(5, 6)] * 5, [0.5] * 5),   # carrying toward storage
        ([(5, 5)] * 5, [0.0] * 5),   # delivered on the storage tile
    ]
    counts = _run(frames, graphic).counts
    assert counts.pickups == 5
    assert counts.deliveries == 5
    assert counts.cargo_lost == 0
    assert counts.summary("")["delivery_rate"] == 1.0


def test_cargo_lost_away_from_storage():
    graphic = _graphic(batteries=[(5, 10)], storages=[(20, 20)])
    frames = [
        ([(5, 10)] * 5, [0.5] * 5),
        ([(5, 9)] * 5, [0.5] * 5),
        ([(5, 9)] * 5, [0.0] * 5),  # cargo gone nowhere near storage
    ]
    counts = _run(frames, graphic).counts
    assert counts.deliveries == 0
    assert counts.cargo_lost == 5


def test_approach_rates_are_positive_when_closing_in():
    graphic = _graphic(batteries=[(5, 10)], storages=[(5, 0)])
    approaching = _run([([(5, c)] * 5, [0.0] * 5) for c in (4, 5, 6, 7)], graphic).counts
    retreating = _run([([(5, c)] * 5, [0.0] * 5) for c in (7, 6, 5, 4)], graphic).counts

    assert approaching.summary("")["approach_battery"] > 0
    assert retreating.summary("")["approach_battery"] < 0

    carrying = _run([([(5, c)] * 5, [0.5] * 5) for c in (7, 6, 5, 4)], graphic).counts
    assert carrying.summary("")["approach_storage"] > 0


def test_carry_fraction_and_pooling():
    graphic = _graphic(batteries=[(5, 10)], storages=[(5, 0)])
    counts = _run([([(5, 5)] * 5, cargo) for cargo in ([0.0] * 5, [0.5] * 5, [0.5] * 5, [0.5] * 5)], graphic).counts
    assert counts.unit_ticks == 20
    assert counts.summary("")["carry_frac"] == 0.75

    pooled = aggregate_objectives([counts, counts], "candidate_")
    assert pooled["candidate_carry_frac"] == 0.75
    assert pooled["candidate_pickups_per_1000_ticks"] == counts.summary("")["pickups_per_1000_ticks"]


def test_no_batteries_on_the_map_is_not_an_error():
    graphic = _graphic(batteries=[], storages=[(5, 0)])
    counts = _run([([(5, 5)] * 5, [0.0] * 5)] * 3, graphic).counts
    assert counts.summary("")["approach_battery"] == 0.0
    assert counts.unit_ticks == 15
