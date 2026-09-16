"""
Per-unit scoring-pipeline diagnostics: approach -> pickup -> carry -> deliver.

Run 6 ended 0-30 against V4 while every aggregate metric (margin, match length, blocked rate)
moved only a little, and it took an offline breakdown to find where the pipeline actually
snapped: the model picked up 113 batteries to the heuristic's 213, but once carrying it headed
to storage at 75% of the heuristic's approach rate and delivered 74% of what it picked up. The
bottleneck was the step before -- with no cargo, its distance to the nearest loose battery
barely changed (-0.003 cells/tick against the heuristic's -0.013), i.e. it wandered instead of
going to get one. See docs/run6_diagnosis_20260916.md §4.

That is the number worth watching during a run, and it is cheap enough to compute inline, so
both the periodic eval and the on-policy collection track it here -- for the opponent too, which
makes the heuristic a live baseline in the same match rather than a number from another session.

Definitions
-----------
approach_battery  mean per-tick *decrease* in grid distance to the nearest FETCHABLE battery --
                  loose on the floor or sitting in the enemy's storage, but not one already
                  banked in own storage -- over ticks where the unit carries nothing
                  (positive = closing in)
approach_storage  same toward the nearest own storage tile, over ticks where it carries a battery
pickups           no-cargo -> cargo transitions
deliveries        cargo -> no-cargo within 1.5 cells of own storage
cargo_lost        cargo -> no-cargo anywhere else (killed while carrying, mostly)
carry_frac        fraction of unit-ticks spent holding a battery

Distances are straight-line in grid cells, not path lengths: this is a direction-of-travel
signal, and a Dijkstra per unit per tick would cost far more than it adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BATTERY_CHANNEL = 8
STORAGE_ALLY_CHANNEL = 6
STORAGE_ENEMY_CHANNEL = 7
CARGO_COL = 4  # agent_states: [pos_x, pos_y, team, item_none, item_battery, ...]
DELIVERY_RADIUS = 1.5  # cells; a unit standing on/next to its storage when the cargo disappears


def _cells(positions: np.ndarray, height: int, width: int) -> np.ndarray:
    """[N, 2] (row, col) grid cells for [N, 2] normalized (x, y) positions."""
    row = np.clip(np.round((1.0 - positions[:, 1]) * 0.5 * height - 0.5), 0, height - 1)
    col = np.clip(np.round((positions[:, 0] + 1.0) * 0.5 * width - 0.5), 0, width - 1)
    return np.stack([row, col], axis=1)


def _nearest(target_cells: np.ndarray, unit_cells: np.ndarray) -> np.ndarray:
    """[N] distance from each unit to the closest target cell (nan when there are none)."""
    if len(target_cells) == 0:
        return np.full(len(unit_cells), np.nan)
    deltas = unit_cells[:, None, :] - target_cells[None, :, :]
    return np.sqrt((deltas**2).sum(axis=-1)).min(axis=1)


@dataclass
class ObjectiveCounts:
    unit_ticks: int = 0
    carry_ticks: int = 0
    pickups: int = 0
    deliveries: int = 0
    cargo_lost: int = 0
    battery_steps: list[float] = field(default_factory=list)
    storage_steps: list[float] = field(default_factory=list)

    def summary(self, prefix: str) -> dict[str, float]:
        ticks = max(1, self.unit_ticks)
        return {
            f"{prefix}carry_frac": self.carry_ticks / ticks,
            f"{prefix}pickups_per_1000_ticks": 1000.0 * self.pickups / ticks,
            f"{prefix}deliveries_per_1000_ticks": 1000.0 * self.deliveries / ticks,
            f"{prefix}cargo_lost_per_1000_ticks": 1000.0 * self.cargo_lost / ticks,
            f"{prefix}delivery_rate": self.deliveries / max(1, self.pickups),
            f"{prefix}approach_battery": float(np.mean(self.battery_steps)) if self.battery_steps else 0.0,
            f"{prefix}approach_storage": float(np.mean(self.storage_steps)) if self.storage_steps else 0.0,
        }


class ObjectiveMonitor:
    """Tracks one side's scoring pipeline across a match. Call observe() once per env step."""

    def __init__(self) -> None:
        self.counts = ObjectiveCounts()
        self._previous: dict[int, tuple[float, float, float]] = {}  # row -> (cargo, d_battery, d_storage)

    def observe(self, graphic: np.ndarray, agent_states: np.ndarray, rows: list[int]) -> None:
        """
        graphic/agent_states: this side's view of the state BEFORE the step (the same pair
        MovementMonitor is given), `rows` the physical unit indices this side controls.
        """
        height, width = graphic.shape[:2]
        unit_cells = _cells(agent_states[rows, :2], height, width)
        # A battery already banked in OWN storage is not something to go and fetch -- the
        # heuristics skip exactly those tiles (advanced.py: `if graphic[y, x, STORAGE_ALLY] >
        # 0.5: continue`) while still treating the enemy's stored batteries as steal targets.
        # Counting them made "distance to the nearest battery" small for a unit loitering on its
        # own storage, which is the behaviour this metric exists to catch.
        fetchable = (graphic[..., BATTERY_CHANNEL] > 1e-3) & (graphic[..., STORAGE_ALLY_CHANNEL] <= 0.5)
        battery_cells = np.argwhere(fetchable).astype(np.float64)
        storage_cells = np.argwhere(graphic[..., STORAGE_ALLY_CHANNEL] > 0.5).astype(np.float64)
        d_battery = _nearest(battery_cells, unit_cells)
        d_storage = _nearest(storage_cells, unit_cells)
        cargo = agent_states[rows, CARGO_COL]

        for i, row in enumerate(rows):
            self.counts.unit_ticks += 1
            carrying = cargo[i] > 1e-3
            self.counts.carry_ticks += int(carrying)

            previous = self._previous.get(row)
            if previous is not None:
                was_carrying, prev_battery, prev_storage = previous
                if not was_carrying and carrying:
                    self.counts.pickups += 1
                elif was_carrying and not carrying:
                    # `prev_storage` is the distance at the last tick it still held the cargo
                    if prev_storage <= DELIVERY_RADIUS:
                        self.counts.deliveries += 1
                    else:
                        self.counts.cargo_lost += 1
                elif not carrying and np.isfinite(prev_battery) and np.isfinite(d_battery[i]):
                    self.counts.battery_steps.append(prev_battery - d_battery[i])
                elif carrying and np.isfinite(prev_storage) and np.isfinite(d_storage[i]):
                    self.counts.storage_steps.append(prev_storage - d_storage[i])

            self._previous[row] = (carrying, d_battery[i], d_storage[i])


def aggregate_objectives(monitors: list[ObjectiveCounts], prefix: str = "") -> dict[str, float]:
    """Pools several matches' counts before summarizing, so rates aren't averages of averages."""
    merged = ObjectiveCounts()
    for counts in monitors:
        merged.unit_ticks += counts.unit_ticks
        merged.carry_ticks += counts.carry_ticks
        merged.pickups += counts.pickups
        merged.deliveries += counts.deliveries
        merged.cargo_lost += counts.cargo_lost
        merged.battery_steps.extend(counts.battery_steps)
        merged.storage_steps.extend(counts.storage_steps)
    return merged.summary(prefix)
