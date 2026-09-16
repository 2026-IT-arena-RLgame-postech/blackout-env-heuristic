"""
Why is a unit standing still? Three separable reasons, each pointing at a different fix.

Watching Run 7's 80k checkpoint in the GUI turned "the units stall" into three distinct
behaviours, and they do not share a cause:

  idle_on_storage        empty-handed, parked on a storage tile. Nothing to deposit and nothing
                         being fetched -- the unit is sitting on the one landmark it has learned.
  cargo_wait_full        carrying a battery, at a storage that cannot accept it. Unity rejects an
                         over-capacity deposit whole and silently (Storage.cs: "storage is full;
                         nop.") -- no event, no reward, no observable change -- so nothing teaches
                         the policy to go to the next storage. The heuristics compute the same
                         capacity check explicitly (StrategicHeuristic._storage_target) and walk
                         away; this counts how often the model does not.
  corner_block           a failed move into a wall where a neighbouring compass direction was
                         free -- the unit is snagged on a convex corner that a small turn would
                         clear. Distinct from being genuinely boxed in, and the signature of not
                         resolving its own position finely enough against the wall geometry.

Capacity and spawn-storage logic mirror what the heuristic policies already do from the same
observation, so "the model could have known" is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackout_env.heuristics.strategic import StrategicHeuristic

VOID, WALL = 0, 1
SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY, BATTERY = 4, 5, 6, 7, 8
FIRST_SPECIAL = 9
CARGO_COL = 4
MAX_ITEM_AMOUNT = 10  # Unity ItemData.MaxItemAmount; the same constant the heuristics assume
BATTERY_SCALE = 15.0  # channel 8 stores amount / 15

BLOCKED_ACTION_NORM = 0.35  # matches MovementMonitor's "commanded hard" threshold
BLOCKED_MOVEMENT = 2e-4
TRANSITION_MOVEMENT = 0.15  # a respawn teleport, not a failed step

# (drow, dcol) for the 8 compass directions, in DIRECTION_VECTORS order.
_DIRECTIONS = np.array(
    [[-int(np.sign(round(np.sin(i * np.pi / 4), 3))), int(np.sign(round(np.cos(i * np.pi / 4), 3)))]
     for i in range(8)],
    dtype=np.int64,
)


@dataclass
class StallCounts:
    unit_ticks: int = 0
    idle_on_storage: int = 0
    cargo_wait_full: int = 0
    cargo_wait_full_spawn: int = 0
    cargo_wait_with_room_elsewhere: int = 0
    blocked: int = 0
    corner_block: int = 0

    def summary(self, prefix: str = "") -> dict[str, float]:
        ticks = max(1, self.unit_ticks)
        return {
            f"{prefix}idle_on_storage_per_1000": 1000.0 * self.idle_on_storage / ticks,
            f"{prefix}cargo_wait_full_per_1000": 1000.0 * self.cargo_wait_full / ticks,
            f"{prefix}cargo_wait_full_spawn_frac": self.cargo_wait_full_spawn / max(1, self.cargo_wait_full),
            # The damning subset: it was waiting at a full storage while another one had room.
            f"{prefix}cargo_wait_avoidable_frac": self.cargo_wait_with_room_elsewhere / max(1, self.cargo_wait_full),
            f"{prefix}blocked_per_1000": 1000.0 * self.blocked / ticks,
            f"{prefix}corner_block_frac": self.corner_block / max(1, self.blocked),
        }


def _component_capacity(graphic: np.ndarray, component: np.ndarray) -> int:
    """Free battery slots in one storage component, by Unity's per-tile rule."""
    special = np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1)
    capacity = 0
    for row, col in component:
        if special[row, col]:
            continue  # a special parked on the tile blocks it for batteries entirely
        amount = int(round(float(graphic[row, col, BATTERY]) * BATTERY_SCALE))
        capacity += max(0, MAX_ITEM_AMOUNT - amount) if amount > 0 else MAX_ITEM_AMOUNT
    return capacity


class StallMonitor:
    """One side's stall taxonomy across a match. Call observe() once per env step."""

    def __init__(self) -> None:
        self.counts = StallCounts()
        self._components: list[np.ndarray] | None = None
        self._spawn_mask: np.ndarray | None = None

    def _storage_components(self, graphic: np.ndarray) -> list[np.ndarray]:
        # Storage geometry is fixed for the episode; only tile contents change.
        if self._components is None:
            mask = graphic[..., STORAGE_ALLY] > 0.5
            self._components = [np.array(c, dtype=np.int64) for c in StrategicHeuristic._components(mask)]
            self._spawn_mask = StrategicHeuristic._protected_storage_mask(graphic, STORAGE_ALLY, SPAWN_ALLY)
        return self._components

    def observe(
        self,
        graphic: np.ndarray,
        before: np.ndarray,
        after: np.ndarray,
        actions: dict[str, np.ndarray],
        rows: dict[str, int],
    ) -> None:
        """
        graphic/before: this side's view before the step; after: agent_states after it.
        actions/rows: the commanded (dx, dy) and physical unit index, keyed by agent name.
        """
        height, width = graphic.shape[:2]
        walkable = graphic[..., WALL] < 0.5
        storage_any = (graphic[..., STORAGE_ALLY] > 0.5) | (graphic[..., STORAGE_ENEMY] > 0.5)
        components = self._storage_components(graphic)
        capacities = [_component_capacity(graphic, c) for c in components]

        for agent, action in actions.items():
            row = rows[agent]
            cell = StrategicHeuristic._to_pixel(before[row, :2], (height, width))
            movement = float(np.linalg.norm(after[row, :2] - before[row, :2]))
            transition = movement > TRANSITION_MOVEMENT or not np.array_equal(before[row, 3:], after[row, 3:])
            still = movement <= BLOCKED_MOVEMENT and not transition
            carrying = float(before[row, CARGO_COL]) > 1e-3
            self.counts.unit_ticks += 1

            if still and not carrying and storage_any[cell]:
                self.counts.idle_on_storage += 1

            if carrying and components:
                held = max(1, int(round(float(before[row, CARGO_COL]) * BATTERY_SCALE)))
                nearest = int(np.argmin([
                    ((component - np.array(cell)) ** 2).sum(axis=1).min() for component in components
                ]))
                touching = ((components[nearest] - np.array(cell)) ** 2).sum(axis=1).min() <= 2
                if touching and capacities[nearest] < held:
                    self.counts.cargo_wait_full += 1
                    if self._spawn_mask is not None and self._spawn_mask[tuple(components[nearest][0])]:
                        self.counts.cargo_wait_full_spawn += 1
                    if any(cap >= held for i, cap in enumerate(capacities) if i != nearest):
                        self.counts.cargo_wait_with_room_elsewhere += 1

            if still and float(np.linalg.norm(action)) >= BLOCKED_ACTION_NORM:
                self.counts.blocked += 1
                commanded = int(np.argmax(np.stack(_unit_directions()) @ np.asarray(action, dtype=np.float64)))
                # Escapable corner: the commanded step is into a wall, but a direction within a
                # quarter turn either way is open -- sidestepping clears the snag. The 45-degree
                # neighbours alone are not enough to test: with a wall straight ahead, both
                # diagonals are refused by the no-corner-cutting rule, and the escape is the
                # 90-degree step.
                neighbours = [(commanded + offset) % 8 for offset in (-2, -1, 1, 2)]
                if not _steps_free(walkable, cell, commanded) and any(
                    _steps_free(walkable, cell, n) for n in neighbours
                ):
                    self.counts.corner_block += 1


def _unit_directions() -> list[np.ndarray]:
    return [np.array([np.cos(i * np.pi / 4), np.sin(i * np.pi / 4)]) for i in range(8)]


def _steps_free(walkable: np.ndarray, cell: tuple[int, int], direction: int) -> bool:
    drow, dcol = _DIRECTIONS[direction]
    row, col = cell[0] + drow, cell[1] + dcol
    height, width = walkable.shape
    if not (0 <= row < height and 0 <= col < width) or not walkable[row, col]:
        return False
    if drow != 0 and dcol != 0:  # no corner cutting, same rule the heuristics navigate by
        return bool(walkable[cell[0] + drow, cell[1]] and walkable[cell[0], cell[1] + dcol])
    return True


def aggregate_stalls(counts: list[StallCounts], prefix: str = "") -> dict[str, float]:
    merged = StallCounts()
    for c in counts:
        merged.unit_ticks += c.unit_ticks
        merged.idle_on_storage += c.idle_on_storage
        merged.cargo_wait_full += c.cargo_wait_full
        merged.cargo_wait_full_spawn += c.cargo_wait_full_spawn
        merged.cargo_wait_with_room_elsewhere += c.cargo_wait_with_room_elsewhere
        merged.blocked += c.blocked
        merged.corner_block += c.corner_block
    return merged.summary(prefix)
