"""Second-generation team-level BlackOut heuristic.

V2 intentionally inherits all navigation and recovery behaviour from V1.  Its isolated
change is coordinated, path-aware economic task assignment, which makes V1/V2 comparisons
and later policy-mixture sampling meaningful.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict

import numpy as np

from . import _native
from .strategic import (
    BATTERY,
    CARRIER,
    COLLECTOR,
    FIRST_SPECIAL,
    HUNTER,
    STORAGE_ALLY,
    STORAGE_ENEMY,
    WALL,
    StrategicHeuristic,
)
from ..env.constants import unit_index


class StrategicHeuristicV2(StrategicHeuristic):
    """V1 plus global path-aware assignment and absorption-feasible stealing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distance_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def reset(self) -> None:
        super().reset()
        self._distance_cache.clear()

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not obs:
            return {}
        sample = next(iter(obs.values()))
        graphic = sample["graphic"]
        states = sample["agent_states"]
        team_state = sample["team_state"]

        time_left = float(team_state[2])
        if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
            self.reset()
        self._last_time_left = time_left
        self._tick += 1

        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}
        local_order = {name: j for j, name in enumerate(controlled)}
        walkable = graphic[..., WALL] < 0.5

        economic_names = []
        for name in controlled:
            state = states[row_for[name]]
            role = self._role(local_order[name])
            cls = self._class_id(state)
            if self._is_holding(state) or cls == HUNTER:
                continue
            if cls == COLLECTOR and role in {"carrier", "hunter"}:
                continue  # transformation remains V1's stable specialist behaviour
            economic_names.append(name)

        assignments = self._assign_economic_tasks(
            economic_names, row_for, states, graphic, team_state, walkable
        )

        actions: dict[str, np.ndarray] = {}
        reservations: set[tuple[int, int]] = set()
        for name in controlled:
            state = states[row_for[name]]
            role = self._role(local_order[name])
            if name in assignments:
                target, kind = assignments[name]
            else:
                target, kind = self._choose_target(
                    role, state, states, graphic, reservations, local_order[name], team_state
                )
            if target is not None and kind in {"battery", "special", "steal"}:
                reservations.add(target)
            actions[name] = self._navigate(
                name, state, states, target, kind, walkable, graphic.shape[:2]
            )
        return actions

    def _assign_economic_tasks(
        self,
        names: list[str],
        row_for: dict[str, int],
        states: np.ndarray,
        graphic: np.ndarray,
        team_state: np.ndarray,
        walkable: np.ndarray,
    ) -> dict[str, tuple[tuple[int, int], str]]:
        if not names:
            return {}

        protected_enemy = self._cached_protected_enemy_storage_mask(graphic)
        tasks: list[tuple[tuple[int, int], str, float]] = []
        for y, x, amount in self._battery_pixels(graphic):
            if graphic[y, x, STORAGE_ALLY] > 0.5 or protected_enemy[y, x]:
                continue
            kind = "steal" if graphic[y, x, STORAGE_ENEMY] > 0.5 else "battery"
            tasks.append(((y, x), kind, amount))

        # Unlike V1, specials in the protected enemy store are filtered too.
        special_value = {9: 11.0, 10: 10.0, 11: 4.5, 12: 4.5}
        if float(team_state[0]) < 0.92:
            for channel, y, x in self._special_pixels(graphic):
                if graphic[y, x, STORAGE_ALLY] > 0.5 or protected_enemy[y, x]:
                    continue
                tasks.append(((y, x), "special", special_value.get(channel, 3.0)))
        if not tasks:
            return {}

        enemy_states = states[states[:, 2] < 0]
        enemy_pixels = [
            (self._to_pixel(s[:2], graphic.shape[:2]), self._class_id(s))
            for s in enemy_states
        ]
        absorption_seconds = max(0.0, float(team_state[3]) * 20.0)
        gap_points = max(0.0, (1.0 - float(team_state[0])) * 100.0)
        pairs: list[tuple[float, str, int]] = []

        for name in names:
            state = states[row_for[name]]
            start = self._to_pixel(state[:2], graphic.shape[:2])
            distances = self._cached_distance_map(walkable, start)
            cls = self._class_id(state)
            speed = 6.0 if cls == CARRIER else 4.0
            old_target = self._memory.get(name).target if name in self._memory else None
            for task_idx, (target, kind, intrinsic) in enumerate(tasks):
                distance = float(distances[target])
                if not np.isfinite(distance):
                    continue
                if kind == "steal" and distance / speed + 0.35 >= absorption_seconds:
                    # It will be absorbed before arrival, so pursuing it cannot deny or collect.
                    continue
                value = intrinsic * (2.5 if kind != "special" else 1.0)
                if kind == "steal":
                    urgency = 1.0 - float(team_state[3])
                    value += intrinsic * (1.0 + urgency)
                if kind == "battery" and intrinsic + 1e-4 >= gap_points:
                    value += 28.0  # a single successful deposit can end the game immediately
                value -= 0.32 * distance

                dangerous_classes = {HUNTER} if cls != CARRIER else {COLLECTOR, HUNTER, CARRIER}
                danger_distance = min(
                    (math.dist(target, p) for p, enemy_cls in enemy_pixels if enemy_cls in dangerous_classes),
                    default=99.0,
                )
                if danger_distance < 5.0:
                    value -= (5.0 - danger_distance) * (1.5 if cls == CARRIER else 0.8)
                    value -= intrinsic * (0.35 if cls == CARRIER else 0.15)
                if old_target == target:
                    value += 1.5  # assignment hysteresis produces cleaner, less twitchy demos
                pairs.append((value, name, task_idx))

        # Global greedy matching considers every unit-task pair together; unit ordering no
        # longer gives the first Collector first refusal on every high-value battery.
        pairs.sort(reverse=True, key=lambda p: p[0])
        assigned_names: set[str] = set()
        assigned_tasks: set[int] = set()
        result: dict[str, tuple[tuple[int, int], str]] = {}
        for _, name, task_idx in pairs:
            if name in assigned_names or task_idx in assigned_tasks:
                continue
            target, kind, _ = tasks[task_idx]
            result[name] = (target, kind)
            assigned_names.add(name)
            assigned_tasks.add(task_idx)
            if len(assigned_names) == len(names):
                break
        return result

    def _cached_distance_map(
        self, walkable: np.ndarray, start: tuple[int, int]
    ) -> np.ndarray:
        """Reuse maps for pixel positions revisited within one static-map episode."""
        cached = self._distance_cache.get(start)
        if cached is None:
            cached = self._distance_map(walkable, start)
            self._distance_cache[start] = cached
            bytes_per_map = max(1, walkable.size * np.dtype(np.float32).itemsize)
            max_entries = max(32, min(walkable.size, (16 * 1024 * 1024) // bytes_per_map))
            if len(self._distance_cache) > max_entries:
                self._distance_cache.popitem(last=False)
        else:
            self._distance_cache.move_to_end(start)
        return cached

    @staticmethod
    def _distance_map(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        """All-cell 8-neighbour shortest path lengths from one unit position."""
        if _native.NUMBA_AVAILABLE:
            return _native.distance_map_numba(walkable, start[0], start[1])
        h, w = walkable.shape
        distances = np.full((h, w), np.inf, dtype=np.float32)
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            return distances
        distances[start] = 0.0
        queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
        neighbours = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                      (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))
        while queue:
            cost, y, x = heapq.heappop(queue)
            # Heap costs are Python float while the dense map is float32. Exact equality
            # discarded almost every node reached through a diagonal (e.g. 1.4142 after
            # float32 rounding), truncating the search to a tiny diamond around the start.
            if cost > float(distances[y, x]) + 1e-5:
                continue
            for dy, dx, step in neighbours:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or not walkable[ny, nx]:
                    continue
                if dy and dx and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                new_cost = cost + step
                if new_cost + 1e-6 >= float(distances[ny, nx]):
                    continue
                distances[ny, nx] = new_cost
                heapq.heappush(queue, (new_cost, ny, nx))
        return distances
