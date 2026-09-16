"""Sixth-generation heuristic: threat-aware A* for fragile cargo movement."""

from __future__ import annotations

import heapq
import math

import numpy as np

from . import _native
from .spread_deposit import StrategicHeuristicV4
from .strategic import CARRIER, HUNTER


class StrategicHeuristicV6(StrategicHeuristicV4):
    """V4 with enemy risk integrated into cargo path planning, not raw steering."""

    def __init__(self, *, risk_radius_tiles: float = 3.0, risk_weight: float = 3.2, **kwargs):
        super().__init__(**kwargs)
        self.risk_radius_tiles = max(0.5, float(risk_radius_tiles))
        self.risk_weight = max(0.0, float(risk_weight))
        self._active_risk: np.ndarray | None = None
        # One team decision tick sees the same enemy states for every unit, so the risk field
        # only depends on whether the planning unit is a Carrier (and on the radius knob).
        self._risk_cache: dict[tuple[int, bool, float, tuple[int, int]], np.ndarray] = {}
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._hypot_tables: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        super().reset()
        self._risk_cache.clear()

    def _risk_field(self, states, unit_class, shape) -> np.ndarray:
        key = (self._tick, unit_class == CARRIER, self.risk_radius_tiles, shape)
        risk = self._risk_cache.get(key)
        if risk is not None:
            return risk
        if len(self._risk_cache) > 8:
            self._risk_cache.clear()
        grid = self._grid_cache.get(shape)
        if grid is None:
            grid = self._grid_cache[shape] = np.indices(shape, dtype=np.float32)
        yy, xx = grid
        risk = np.zeros(shape, dtype=np.float32)
        for enemy in states[states[:, 2] < 0]:
            enemy_class = self._class_id(enemy)
            dangerous = enemy_class == HUNTER or unit_class == CARRIER
            if not dangerous:
                continue
            ey, ex = self._to_pixel_float(enemy[:2], shape)
            distance = np.sqrt((yy - ey) ** 2 + (xx - ex) ** 2)
            risk += np.exp(-distance / self.risk_radius_tiles).astype(np.float32)
        self._risk_cache[key] = risk
        return risk

    def _navigate(self, name, state, states, target, kind, walkable, shape):
        unit_class = self._class_id(state)
        vulnerable = self._is_holding(state) or unit_class == CARRIER
        if not vulnerable:
            return super()._navigate(name, state, states, target, kind, walkable, shape)

        risk = self._risk_field(states, unit_class, shape)

        # Base V1 adds a continuous repulsion after A*.  Hide enemies from that final stage so
        # V6 follows the same risk model during planning and execution instead of being pushed
        # off its selected corridor into a wall or cul-de-sac.
        planning_states = states.copy()
        planning_states[planning_states[:, 2] < 0, 2] = 1
        self._active_risk = risk
        try:
            return super()._navigate(
                name, state, planning_states, target, kind, walkable, shape
            )
        finally:
            self._active_risk = None

    def _astar(
        self,
        walkable: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        risk = self._active_risk
        if risk is None or self.risk_weight <= 0:
            return super()._astar(walkable, start, goal)
        h, w = walkable.shape
        if not (0 <= goal[0] < h and 0 <= goal[1] < w):
            return []
        if _native.NUMBA_AVAILABLE:
            path_y, path_x = _native.risk_astar_numba(
                walkable, risk, self._hypot_table(h, w), start[0], start[1], goal[0], goal[1],
                float(self.risk_weight), np.float32(0.15),
            )
            return list(zip(path_y.tolist(), path_x.tolist()))
        neighbours = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                      (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))
        queue = [(0.0, 0.0, start)]
        came = {}
        costs = {start: 0.0}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost != costs.get(current):
                continue
            if current == goal:
                path = [current]
                while path[-1] != start:
                    path.append(came[path[-1]])
                path.reverse()
                return path
            y, x = current
            for dy, dx, step in neighbours:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or not walkable[ny, nx]:
                    continue
                if dy and dx and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                # Degree penalty discourages entering a narrow pocket while threats are near.
                degree = sum(
                    0 <= ny + ddy < h and 0 <= nx + ddx < w
                    and walkable[ny + ddy, nx + ddx]
                    for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                narrow = 0.45 if degree <= 2 and risk[ny, nx] > 0.15 else 0.0
                new_cost = cost + step * (1.0 + self.risk_weight * float(risk[ny, nx])) + narrow
                node = (ny, nx)
                if new_cost + 1e-6 >= costs.get(node, float("inf")):
                    continue
                costs[node] = new_cost
                came[node] = current
                heuristic = math.hypot(goal[0] - ny, goal[1] - nx)
                heapq.heappush(queue, (new_cost + heuristic, new_cost, node))
        return []

    def _hypot_table(self, h: int, w: int) -> np.ndarray:
        """CPython ``math.hypot`` for every |dy|, |dx| on the grid, for the Numba kernel."""
        table = self._hypot_tables.get((h, w))
        if table is None:
            table = np.array(
                [[math.hypot(dy, dx) for dx in range(w)] for dy in range(h)], dtype=np.float64
            )
            self._hypot_tables[(h, w)] = table
        return table
