"""Optional Numba-JIT kernels for heuristic hot paths.

Three kernels, each a drop-in for a pure-Python routine:

* ``distance_map_numba`` -- ``advanced.StrategicHeuristicV2._distance_map`` (all-cell path
  lengths; task assignment, V5 intercepts, V7+ role choice, V17 planning);
* ``astar_numba`` -- ``strategic.StrategicHeuristic._astar`` (every unit's path, V1+);
* ``risk_astar_numba`` -- ``risk_path.StrategicHeuristicV6._astar`` (threat-cost path for
  cargo holders in V6 and V17-V19).

``blackout_env.heuristics.advanced._distance_map`` is an 8-neighbour Dijkstra over the
semantic-map grid, called once per unique unit pixel per decision tick (memoised per
episode by ``_cached_distance_map``). Profiling a full match under cProfile showed it
dominates wall-clock time for every heuristic built on ``StrategicHeuristicV2``
(all of V2-V19) — it is pure numeric code (bool grid in, float32 grid out)
with no Python objects in the loop, which makes it a good Numba target unlike the rest
of the heuristic stack (dataclasses, OrderedDict caches, heapq of tuples).  The two A*
kernels follow the same rule: identical paths to the Python versions, tie-breaks included.

Numba is optional: if it isn't installed, ``NUMBA_AVAILABLE`` is False and callers fall
back to the pure-Python implementation.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import boolean, float32, int64, njit

    _SIGNATURE = float32[:, :](boolean[:, :], int64, int64)

    @njit(_SIGNATURE, cache=True, fastmath=True)
    def distance_map_numba(walkable: np.ndarray, start_y: int, start_x: int) -> np.ndarray:
        """8-neighbour Dijkstra from one cell with corner-cut prevention (float32, inf = unreachable).

        Hand-rolled binary heap over parallel arrays because Numba has no heapq of tuples.
        """
        h, w = walkable.shape
        distances = np.full((h, w), np.inf, dtype=np.float32)
        if not (0 <= start_y < h and 0 <= start_x < w):
            return distances
        distances[start_y, start_x] = 0.0

        # Each cell can only be pushed once per incoming edge that improves its distance,
        # and it has at most 8 neighbours, so this bound is never exceeded.
        capacity = h * w * 8 + 1
        heap_cost = np.empty(capacity, dtype=np.float64)
        heap_y = np.empty(capacity, dtype=np.int32)
        heap_x = np.empty(capacity, dtype=np.int32)
        size = 1
        heap_cost[0] = 0.0
        heap_y[0] = start_y
        heap_x[0] = start_x

        dys = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
        dxs = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
        steps = np.array([1.0, 1.0, 1.0, 1.0, 1.4142, 1.4142, 1.4142, 1.4142], dtype=np.float64)

        while size > 0:
            cost = heap_cost[0]
            y = heap_y[0]
            x = heap_x[0]
            size -= 1
            heap_cost[0] = heap_cost[size]
            heap_y[0] = heap_y[size]
            heap_x[0] = heap_x[size]
            i = 0
            while True:
                left = 2 * i + 1
                right = 2 * i + 2
                smallest = i
                if left < size and heap_cost[left] < heap_cost[smallest]:
                    smallest = left
                if right < size and heap_cost[right] < heap_cost[smallest]:
                    smallest = right
                if smallest == i:
                    break
                heap_cost[i], heap_cost[smallest] = heap_cost[smallest], heap_cost[i]
                heap_y[i], heap_y[smallest] = heap_y[smallest], heap_y[i]
                heap_x[i], heap_x[smallest] = heap_x[smallest], heap_x[i]
                i = smallest

            # Same lazy-deletion + float32/float64 epsilon tolerance as the pure-Python
            # version: a stale heap entry (superseded by a later, cheaper push) is skipped
            # instead of removed, and diagonal costs (1.4142) need slack against float32
            # rounding in `distances` or they get rejected almost every time.
            if cost > float(distances[y, x]) + 1e-5:
                continue

            for k in range(8):
                ny = y + dys[k]
                nx = x + dxs[k]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not walkable[ny, nx]:
                    continue
                if dys[k] != 0 and dxs[k] != 0 and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                new_cost = cost + steps[k]
                if new_cost + 1e-6 >= float(distances[ny, nx]):
                    continue
                distances[ny, nx] = new_cost
                j = size
                heap_cost[j] = new_cost
                heap_y[j] = ny
                heap_x[j] = nx
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if heap_cost[parent] <= heap_cost[j]:
                        break
                    heap_cost[parent], heap_cost[j] = heap_cost[j], heap_cost[parent]
                    heap_y[parent], heap_y[j] = heap_y[j], heap_y[parent]
                    heap_x[parent], heap_x[j] = heap_x[j], heap_x[parent]
                    j = parent

        return distances

    @njit(cache=True, inline="always")
    def _less_fgyx(f1, g1, y1, x1, f2, g2, y2, x2) -> bool:
        """Match Python's ``heapq`` ordering on ``(f, g, (y, x))`` tuples exactly.

        Needed so ties (equal f *and* g) resolve the same way as the pure-Python
        version — otherwise A* still finds an optimal path, but a differently-tied
        one, which would silently change which route a unit walks in ambiguous cases.
        """
        if f1 != f2:
            return f1 < f2
        if g1 != g2:
            return g1 < g2
        if y1 != y2:
            return y1 < y2
        return x1 < x2

    @njit(cache=True, fastmath=True)
    def astar_numba(walkable: np.ndarray, start_y: int, start_x: int, goal_y: int, goal_x: int):
        """8-neighbour A* with diagonal corner-cut prevention.

        Caller guarantees ``goal`` is in bounds and ``start != goal`` (see
        ``strategic.StrategicHeuristic._astar``), so this only handles the search itself.
        Returns parallel (path_y, path_x) int32 arrays, empty when no path exists.
        """
        h, w = walkable.shape
        max_nodes = h * w
        cost = np.full((h, w), np.inf, dtype=np.float64)
        came_y = np.full((h, w), -1, dtype=np.int32)
        came_x = np.full((h, w), -1, dtype=np.int32)
        cost[start_y, start_x] = 0.0

        capacity = max_nodes * 8 + 1
        heap_f = np.empty(capacity, dtype=np.float64)
        heap_g = np.empty(capacity, dtype=np.float64)
        heap_y = np.empty(capacity, dtype=np.int32)
        heap_x = np.empty(capacity, dtype=np.int32)
        size = 1
        heap_f[0] = 0.0
        heap_g[0] = 0.0
        heap_y[0] = start_y
        heap_x[0] = start_x

        dys = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
        dxs = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
        steps = np.array([1.0, 1.0, 1.0, 1.0, 1.4142, 1.4142, 1.4142, 1.4142], dtype=np.float64)

        found = False
        while size > 0:
            g = heap_g[0]
            y = heap_y[0]
            x = heap_x[0]
            size -= 1
            heap_f[0] = heap_f[size]
            heap_g[0] = heap_g[size]
            heap_y[0] = heap_y[size]
            heap_x[0] = heap_x[size]
            i = 0
            while True:
                left = 2 * i + 1
                right = 2 * i + 2
                smallest = i
                if left < size and _less_fgyx(
                    heap_f[left], heap_g[left], heap_y[left], heap_x[left],
                    heap_f[smallest], heap_g[smallest], heap_y[smallest], heap_x[smallest],
                ):
                    smallest = left
                if right < size and _less_fgyx(
                    heap_f[right], heap_g[right], heap_y[right], heap_x[right],
                    heap_f[smallest], heap_g[smallest], heap_y[smallest], heap_x[smallest],
                ):
                    smallest = right
                if smallest == i:
                    break
                heap_f[i], heap_f[smallest] = heap_f[smallest], heap_f[i]
                heap_g[i], heap_g[smallest] = heap_g[smallest], heap_g[i]
                heap_y[i], heap_y[smallest] = heap_y[smallest], heap_y[i]
                heap_x[i], heap_x[smallest] = heap_x[smallest], heap_x[i]
                i = smallest

            # Lazy deletion: skip a heap entry superseded by a later, cheaper push instead
            # of removing it. g/cost are both float64 here (unlike distance_map_numba)
            # so exact equality is safe.
            if g != cost[y, x]:
                continue
            if y == goal_y and x == goal_x:
                found = True
                break

            for k in range(8):
                ny = y + dys[k]
                nx = x + dxs[k]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not walkable[ny, nx]:
                    continue
                if dys[k] != 0 and dxs[k] != 0 and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                ng = g + steps[k]
                if ng >= cost[ny, nx]:
                    continue
                cost[ny, nx] = ng
                came_y[ny, nx] = y
                came_x[ny, nx] = x
                dy_goal = float(goal_y - ny)
                dx_goal = float(goal_x - nx)
                heuristic = math.hypot(dy_goal, dx_goal)
                nf = ng + heuristic
                j = size
                heap_f[j] = nf
                heap_g[j] = ng
                heap_y[j] = ny
                heap_x[j] = nx
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if not _less_fgyx(
                        heap_f[j], heap_g[j], heap_y[j], heap_x[j],
                        heap_f[parent], heap_g[parent], heap_y[parent], heap_x[parent],
                    ):
                        break
                    heap_f[parent], heap_f[j] = heap_f[j], heap_f[parent]
                    heap_g[parent], heap_g[j] = heap_g[j], heap_g[parent]
                    heap_y[parent], heap_y[j] = heap_y[j], heap_y[parent]
                    heap_x[parent], heap_x[j] = heap_x[j], heap_x[parent]
                    j = parent

        if not found:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        path_y_buf = np.empty(max_nodes, dtype=np.int32)
        path_x_buf = np.empty(max_nodes, dtype=np.int32)
        cy, cx = goal_y, goal_x
        n = 0
        while not (cy == start_y and cx == start_x):
            path_y_buf[n] = cy
            path_x_buf[n] = cx
            n += 1
            py = came_y[cy, cx]
            px = came_x[cy, cx]
            cy, cx = py, px
        path_y_buf[n] = start_y
        path_x_buf[n] = start_x
        n += 1

        out_y = np.empty(n, dtype=np.int32)
        out_x = np.empty(n, dtype=np.int32)
        for idx in range(n):
            out_y[idx] = path_y_buf[n - 1 - idx]
            out_x[idx] = path_x_buf[n - 1 - idx]
        return out_y, out_x

    # No fastmath here: the path must equal risk_path.StrategicHeuristicV6's pure-Python A*
    # bit for bit, and fastmath licenses reassociation/FMA that change float64 results.
    @njit(cache=True)
    def risk_astar_numba(
        walkable: np.ndarray,
        risk: np.ndarray,
        hypot_table: np.ndarray,
        start_y: int,
        start_x: int,
        goal_y: int,
        goal_x: int,
        risk_weight: float,
        narrow_risk: np.float32,
    ):
        """V6's threat-cost A*: 8 neighbours, corner-cut prevention, narrow-pocket penalty.

        Caller guarantees ``goal`` is in bounds. ``hypot_table[|dy|, |dx|]`` holds CPython's
        ``math.hypot`` values (which need not match libm's ulp for ulp), and ``narrow_risk``
        is float32 because NumPy compares a float32 cell with a Python float in float32.
        Returns parallel (path_y, path_x) int32 arrays, empty when no path exists.
        """
        h, w = walkable.shape
        degree = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                count = 0
                if y > 0 and walkable[y - 1, x]:
                    count += 1
                if y + 1 < h and walkable[y + 1, x]:
                    count += 1
                if x > 0 and walkable[y, x - 1]:
                    count += 1
                if x + 1 < w and walkable[y, x + 1]:
                    count += 1
                degree[y, x] = count

        cost = np.full((h, w), np.inf, dtype=np.float64)
        came_y = np.full((h, w), -1, dtype=np.int32)
        came_x = np.full((h, w), -1, dtype=np.int32)
        cost[start_y, start_x] = 0.0

        capacity = h * w * 8 + 1
        heap_f = np.empty(capacity, dtype=np.float64)
        heap_g = np.empty(capacity, dtype=np.float64)
        heap_y = np.empty(capacity, dtype=np.int32)
        heap_x = np.empty(capacity, dtype=np.int32)
        size = 1
        heap_f[0] = 0.0
        heap_g[0] = 0.0
        heap_y[0] = start_y
        heap_x[0] = start_x

        dys = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
        dxs = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
        steps = np.array([1.0, 1.0, 1.0, 1.0, 1.4142, 1.4142, 1.4142, 1.4142], dtype=np.float64)

        found = False
        while size > 0:
            g = heap_g[0]
            y = heap_y[0]
            x = heap_x[0]
            size -= 1
            heap_f[0] = heap_f[size]
            heap_g[0] = heap_g[size]
            heap_y[0] = heap_y[size]
            heap_x[0] = heap_x[size]
            i = 0
            while True:
                left = 2 * i + 1
                right = 2 * i + 2
                smallest = i
                if left < size and _less_fgyx(
                    heap_f[left], heap_g[left], heap_y[left], heap_x[left],
                    heap_f[smallest], heap_g[smallest], heap_y[smallest], heap_x[smallest],
                ):
                    smallest = left
                if right < size and _less_fgyx(
                    heap_f[right], heap_g[right], heap_y[right], heap_x[right],
                    heap_f[smallest], heap_g[smallest], heap_y[smallest], heap_x[smallest],
                ):
                    smallest = right
                if smallest == i:
                    break
                heap_f[i], heap_f[smallest] = heap_f[smallest], heap_f[i]
                heap_g[i], heap_g[smallest] = heap_g[smallest], heap_g[i]
                heap_y[i], heap_y[smallest] = heap_y[smallest], heap_y[i]
                heap_x[i], heap_x[smallest] = heap_x[smallest], heap_x[i]
                i = smallest

            if g != cost[y, x]:
                continue
            if y == goal_y and x == goal_x:
                found = True
                break

            for k in range(8):
                ny = y + dys[k]
                nx = x + dxs[k]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not walkable[ny, nx]:
                    continue
                if dys[k] != 0 and dxs[k] != 0 and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                cell_risk = risk[ny, nx]
                narrow = 0.45 if degree[ny, nx] <= 2 and cell_risk > narrow_risk else 0.0
                ng = g + steps[k] * (1.0 + risk_weight * np.float64(cell_risk)) + narrow
                if ng + 1e-6 >= cost[ny, nx]:
                    continue
                cost[ny, nx] = ng
                came_y[ny, nx] = y
                came_x[ny, nx] = x
                nf = ng + hypot_table[abs(goal_y - ny), abs(goal_x - nx)]
                if size == capacity:
                    capacity *= 2
                    heap_f = np.concatenate((heap_f, np.empty(capacity - size, dtype=np.float64)))
                    heap_g = np.concatenate((heap_g, np.empty(capacity - size, dtype=np.float64)))
                    heap_y = np.concatenate((heap_y, np.empty(capacity - size, dtype=np.int32)))
                    heap_x = np.concatenate((heap_x, np.empty(capacity - size, dtype=np.int32)))
                j = size
                heap_f[j] = nf
                heap_g[j] = ng
                heap_y[j] = ny
                heap_x[j] = nx
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if not _less_fgyx(
                        heap_f[j], heap_g[j], heap_y[j], heap_x[j],
                        heap_f[parent], heap_g[parent], heap_y[parent], heap_x[parent],
                    ):
                        break
                    heap_f[parent], heap_f[j] = heap_f[j], heap_f[parent]
                    heap_g[parent], heap_g[j] = heap_g[j], heap_g[parent]
                    heap_y[parent], heap_y[j] = heap_y[j], heap_y[parent]
                    heap_x[parent], heap_x[j] = heap_x[j], heap_x[parent]
                    j = parent

        if not found:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        max_nodes = h * w
        path_y_buf = np.empty(max_nodes, dtype=np.int32)
        path_x_buf = np.empty(max_nodes, dtype=np.int32)
        cy, cx = goal_y, goal_x
        n = 0
        while not (cy == start_y and cx == start_x):
            path_y_buf[n] = cy
            path_x_buf[n] = cx
            n += 1
            py = came_y[cy, cx]
            px = came_x[cy, cx]
            cy, cx = py, px
        path_y_buf[n] = start_y
        path_x_buf[n] = start_x
        n += 1

        out_y = np.empty(n, dtype=np.int32)
        out_x = np.empty(n, dtype=np.int32)
        for idx in range(n):
            out_y[idx] = path_y_buf[n - 1 - idx]
            out_x[idx] = path_x_buf[n - 1 - idx]
        return out_y, out_x

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def distance_map_numba(walkable: np.ndarray, start_y: int, start_x: int) -> np.ndarray:  # noqa: D401
        raise RuntimeError("numba is not installed")

    def astar_numba(walkable: np.ndarray, start_y: int, start_x: int, goal_y: int, goal_x: int):  # noqa: D401
        raise RuntimeError("numba is not installed")

    def risk_astar_numba(*args):  # noqa: D401
        raise RuntimeError("numba is not installed")
