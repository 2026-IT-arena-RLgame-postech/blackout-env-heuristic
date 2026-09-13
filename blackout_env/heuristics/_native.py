"""Optional Numba-JIT kernels for heuristic hot paths.

``blackout_env.heuristics.advanced._distance_map`` is an 8-neighbour Dijkstra over the
semantic-map grid, called once per unique unit pixel per decision tick (memoised per
episode by ``_cached_distance_map``). Profiling a full match under cProfile showed it
dominates wall-clock time for every heuristic built on ``StrategicHeuristicV2``
(V3/V4/V5/V7/V8/V10/V11/V12) — it is pure numeric code (bool grid in, float32 grid out)
with no Python objects in the loop, which makes it a good Numba target unlike the rest
of the heuristic stack (dataclasses, OrderedDict caches, heapq of tuples).

Numba is optional: if it isn't installed, ``NUMBA_AVAILABLE`` is False and callers fall
back to the pure-Python implementation.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import boolean, float32, int64, njit

    _SIGNATURE = float32[:, :](boolean[:, :], int64, int64)

    @njit(_SIGNATURE, cache=True, fastmath=True)
    def distance_map_numba(walkable: np.ndarray, start_y: int, start_x: int) -> np.ndarray:
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

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def distance_map_numba(walkable: np.ndarray, start_y: int, start_x: int) -> np.ndarray:  # noqa: D401
        raise RuntimeError("numba is not installed")
