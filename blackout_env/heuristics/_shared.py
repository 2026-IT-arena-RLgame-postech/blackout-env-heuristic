"""Process-wide caches of pure functions of the map (O1, docs/policy_v2_design_ko.md "O1").

The heuristics cache distance maps, any-angle fields and race planners per instance and drop them at
every episode boundary (``reset``), so every new match -- and every new policy instance, one per
match in the sim evaluations -- rebuilt them.  They are pure functions of their key (the walkable mask's
bytes and the sources), so one process-wide LRU serves every instance and every match on the same map.
Cached arrays are made read-only: they are shared, and an in-place write would fail loudly instead of
leaking into another instance.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class SharedCache:
    """A bounded LRU dict (least recently used entry evicted first)."""

    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        self._data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        value = self._data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return value

    def put(self, key, value):
        self._data[key] = value
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# ~2.3 KB per 24x24 float32 map
DISTANCE_MAPS = SharedCache(16384)    # (distance fn, shape, walkable bytes, start) -> map
GOAL_FIELDS = SharedCache(8192)       # (kind, shape, walkable bytes, sorted tiles) -> field (+ nodes)
RACE_PLANNERS = SharedCache(64)       # (shape, walkable bytes, tiles, radius, delta) -> RacePlanner


def clear_all() -> None:
    for cache in (DISTANCE_MAPS, GOAL_FIELDS, RACE_PLANNERS):
        cache.clear()
