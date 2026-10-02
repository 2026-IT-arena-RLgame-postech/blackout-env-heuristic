"""UnityEngine.Random (Xorshift128) and the seeded parts of match generation.

``Random.InitState(seed)`` fills the four 32-bit words with the MT19937 initialiser recurrence;
every draw is one xorshift128 step.  ``Range(int min, int max)`` is ``min + draw % (max - min)``
and ``value`` is ``(draw & 0x7FFFFF) / 8388607``.  This was verified against recorded matches:
the storage choice and every battery's tile and amount reproduce from the env's reset seed
(tests/test_sim_rules.py::test_seeded_generation_matches_trace).

The state is a ``uint32[4]`` array so numba kernels can draw from it in place.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

from .mapdata import (FLOOR, H, NEUTRAL, STORAGE_CANDIDATES, STORAGE_COUNT, W)

MASK = 0xFFFFFFFF
INITIAL_BATTERY_TOTAL = 200
MIN_BATTERY, MAX_BATTERY = 1, 8


def init_state(seed: int) -> np.ndarray:
    s = np.zeros(4, dtype=np.uint32)
    x = int(seed) & MASK
    s[0] = x
    for i in range(1, 4):
        x = (1812433253 * x + 1) & MASK
        s[i] = x
    return s


@njit(cache=True)
def next_uint(s) -> int:
    # numba widens uint32 arithmetic to 64 bits: mask explicitly
    m = np.uint64(0xFFFFFFFF)
    x = np.uint64(s[0])
    w = np.uint64(s[3])
    t = (x ^ (x << np.uint64(11))) & m
    s[0] = s[1]
    s[1] = s[2]
    s[2] = s[3]
    r = (w ^ (w >> np.uint64(19)) ^ t ^ (t >> np.uint64(8))) & m
    s[3] = r
    return np.int64(r)


@njit(cache=True)
def range_int(s, lo: int, hi: int) -> int:
    if hi <= lo:
        return lo
    return lo + next_uint(s) % (hi - lo)


@njit(cache=True)
def value(s) -> np.float32:
    return np.float32(next_uint(s) & 0x7FFFFF) * np.float32(1.0 / 8388607.0)


def generate(seed: int):
    """Replays ``GenerateMapData`` + ``SpawnItemObjects`` for ``seed``.

    Returns (chosen candidate indices in shuffle order, batteries [(x, y, amount)] in spawn
    order, RNG state after generation)."""
    from .mapdata import build_layout

    s = init_state(seed)
    keys = [float(value(s)) for _ in STORAGE_CANDIDATES]
    order = sorted(range(len(keys)), key=lambda i: keys[i])  # OrderBy is a stable sort
    chosen = tuple(order[:STORAGE_COUNT])
    lay = build_layout(chosen)
    valid = []
    for y in range(H):
        for x in range(W):
            if lay.reg_team[lay.region[x, y]] == NEUTRAL and lay.tile[x, y] == FLOOR:
                valid.append((x, y))
    batteries, total = [], 0
    while total < INITIAL_BATTERY_TOTAL and valid:
        idx = range_int(s, 0, len(valid))
        x, y = valid.pop(idx)
        amount = range_int(s, MIN_BATTERY, MAX_BATTERY + 1)
        batteries.append((x, y, amount))
        total += amount
    return chosen, batteries, s
