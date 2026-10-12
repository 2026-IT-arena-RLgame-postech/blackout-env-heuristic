"""The reset seed from the first official observation (storage choice + battery tiles).

``Random.InitState(seed)`` has 2^32 states.  Generation draws 5 keys (the storage shuffle: the
largest key is the candidate left out), then per battery a tile ``valid.pop(Range(0, n))`` and an
amount.  The first observation shows which candidate is out (1/5 survive) and the *set* of
battery tiles (a draw lands in it with p ~ 45/280), so checking the first K tile draws leaves
~2^32 / 5 * 0.16^K seeds -- K = 12 makes a false hit astronomically rare; survivors are then
verified against the whole set.  The scan is embarrassingly parallel (numba prange).

When Unity was not reseeded between matches (the official ``run_match`` resets without a seed)
the next match's generation continues from the previous match's RNG state: ``find_continuation``
replays a known state forward over a window of draw offsets instead.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from ..sim.mapdata import FLOOR, H, NEUTRAL, STORAGE_CANDIDATES, W, build_layout
from ..sim.unity_random import generate, generate_state, init_state

N_CAND = len(STORAGE_CANDIDATES)
K_CHECK = 12
MAX_HITS = 64


def valid_cells(chosen) -> np.ndarray:
    """``SpawnItemObjects`` candidate cells in ``allTilesCache`` order (y outer, x inner)."""
    lay = build_layout(tuple(chosen))
    out = [(x, y) for y in range(H) for x in range(W)
           if lay.reg_team[lay.region[x, y]] == NEUTRAL and lay.tile[x, y] == FLOOR]
    return np.array(out, dtype=np.int64)


M32 = np.uint64(0xFFFFFFFF)


@njit(cache=True, inline='always')
def _xs(s0, s1, s2, s3):
    """One xorshift128 draw on uint64-held words (numba widens uint32 arithmetic: mask)."""
    t = (s0 ^ (s0 << np.uint64(11))) & M32
    r = (s3 ^ (s3 >> np.uint64(19)) ^ t ^ (t >> np.uint64(8))) & M32
    return s1, s2, s3, r


@njit(cache=True)
def _check_state(s0, s1, s2, s3, excluded, in_set, n_valid, k_check):
    """Keys, then the first ``k_check`` battery tiles; True if all consistent."""
    best = -1
    best_k = np.uint64(0)
    for c in range(N_CAND):
        s0, s1, s2, s3 = _xs(s0, s1, s2, s3)
        k = s3 & np.uint64(0x7FFFFF)
        if c == 0 or k >= best_k:  # stable OrderBy: of equal keys the later candidate sorts last
            best_k = k
            best = c
    if best != excluded:
        return False
    rem = np.arange(n_valid)
    n = n_valid
    for _ in range(k_check):
        s0, s1, s2, s3 = _xs(s0, s1, s2, s3)
        idx = np.int64(s3 % np.uint64(n))
        cell = rem[idx]
        if not in_set[cell]:
            return False
        rem[idx:n - 1] = rem[idx + 1:n]
        n -= 1
        s0, s1, s2, s3 = _xs(s0, s1, s2, s3)  # amount
    return True


@njit(cache=True, parallel=True)
def _scan(lo, hi, excluded, in_set, n_valid, k_check, hits, n_hits, n_chunks):
    span = hi - lo
    chunk = (span + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        a = lo + c * chunk
        b = min(hi, a + chunk)
        for seed in range(a, b):
            s0 = np.uint64(seed) & M32
            s1 = (np.uint64(1812433253) * s0 + np.uint64(1)) & M32
            s2 = (np.uint64(1812433253) * s1 + np.uint64(1)) & M32
            s3 = (np.uint64(1812433253) * s2 + np.uint64(1)) & M32
            # cheapest filter first: the excluded candidate needs the largest of 5 keys
            if _check_state(s0, s1, s2, s3, excluded, in_set, n_valid, k_check):
                k = n_hits[c]
                if k < hits.shape[1]:
                    hits[c, k] = seed
                n_hits[c] = k + 1


def _target(chosen, tiles):
    cells = valid_cells(chosen)
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    in_set = np.zeros(len(cells), dtype=np.bool_)
    for t in tiles:
        if tuple(t) in index:
            in_set[index[tuple(t)]] = True
    excluded = [c for c in range(N_CAND) if c not in chosen][0]
    return excluded, in_set, len(cells)


def matches(seed: int, chosen, tiles) -> bool:
    ch, batteries, _ = generate(int(seed) & 0xFFFFFFFF)
    return set(ch) == set(chosen) and {(x, y) for x, y, _ in batteries} == {tuple(map(int, t)) for t in tiles}


def find_seed(chosen, tiles, lo: int = 0, hi: int = 1 << 32, n_chunks: int = 4096, verbose=False):
    """All seeds in [lo, hi) generating this storage set and battery tile set (normally one)."""
    excluded, in_set, n_valid = _target(chosen, tiles)
    hits = np.zeros((n_chunks, MAX_HITS), dtype=np.int64)
    n_hits = np.zeros(n_chunks, dtype=np.int64)
    t0 = time.time()
    _scan(lo, hi, excluded, in_set, n_valid, K_CHECK, hits, n_hits, n_chunks)
    cands = [int(hits[c, k]) for c in range(n_chunks) for k in range(min(n_hits[c], MAX_HITS))]
    out = [s for s in cands if matches(s, chosen, tiles)]
    if verbose:
        print(f"scan [{lo}, {hi}) {time.time() - t0:.1f}s: {len(cands)} candidates, {len(out)} verified")
    return out


def seed_finder(lo: int = 0, hi: int = 1 << 32):
    """``OfficialTracker`` hook: the unique seed, else None."""
    def find(chosen, tiles):
        out = find_seed(chosen, tiles, lo, hi)
        return out[0] if len(out) == 1 else None
    return find


generate_from_state = generate_state


def find_continuation(state, chosen, tiles, max_offset: int = 4096):
    """The draw offset k (state advanced k draws) whose generation shows this map, or None."""
    from ..sim.unity_random import next_uint

    s = np.array(state, dtype=np.uint32).copy()
    want = {tuple(map(int, t)) for t in tiles}
    for k in range(max_offset + 1):
        ch, bat, after = generate_from_state(s)
        if set(ch) == set(chosen) and {(x, y) for x, y, _ in bat} == want:
            return k, s.copy(), ch, bat, after
        next_uint(s)
    return None


if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for _ in range(3):
        seed = int(rng.integers(0, 1 << 32))
        ch, bat, _ = generate(seed)
        t0 = time.time()
        found = find_seed(ch, [(x, y) for x, y, _ in bat], verbose=True)
        print(f"seed {seed}: found {found} in {time.time() - t0:.1f}s")


@njit(cache=True)
def _scan_offsets(state, max_offset, excluded, in_set, n_valid, k_check, hits):
    """Offsets k <= max_offset at which a generation from ``state`` advanced k draws fits."""
    s0, s1, s2, s3 = np.uint64(state[0]), np.uint64(state[1]), np.uint64(state[2]), np.uint64(state[3])
    n = 0
    for k in range(max_offset + 1):
        if _check_state(s0, s1, s2, s3, excluded, in_set, n_valid, k_check):
            if n < hits.shape[0]:
                hits[n] = k
            n += 1
        s0, s1, s2, s3 = _xs(s0, s1, s2, s3)
    return n


def advance(state, k: int) -> np.ndarray:
    from ..sim.unity_random import next_uint

    s = np.array(state, dtype=np.uint32).copy()
    for _ in range(k):
        next_uint(s)
    return s


def find_offset(state, chosen, tiles, max_offset: int = 10_000_000):
    """Fast ``find_continuation``: (offset, state at generation start) or None."""
    excluded, in_set, n_valid = _target(chosen, tiles)
    hits = np.zeros(MAX_HITS, dtype=np.int64)
    n = _scan_offsets(np.asarray(state, dtype=np.uint64), max_offset, excluded, in_set, n_valid, K_CHECK, hits)
    want = {tuple(map(int, t)) for t in tiles}
    for k in hits[:min(n, MAX_HITS)]:
        s = advance(state, int(k))
        ch, bat, _ = generate_from_state(s)
        if set(ch) == set(chosen) and {(x, y) for x, y, _ in bat} == want:
            return int(k), s
    return None


class RngFinder:
    """The UnityEngine.Random state a match's generation started from, given its first frame.

    Tried in order (each verified against the storage set and every battery tile):
      1. ``prev``: the previous match's state as the tracker left it -- an unseeded Unity just
         runs on (offset 0 unless draws went unseen);
      2. ``InitState(env_seed) + offset``: ML-Agents' Academy calls ``Random.InitState`` with the
         UnityEnvironment seed (0 unless given) when the communicator connects; the startup
         ``GameScenario.Initialize`` then generates a throw-away map (109 draws for seed 0) and
         every later match continues the same stream -- measured on the Mac build, amounts too;
      3. every ``InitState(seed)``: a match reset with an explicit seed (~11 s, 18 threads).
    Returns (state, how) or None."""

    def __init__(self, env_seed: int = 0, max_offset: int = 2_000_000, prev_window: int = 100_000,
                 full_scan: bool = True):
        self.env_seed, self.max_offset, self.prev_window, self.full_scan = env_seed, max_offset, prev_window, full_scan

    def __call__(self, chosen, tiles, prev=None):
        if prev is not None:
            r = find_offset(prev, chosen, tiles, self.prev_window)
            if r is not None:
                return r[1], f"prev+{r[0]}"
        r = find_offset(init_state(self.env_seed), chosen, tiles, self.max_offset)
        if r is not None:
            return r[1], f"init{self.env_seed}+{r[0]}"
        if self.full_scan:
            seeds = find_seed(chosen, tiles)
            if len(seeds) == 1:
                return init_state(seeds[0]), f"seed {seeds[0]}"
        return None
