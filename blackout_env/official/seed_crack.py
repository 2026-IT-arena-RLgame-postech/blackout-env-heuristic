"""Seed recovery from the first frame, as batched tensor ops (exportable to ONNX; prototype).

``Random.InitState(seed)`` + ``GenerateMapData``/``SpawnItemObjects`` (sim/unity_random.py): 5 ``value`` draws pick
the 4 of 5 storage candidates (the largest key is left out), then each battery takes ``valid.pop(draw % len)`` and
an amount ``1 + draw % 8``.  The first frame shows the left-out candidate and the set of battery tiles, so a seed
is checked by replaying the first K battery picks: each must land on an observed battery tile (~15 % by chance),
K = 12 leaves ~1e-10 false positives per seed.  ``check_seeds`` tests a block of candidate seeds at once.
"""

from __future__ import annotations

import torch

M32 = 0xFFFFFFFF
MULT = 1812433253


def _next(s0, s1, s2, s3):
    t = (s0 ^ ((s0 << 11) & M32)) & M32
    r = (s3 ^ (s3 >> 19) ^ t ^ (t >> 8)) & M32
    return s1, s2, s3, r


def check_seeds(seeds: torch.Tensor, excluded: int, valid_count: int, observed: torch.Tensor, k: int = 12):
    """seeds int64 [C]; excluded: the storage candidate missing on the map; valid_count: len(valid) for that
    storage choice; observed: bool [valid_count] = valid tile i holds a battery in the first frame.
    Returns bool [C]: the seed reproduces the storage choice and the first k battery tiles."""
    s0 = seeds & M32
    s1 = (MULT * s0 + 1) & M32
    s2 = (MULT * s1 + 1) & M32
    s3 = (MULT * s2 + 1) & M32
    keys = []
    for _ in range(5):
        s0, s1, s2, s3 = _next(s0, s1, s2, s3)
        keys.append(s3 & 0x7FFFFF)
    keys = torch.stack(keys, dim=1)
    # OrderBy (stable): the left-out candidate is the last of the sort = the largest key, the latest index on ties
    top = keys.max(dim=1, keepdim=True).values
    idx = torch.arange(5, device=seeds.device).expand_as(keys)
    left_out = torch.where(keys == top, idx, torch.full_like(idx, -1)).max(dim=1).values
    ok = left_out == excluded
    picks = []
    for i in range(k):
        s0, s1, s2, s3 = _next(s0, s1, s2, s3)
        r = s3 % (valid_count - i)
        orig = r
        for _ in range(i):  # index in the remaining list -> original index (skip earlier picks)
            if picks:
                below = torch.stack([(p <= orig) for p in picks], dim=1).sum(dim=1)
                orig = r + below
        picks.append(orig)
        ok = ok & observed.index_select(0, orig.clamp(0, valid_count - 1))
        s0, s1, s2, s3 = _next(s0, s1, s2, s3)  # the amount draw
    return ok
