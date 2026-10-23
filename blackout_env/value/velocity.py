"""Unit velocity (Delta-pos) features of the value net: ``agent_vel`` f32 [N, 10, 3] = (vx, vy, valid).

Per row of a shard (one match, canonical frame, rows = decision ticks, 25 per second), per unit:
position(t) - position(previous row), in the same normalised [-1, 1] frame as ``agent_states``
columns 0-1, divided by the tick gap and by ``SPEED_UNIT`` (= one normal move, 0.02 per tick, so a
unit walking at normal speed has |v| ~ 1 and a boosted one ~ 1.5).  ``valid`` = 1 when the value is
a real velocity; vx = vy = valid = 0 on the first row of a match, when the previous row is more
than ``MAX_GAP`` ticks back (old shards drop unlabelled ticks) or its tick is unknown, and on a
death / respawn / teleport: a displacement larger than ``MAX_STEP`` x gap (the fastest real move
in the shards is ~0.03-0.04 per tick, 1.5 x the normal 0.02; the tracker's bound is
official/tracker.py ``_tick``).  The previous *decision row's* position is what a policy or search
leaf has at play time (the previous observation; official/tracker.py keeps per-observation
positions).
"""

from __future__ import annotations

import numpy as np

SPEED_UNIT = 0.02  # normalised position change per tick of a normal-speed unit
MAX_STEP = 0.045  # per tick: larger = teleport (observed fastest real move ~0.04)
MAX_GAP = 10  # ticks
VEL_CLIP = 3.0


def shard_velocity(agent_states: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """``agent_states`` [T, 10, D] (cols 0-1 = x, y), ``ticks`` [T] -> agent_vel f32 [T, 10, 3]."""
    t = len(agent_states)
    out = np.zeros((t, agent_states.shape[1], 3), np.float32)
    if t < 2:
        return out
    ticks = np.asarray(ticks, np.int64)
    gap = np.diff(ticks)  # [T-1]
    delta = np.diff(agent_states[:, :, :2].astype(np.float64), axis=0)  # [T-1, 10, 2]
    ok_gap = (gap >= 1) & (gap <= MAX_GAP) & (ticks[:-1] >= 0)
    g = np.maximum(gap, 1)[:, None]
    dist = np.abs(delta).max(-1)  # [T-1, 10]
    valid = ok_gap[:, None] & (dist <= MAX_STEP * g)
    vel = np.clip(delta / (g[:, :, None] * SPEED_UNIT), -VEL_CLIP, VEL_CLIP)
    out[1:, :, :2] = np.where(valid[:, :, None], vel, 0.0)
    out[1:, :, 2] = valid
    return out
