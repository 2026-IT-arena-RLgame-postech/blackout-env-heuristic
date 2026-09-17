"""
Drop training segments that start with no battery left anywhere.

Batteries are spawned once per match and never respawn (heuristic_findings_for_reward_20260916.md
§2.1). Once none is left -- on the floor, in any storage, or carried -- no score can change for the
rest of the match: the result is already fixed, and every later transition teaches nothing about
winning. In heuristic_mixv5_hunterphi_20260916 that is 73.6% of all rows (84% of rows after the
first 30 s, 94% after 60 s).

A "segment" is the training episode between `done` flags (absorption cuts and match ends). A
segment is dropped when its first row already has zero batteries in play; batteries cannot come
back, so every later segment of that match is dropped too. The match outcome is not lost: the last
kept row of the match becomes its end (done and terminal), and the outcome reward that sat on the
dropped final row moves onto it.
"""

from __future__ import annotations

import numpy as np

BATTERY_CHANNEL, CARGO_COL = 8, 4
BATTERY_SCALE = 15.0


def batteries_in_play(graphic: np.ndarray, agent_states: np.ndarray) -> np.ndarray:
    """[T] battery points on the map (floor and storages) plus carried."""
    on_map = np.rint(graphic[..., BATTERY_CHANNEL] * BATTERY_SCALE).reshape(len(graphic), -1).sum(1)
    carried = np.rint(np.maximum(agent_states[..., CARGO_COL], 0) * BATTERY_SCALE).sum(1)
    return on_map + carried


def live_rows(batteries: np.ndarray, done: np.ndarray, match_end: np.ndarray) -> np.ndarray:
    """[T] keep mask: rows of segments whose first row still has batteries in play."""
    ends = done | match_end
    starts = np.r_[True, ends[:-1]]
    segment_id = np.cumsum(starts) - 1
    start_rows = np.flatnonzero(starts)
    alive_at_start = batteries[start_rows] > 0
    return alive_at_start[segment_id]


def drop_dead_segments(
    reward: np.ndarray, done: np.ndarray, terminal: np.ndarray, batteries: np.ndarray,
    match_end: np.ndarray, outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    All inputs are [T] over one sequential stream. outcome is the part of each row's reward that is
    the match result (non-zero only on finished match ends).

    Returns full-length (reward, done, terminal, keep): push only rows where keep is True. Each match
    that lost its tail has the outcome moved from its dropped last row onto its new last kept row,
    which is also marked done and terminal.
    """
    keep = live_rows(batteries, done, match_end)
    reward = reward.astype(np.float32).copy()
    done, terminal = done.copy(), terminal.copy()
    match_id = np.cumsum(np.r_[True, match_end[:-1]]) - 1
    boundaries = np.flatnonzero(np.r_[True, match_id[1:] != match_id[:-1], True])
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        last = stop - 1
        kept = np.flatnonzero(keep[first:stop])
        if len(kept) == 0 or first + kept[-1] == last:
            continue
        new_last = first + kept[-1]
        reward[new_last] += outcome[last]
        reward[last] -= outcome[last]
        done[new_last] = terminal[new_last] = True
    return reward, done, terminal, keep
