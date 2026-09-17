"""Dropping segments that start with no battery left, without losing the match outcome."""

from __future__ import annotations

import numpy as np

from blackout_env.train.dead_segments import batteries_in_play, drop_dead_segments, live_rows


def test_batteries_count_floor_storage_and_cargo():
    g = np.zeros((2, 24, 24, 13), dtype=np.float32)
    s = np.zeros((2, 10, 12), dtype=np.float32)
    g[0, 3, 3, 8] = 4 / 15
    s[1, 7, 4] = 2 / 15
    assert batteries_in_play(g, s).tolist() == [4.0, 2.0]


def test_segments_that_start_dead_are_dropped_and_partial_ones_kept():
    #            seg0 (live)      seg1 (runs out inside) seg2 (starts dead)   next match
    batteries = np.array([5, 5, 5, 3, 0, 0, 0, 0, 0, 9, 9], dtype=float)
    done = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    match_end = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=bool)
    keep = live_rows(batteries, done, match_end)
    assert keep.tolist() == [True] * 6 + [False] * 3 + [True] * 2


def test_the_outcome_moves_to_the_new_last_row():
    batteries = np.array([5, 5, 0, 0, 7, 7], dtype=float)
    done = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    match_end = np.array([0, 0, 0, 1, 0, 1], dtype=bool)
    outcome = np.array([0, 0, 0, 5.0, 0, -5.0])
    reward = np.array([0.1, 0.2, 0.0, 5.0, 0.3, -5.0])
    new_reward, new_done, terminal, keep = drop_dead_segments(reward, done, match_end.copy(), batteries, match_end, outcome)
    assert keep.tolist() == [True, True, False, False, True, True]
    assert new_reward[1] == np.float32(5.2) and new_done[1] and terminal[1]
    assert new_reward[5] == np.float32(-5.0) and terminal[5]  # untouched: its last row was kept
    assert abs(float(new_reward[keep].sum()) - (0.1 + 5.2 + 0.3 - 5.0)) < 1e-5
