"""Reward v2 in training: shaping inside the n-step return, match ends vs absorption cuts, annotation."""

from __future__ import annotations

import numpy as np
import pytest

from blackout_env.train.returns import compute_n_step_return
from blackout_env.train.reward_v2 import RewardV2Config, annotate_sequence, match_ends

CAP = 16


def _explicit(reward, potential, done, terminal, anchor, n, gamma):
    """Per-step shaped rewards summed by hand."""
    total, discount = 0.0, 1.0
    for k in range(n):
        i = anchor + k
        next_p = 0.0 if terminal[i] else potential[i + 1]
        total += discount * (reward[i] + gamma * next_p - potential[i])
        discount *= gamma
        if done[i]:
            break
    return total


@pytest.mark.parametrize("anchor", [0, 3, 5, 9])
def test_potential_shaping_telescopes_inside_the_n_step_return(anchor):
    rng = np.random.default_rng(anchor)
    reward = rng.normal(size=CAP).astype(np.float32)
    potential = rng.normal(size=CAP).astype(np.float32)
    done = np.zeros(CAP, dtype=bool); terminal = np.zeros(CAP, dtype=bool)
    done[6] = True                       # absorption cut: bootstrap potential kept
    done[11] = terminal[11] = True       # match end: next row is another match
    gamma, n = 0.97, 4
    returns, *_ = compute_n_step_return(reward, done, np.array([anchor]), CAP, n, gamma, potential, terminal)
    assert returns[0] == pytest.approx(_explicit(reward, potential, done, terminal, anchor, n, gamma), abs=1e-5)


def test_without_potentials_the_return_is_unchanged():
    reward = np.arange(CAP, dtype=np.float32)
    done = np.zeros(CAP, dtype=bool)
    plain, *_ = compute_n_step_return(reward, done, np.array([2]), CAP, 3, 0.9)
    assert plain[0] == pytest.approx(2 + 0.9 * 3 + 0.81 * 4)


def test_match_ends_are_resets_and_the_last_row():
    ts = np.zeros((6, 4), dtype=np.float32)
    ts[:, 2] = [0.5, 0.4, 0.3, 1.0, 0.9, 0.8]
    assert match_ends(ts).tolist() == [False, False, True, False, False, True]


def test_annotation_recovers_the_outcome_and_pays_confirmed_score():
    from tests.test_reward_v2 import FAR_ENEMY, FAR_OWN, _graphic, _states

    g = np.stack([_graphic()] * 4)
    s = np.stack([_states(FAR_OWN + FAR_ENEMY)] * 4)
    ts = np.zeros((4, 4), dtype=np.float32)
    ts[:, 2] = [0.9, 0.8, 0.7, 0.6]
    ts[:, 0] = [0.0, 0.0, 0.1, 0.1]  # 10 points confirmed between rows 1 and 2
    stored = np.array([0.1, -0.2, 0.3, 5.2], dtype=np.float32)
    done = np.array([False, False, False, True])
    out = annotate_sequence(g, s, ts, stored, done, RewardV2Config(), points_per_reward=20.0)
    assert out["terminal"].tolist() == [False, False, False, True]
    assert out["reward"][1] == pytest.approx(10.0 / 20.0)
    assert out["reward"][3] == pytest.approx(5.0)
    assert out["reward"][0] == pytest.approx(0.0) and out["reward"][2] == pytest.approx(0.0)
