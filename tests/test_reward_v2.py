"""Reward v2 potentials: geometry, continuity at pickup/absorption, zero-sum, and kill credit."""

from __future__ import annotations

import numpy as np
import pytest

from blackout_env.train.reward_v2 import (
    BATTERY, BATTERY_SCALE, CARGO_COL, CARRIER, COLLECTOR, HUNTER, SPAWN_ENEMY, SPAWN_OWN,
    STORAGE_ENEMY, STORAGE_OWN, TARGET_SCORE, WALL, Potentials, RewardV2Config,
    all_pairs_distance, base_mask, compute_potentials, geometry_for, score_reward, shaped_rewards,
)

H = W = 24


def _graphic() -> np.ndarray:
    g = np.zeros((H, W, 13), dtype=np.float32)
    g[0, :, WALL] = g[-1, :, WALL] = g[:, 0, WALL] = g[:, -1, WALL] = 1.0
    g[1, 1, SPAWN_OWN] = 1.0
    g[22, 22, SPAWN_ENEMY] = 1.0
    g[10:12, 3:5, STORAGE_OWN] = 1.0      # own external storage
    g[3:5, 10:12, STORAGE_ENEMY] = 1.0    # enemy external storage (mirror)
    return g


def _states(units: list[tuple[float, float, int, float]]) -> np.ndarray:
    """10 rows of (row, col, class, cargo points); rows 0-4 own team."""
    s = np.zeros((10, 12), dtype=np.float32)
    for i, (row, col, cls, cargo) in enumerate(units):
        s[i, 0] = (col + 0.5) * 2.0 / W - 1.0
        s[i, 1] = 1.0 - (row + 0.5) * 2.0 / H
        s[i, 2] = 1.0 if i < 5 else -1.0
        s[i, 3] = 0.0 if cargo else 1.0
        s[i, CARGO_COL] = cargo / BATTERY_SCALE
        s[i, 9 + cls] = 1.0
    return s


FAR_OWN = [(2.0, 2.0, COLLECTOR, 0)] * 5
FAR_ENEMY = [(21.0, 21.0, COLLECTOR, 0)] * 5


def _pot(graphic, states, team_state=None) -> Potentials:
    ts = np.array([[0.0, 0.0, 1.0, 0.5]], dtype=np.float32) if team_state is None else team_state[None]
    return compute_potentials(graphic[None], states[None], ts)


def _slice(p: Potentials, i: int) -> Potentials:
    return Potentials(p.unit[i : i + 1], p.common[i : i + 1], p.confirmed[i : i + 1], p.hunter_of[i : i + 1], p.own[i : i + 1])


# ------------------------------------------------------------------ geometry


def test_bases_block_only_the_enemy():
    geo = geometry_for(_graphic())
    assert geo.base[0][1, 1] and geo.base[0][4, 4] and not geo.base[0][5, 5]
    assert geo.walkable[0][2, 2] and not geo.walkable[1][2, 2]   # own base: own walks, enemy cannot
    assert not geo.walkable[0][21, 21] and geo.walkable[1][21, 21]


def test_all_pairs_distance_diagonal_and_no_corner_cutting():
    walk = np.ones((H, W), dtype=bool)
    walk[5, 6] = False
    d = all_pairs_distance(walk)
    assert d[0, 1] == pytest.approx(1.0)
    assert d[0, W + 1] == pytest.approx(np.sqrt(2))
    # (5,5) -> (4,6): the diagonal's flank (5,6) is blocked, so it goes round via (4,5)
    assert d[5 * W + 5, 4 * W + 6] == pytest.approx(2.0)
    assert d[5 * W + 5, 4 * W + 4] == pytest.approx(np.sqrt(2))  # both flanks open


# ------------------------------------------------------------------ continuity


def test_pickup_is_worth_nothing_approach_is():
    g = _graphic()
    g[10, 8, BATTERY] = 5 / BATTERY_SCALE
    far = _pot(g, _states([(10.0, 14.0, COLLECTOR, 0)] + FAR_OWN[1:] + FAR_ENEMY))
    near = _pot(g, _states([(10.0, 8.0, COLLECTOR, 0)] + FAR_OWN[1:] + FAR_ENEMY))
    g_after = g.copy(); g_after[10, 8, BATTERY] = 0
    carried = _pot(g_after, _states([(10.0, 8.0, COLLECTOR, 5)] + FAR_OWN[1:] + FAR_ENEMY))
    assert near.terms["fetch"][0, 0] > far.terms["fetch"][0, 0] > 0
    assert carried.terms["carry"][0, 0] == pytest.approx(near.terms["fetch"][0, 0], rel=1e-3)


def test_carrying_toward_storage_raises_value_and_a_full_storage_does_not_count():
    g = _graphic()
    near = _pot(g, _states([(10.0, 6.0, COLLECTOR, 5)] + FAR_OWN[1:] + FAR_ENEMY)).terms["carry"][0, 0]
    far = _pot(g, _states([(10.0, 16.0, COLLECTOR, 5)] + FAR_OWN[1:] + FAR_ENEMY)).terms["carry"][0, 0]
    assert near > far
    full = g.copy()
    full[10:12, 3:5, BATTERY] = 9 / BATTERY_SCALE  # 4 tiles x 1 free = 4 < 5 cargo
    blocked = _pot(full, _states([(10.0, 6.0, COLLECTOR, 5)] + FAR_OWN[1:] + FAR_ENEMY)).terms["carry"][0, 0]
    assert blocked < near  # nearest storage refuses it: delivery is the base storage... or nothing


def test_absorption_moves_stored_value_into_confirmed_score_without_a_jump():
    g = _graphic()
    g[10, 3, BATTERY] = 8 / BATTERY_SCALE
    before = _pot(g, _states(FAR_OWN + FAR_ENEMY), np.array([0.08, 0.0, 0.9, 0.0001], dtype=np.float32))
    g2 = g.copy(); g2[10, 3, BATTERY] = 0
    after = _pot(g2, _states(FAR_OWN + FAR_ENEMY), np.array([0.08, 0.0, 0.9, 1.0], dtype=np.float32))
    team = shaped_rewards(before, after, 1.0)[1] + score_reward(before, after)
    assert before.confirmed[0, 0] == pytest.approx(0.0) and after.confirmed[0, 0] == pytest.approx(8.0)
    assert abs(float(team[0])) < 0.05


# ------------------------------------------------------------------ zero-sum and credit


def _enemy_view(graphic: np.ndarray, states: np.ndarray, team_state: np.ndarray):
    g = graphic.copy()
    g[..., [SPAWN_OWN, SPAWN_ENEMY]] = graphic[..., [SPAWN_ENEMY, SPAWN_OWN]]
    g[..., [STORAGE_OWN, STORAGE_ENEMY]] = graphic[..., [STORAGE_ENEMY, STORAGE_OWN]]
    s = states.copy(); s[:, 2] *= -1
    t = team_state.copy(); t[[0, 1]] = team_state[[1, 0]]
    return g, s, t


def test_rewards_are_zero_sum_across_perspectives():
    rng = np.random.default_rng(0)
    g0, g1 = _graphic(), _graphic()
    g0[12, 12, BATTERY] = 4 / BATTERY_SCALE
    g1[12, 12, BATTERY] = 4 / BATTERY_SCALE
    ts = np.array([0.1, 0.05, 0.9, 0.5], dtype=np.float32)
    units0 = [(float(r), float(c), int(k), 0) for r, c, k in zip(rng.uniform(5, 18, 10), rng.uniform(5, 18, 10), rng.integers(0, 3, 10))]
    units1 = [(r + 0.2, c - 0.2, k, x) for r, c, k, x in units0]
    s0, s1 = _states(units0), _states(units1)
    own = shaped_rewards(_pot(g0, s0, ts), _pot(g1, s1, ts), 0.997)[1]
    e0, e1 = _enemy_view(g0, s0, ts), _enemy_view(g1, s1, ts)
    enemy = shaped_rewards(_pot(*e0), _pot(*e1), 0.997)[1]
    assert float(own[0]) == pytest.approx(-float(enemy[0]), abs=1e-3)


def test_killing_a_carrier_credits_the_hunter_assigned_to_it():
    g = _graphic()
    hunter = (12.0, 12.0, HUNTER, 0)
    enemy_carrier = (12.0, 13.0, COLLECTOR, 6)
    before = _pot(g, _states([hunter] + FAR_OWN[1:] + [enemy_carrier] + FAR_ENEMY[1:]))
    assert before.hunter_of[0, 5] == 0
    respawned = (21.0, 21.0, COLLECTOR, 0)
    after = _pot(g, _states([(12.0, 13.0, HUNTER, 0)] + FAR_OWN[1:] + [respawned] + FAR_ENEMY[1:]))
    per_unit, team = shaped_rewards(before, after, 1.0)
    assert float(per_unit[0, 0]) > 0.5 * float(per_unit[0, 1:5].max(initial=0)) + 1.0
    assert float(team[0]) > 0
    assert float(per_unit[0].sum()) == pytest.approx(float(team[0]), abs=1e-4)


def test_an_empty_field_leaves_no_positional_value():
    p = _pot(_graphic(), _states([(12.0, 12.0, HUNTER, 0)] * 5 + FAR_ENEMY))
    assert np.allclose(p.terms["ready"], 0) and np.allclose(p.terms["hunt_exit"], 0)
