"""Hand features for the S1 baselines: score, clock and where the battery points are.

Battery points (score units / 100, like team_state): on the floor, banked in own / enemy storage
(deposited, not yet absorbed -- the enemy can still steal them), carried by own / enemy units.
Channel layout: env/my_obs_preprocessor.py (6 = own storage, 7 = enemy storage, 8 = battery
count / 15); agent_states column 4 = carried battery count / 15, rows 0-4 own (canonical frame).
"""

from __future__ import annotations

import numpy as np

STORAGE_OWN, STORAGE_ENEMY, BATTERY = 6, 7, 8
CARGO_COL = 4
SCALE = 15.0
BASE_NAMES = ("lead", "own", "opp", "time_left", "absorb_left", "floor", "store_own", "store_enemy",
              "carry_own", "carry_enemy")
NAMES = BASE_NAMES + ("lead_x_time", "store_diff", "carry_diff", "floor_x_time", "store_diff_x_absorb")


def hand_features(graphic_u8: np.ndarray, team_state: np.ndarray, agent_states: np.ndarray) -> np.ndarray:
    """[N, len(NAMES)] float32.  ``graphic_u8`` is the shard encoding (x15, [N, H, W, C])."""
    g = np.asarray(graphic_u8)
    battery = g[..., BATTERY].astype(np.float32)  # counts (x15 already)
    own_cells = g[..., STORAGE_OWN] > 0
    enemy_cells = g[..., STORAGE_ENEMY] > 0
    n = len(g)
    store_own = (battery * own_cells).reshape(n, -1).sum(1) / 100.0
    store_enemy = (battery * enemy_cells).reshape(n, -1).sum(1) / 100.0
    floor = (battery * ~(own_cells | enemy_cells)).reshape(n, -1).sum(1) / 100.0
    cargo = np.maximum(np.asarray(agent_states)[..., CARGO_COL], 0) * SCALE / 100.0
    carry_own, carry_enemy = cargo[:, :5].sum(1), cargo[:, 5:].sum(1)
    ts = np.asarray(team_state, np.float32)
    lead = ts[:, 0] - ts[:, 1]
    cols = [lead, ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3], floor, store_own, store_enemy, carry_own, carry_enemy,
            lead * ts[:, 2], store_own - store_enemy, carry_own - carry_enemy, floor * ts[:, 2],
            (store_own - store_enemy) * ts[:, 3]]
    return np.stack(cols, axis=1).astype(np.float32)
