import numpy as np

from blackout_env.heuristics import StrategicHeuristic


def _observation(team_b=False):
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[10, :, 0] = 0
    graphic[10, :, 1] = 1
    graphic[10, 12, 1] = 0
    graphic[10, 12, 0] = 1
    graphic[2:4, 2:4, 3] = 1
    graphic[11:13, 11:13, 2] = 1
    graphic[1:4, 1:4, 6] = 1
    graphic[20:23, 20:23, 7] = 1
    graphic[18, 18, 8] = 8 / 15
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        states[i, :2] = (-0.75 + 0.04 * i, -0.75 + 0.03 * i)
        own = i >= 5 if team_b else i < 5
        states[i, 2] = 1 if own else -1
        states[i, 3] = 1  # holding-none slot
        states[i, 9] = 1  # Collector
    return {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}


def test_actions_are_finite_normalized_and_cover_team_a():
    policy = StrategicHeuristic()
    o = _observation()
    actions = policy.act({f"unit_{i}": o for i in range(5)})
    assert set(actions) == {f"unit_{i}" for i in range(5)}
    for action in actions.values():
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.linalg.norm(action) <= 1.00001


def test_team_b_uses_absolute_rows_without_mirroring():
    policy = StrategicHeuristic(use_specialists=False)
    o = _observation(team_b=True)
    actions = policy.act({f"unit_{i}": o for i in range(5, 10)})
    assert set(actions) == {f"unit_{i}" for i in range(5, 10)}
    assert all(np.isfinite(a).all() for a in actions.values())
    # World y=-0.75 maps near the bottom (large top-down graphic row), not row 2.
    assert policy._to_pixel(np.array([-0.75, -0.75]), (24, 24)) == (20, 2)


def test_astar_uses_gap_and_does_not_cut_wall_corners():
    walkable = np.ones((9, 9), dtype=bool)
    walkable[4, :] = False
    walkable[4, 6] = True
    path = StrategicHeuristic._astar(walkable, (1, 1), (7, 1))
    assert path
    assert (4, 6) in path
    assert all(walkable[p] for p in path)


def test_full_nearest_storage_is_skipped_and_all_full_stages_outside():
    graphic = np.zeros((12, 12, 13), dtype=np.float32)
    graphic[1:3, 1:3, 6] = 1
    graphic[8:10, 8:10, 6] = 1
    graphic[1:3, 1:3, 8] = 10 / 15  # nearest component has no battery capacity
    state = np.zeros(12, dtype=np.float32)
    state[4] = 8 / 15  # carrying an 8-point battery

    target, can_deposit = StrategicHeuristic._storage_target(graphic, state, (0, 0))
    assert can_deposit
    assert target[0] >= 8 and target[1] >= 8

    graphic[8:10, 8:10, 8] = 10 / 15
    target, can_deposit = StrategicHeuristic._storage_target(graphic, state, (0, 0))
    assert not can_deposit
    assert graphic[target[0], target[1], 6] == 0  # waits outside instead of oscillating inside


def test_empty_collector_ignores_battery_already_in_own_storage():
    policy = StrategicHeuristic(use_specialists=False)
    graphic = np.zeros((12, 12, 13), dtype=np.float32)
    graphic[1:3, 1:3, 6] = 1
    graphic[1, 1, 8] = 10 / 15  # high-value but already deposited by our team
    graphic[9, 9, 8] = 2 / 15   # lower-value field battery is the valid target
    state = np.zeros(12, dtype=np.float32)
    state[:2] = [-0.8, 0.8]
    state[3] = 1
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    states[0, 2] = 1
    states[1:, 2] = -1
    target, kind = policy._choose_target(
        "collector", states[0], states, graphic, set(), 0,
        np.array([0, 0, 1, 1], dtype=np.float32),
    )
    assert target == (9, 9)
    assert kind == "battery"


def test_enemy_spawn_storage_is_excluded_from_raids():
    graphic = np.zeros((14, 14, 13), dtype=np.float32)
    graphic[1, 1, 5] = 1
    graphic[1:3, 2:4, 7] = 1   # protected home storage
    graphic[9:11, 9:11, 7] = 1  # external storage
    protected = StrategicHeuristic._protected_enemy_storage_mask(graphic)
    assert protected[1:3, 2:4].all()
    assert not protected[9:11, 9:11].any()


def test_late_game_patrol_rotates_between_storage_components():
    policy = StrategicHeuristic()
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:3, 1:3] = True
    mask[8:10, 8:10] = True
    first = policy._patrol_component_center(mask, 0, period=10)
    policy._tick = 10
    second = policy._patrol_component_center(mask, 0, period=10)
    assert first != second


def test_unreachable_astar_target_does_not_drive_straight_into_wall():
    policy = StrategicHeuristic()
    walkable = np.ones((5, 5), dtype=bool)
    walkable[:, 2] = False  # separates start from goal
    state = np.zeros(12, dtype=np.float32)
    state[:2] = [-0.4, 0.0]  # exact center of graphic row=2, col=1
    state[2] = 1
    state[3] = 1
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    action = policy._navigate("unit_2", state, states, (2, 3), "battery", walkable, (5, 5))
    # Goal is directly right, but that direction is disconnected by the wall column.
    assert action[0] <= 0


def test_failed_interaction_arrival_retries_instead_of_idling_forever():
    policy = StrategicHeuristic()
    walkable = np.ones((5, 5), dtype=bool)
    state = np.zeros(12, dtype=np.float32)
    state[:2] = [0.0, 0.0]  # center of row=2, col=2
    state[2] = 1
    state[3] = 1
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    actions = [
        policy._navigate("unit_0", state, states, (2, 2), "transform", walkable, (5, 5))
        for _ in range(24)
    ]
    assert all(np.linalg.norm(a) == 0 for a in actions[:-1])
    assert np.linalg.norm(actions[-1]) > 0.9
