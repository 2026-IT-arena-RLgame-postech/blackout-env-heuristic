import numpy as np

from blackout_env.heuristics import (
    HeuristicPolicyMixture, StrategicHeuristic, StrategicHeuristicV2,
    StrategicHeuristicV3,
    StrategicHeuristicV4,
)


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


def test_v2_global_assignment_uses_path_distance_not_agent_order():
    policy = StrategicHeuristicV2(use_specialists=False)
    graphic = np.zeros((12, 12, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[2, 2, 8] = 5 / 15
    graphic[9, 9, 8] = 5 / 15
    states = np.zeros((10, 12), dtype=np.float32)
    states[:, 3] = 1
    states[:, 9] = 1
    states[:2, 2] = 1
    states[2:, 2] = -1
    # unit_0 starts next to the bottom-right task and unit_1 next to top-left.
    states[0, :2] = [0.72, -0.72]
    states[1, :2] = [-0.72, 0.72]
    assigned = policy._assign_economic_tasks(
        ["unit_0", "unit_1"], {"unit_0": 0, "unit_1": 1}, states, graphic,
        np.array([0, 0, 1, 1], dtype=np.float32), np.ones((12, 12), dtype=bool),
    )
    assert assigned["unit_0"][0] == (9, 9)
    assert assigned["unit_1"][0] == (2, 2)


def test_v2_skips_steal_that_will_be_absorbed_before_arrival():
    policy = StrategicHeuristicV2(use_specialists=False)
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[1, 1, 7] = 1
    graphic[1, 1, 8] = 8 / 15
    states = np.zeros((10, 12), dtype=np.float32)
    states[:, 3] = 1
    states[:, 9] = 1
    states[0, :2] = [0.8, -0.8]
    states[0, 2] = 1
    states[1:, 2] = -1
    assigned = policy._assign_economic_tasks(
        ["unit_0"], {"unit_0": 0}, states, graphic,
        np.array([0, 0, 1, 0.01], dtype=np.float32), np.ones((24, 24), dtype=bool),
    )
    assert assigned == {}


def test_policy_mixture_is_reproducible_and_exposes_dataset_metadata():
    first = HeuristicPolicyMixture(seed=123)
    second = HeuristicPolicyMixture(seed=123)
    assert first.current_sample == second.current_sample
    assert first.current_sample.policy_id in {
        "strategic_v1", "strategic_v2", "strategic_v3", "strategic_v4"
    }
    assert 8 <= first.current_sample.parameters["replan_interval"] <= 13
    next_first = first.reset()
    next_second = second.reset()
    assert next_first == next_second


def test_policy_mixture_resamples_when_unity_episode_time_resets():
    mixture = HeuristicPolicyMixture(seed=7)
    initial = mixture.current_sample
    o = _observation()
    o["team_state"][2] = 0.2
    mixture.act({"unit_0": o})
    o["team_state"][2] = 1.0
    mixture.act({"unit_0": o})
    assert mixture.current_sample != initial


def test_v3_prefers_safe_home_storage_when_absorption_is_not_imminent():
    graphic = np.zeros((16, 16, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[1, 1, 4] = 1
    graphic[1:3, 2:4, 6] = 1      # protected, farther from unit
    graphic[9:11, 9:11, 6] = 1    # exposed, nearer to unit
    state = np.zeros(12, dtype=np.float32)
    state[2] = 1
    state[4] = 5 / 15
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    states[1:, 2] = -1
    target, can_deposit = StrategicHeuristicV3._risk_aware_storage_target(
        graphic, state, states, (7, 7), np.array([0, 0, 1, 1], np.float32)
    )
    assert can_deposit
    assert target[0] <= 2 and target[1] <= 3


def test_v4_reserves_distinct_storage_entry_tiles_for_simultaneous_deposits():
    policy = StrategicHeuristicV4(use_specialists=False)
    graphic = np.zeros((12, 12, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[8:10, 8:10, 6] = 1
    state = np.zeros(12, dtype=np.float32)
    state[2] = 1
    state[4] = 4 / 15
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    states[1:, 2] = -1
    reservations = set()
    first = policy._choose_target(
        "collector", state, states, graphic, reservations, 0,
        np.array([0, 0, 1, 1], np.float32),
    )[0]
    second = policy._choose_target(
        "collector", state, states, graphic, reservations, 1,
        np.array([0, 0, 1, 1], np.float32),
    )[0]
    assert first != second
    assert first in reservations and second in reservations
