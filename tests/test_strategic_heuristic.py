import numpy as np

from blackout_env.heuristics import (
    HeuristicPolicyMixture, StrategicHeuristic, StrategicHeuristicV2,
    StrategicHeuristicV3,
    StrategicHeuristicV4,
    StrategicHeuristicV5,
    StrategicHeuristicV6,
    V4PolicyFamily,
    StrategicHeuristicV7,
    StrategicHeuristicV8,
    StrategicHeuristicV9,
    StrategicHeuristicV10,
    StrategicHeuristicV11,
    StrategicHeuristicV12,
    StrategicHeuristicV13,
    StrategicHeuristicV14,
    StrategicHeuristicV15,
    StrategicHeuristicV16,
    StrategicHeuristicV17,
    StrategicHeuristicV18,
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


def test_v2_distance_map_continues_after_diagonal_float_rounding():
    walkable = np.ones((20, 20), dtype=bool)
    distances = StrategicHeuristicV2._distance_map(walkable, (18, 1))
    assert np.isfinite(distances[1, 18])


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
        "strategic_v1", "strategic_v2", "strategic_v3", "strategic_v4",
        "strategic_v5",
        "strategic_v6",
        "strategic_v4_near",
        "strategic_v7",
        "strategic_v8",
        "strategic_v9",
        "strategic_v10", "strategic_v11", "strategic_v12",
        "strategic_v13", "strategic_v14", "strategic_v15",
        "strategic_v16",
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


def _absorb(policy, o):
    """Tick once near the end of an absorption window, then once right after it fires."""
    o["team_state"][3] = 0.01
    policy.act({f"unit_{i}": o for i in range(5)})
    o["team_state"][3] = 1.0
    policy.act({f"unit_{i}": o for i in range(5)})


def test_policy_mixture_absorption_retunes_in_place_for_every_policy():
    from blackout_env.heuristics import POLICY_REGISTRY, make_heuristic

    for policy_id in POLICY_REGISTRY:
        mixture = HeuristicPolicyMixture(seed=11, weights={policy_id: 1.0})
        o = _observation()
        mixture.act({f"unit_{i}": o for i in range(5)})
        running = mixture._policy
        before = mixture.current_sample
        _absorb(mixture, o)
        after = mixture.current_sample
        assert mixture._policy is running, policy_id  # episode state survives
        assert after.policy_id == before.policy_id
        assert after.policy_seed != before.policy_seed
        assert after.parameters["use_specialists"] == before.parameters["use_specialists"]
        inner = running._policy if policy_id == "strategic_v4_near" else running
        params = {k: v for k, v in after.parameters.items() if k != "profile"}
        fresh = type(inner)(**params)
        for name in params:
            assert getattr(inner, name) == getattr(fresh, name), (policy_id, name)


def test_policy_mixture_keeps_respec_cooldown_across_absorption():
    mixture = HeuristicPolicyMixture(seed=3, weights={"strategic_v8": 1.0})
    o = _observation()
    mixture.act({f"unit_{i}": o for i in range(5)})
    mixture._policy._hunter_cooldown = 500
    _absorb(mixture, o)
    assert mixture._policy._hunter_cooldown >= 498


def test_v4_family_absorption_resample_keeps_running_policy():
    family = V4PolicyFamily(seed=4, resample_each_absorption=True)
    o = _observation()
    family.act({f"unit_{i}": o for i in range(5)})
    running = family._policy
    before = family.current_sample
    _absorb(family, o)
    assert family._policy is running
    assert family.current_sample.policy_seed != before.policy_seed
    assert running.replan_interval == family.current_sample.parameters["replan_interval"]
    o["team_state"][2] = 0.2
    family.act({f"unit_{i}": o for i in range(5)})
    o["team_state"][2] = 1.0
    family.act({f"unit_{i}": o for i in range(5)})
    assert family._policy is not running  # a real match reset still starts clean


def test_v4_family_exact_profile_matches_v4_and_is_reproducible():
    first = V4PolicyFamily(seed=99, profile_weights={"exact": 1.0})
    second = V4PolicyFamily(seed=99, profile_weights={"exact": 1.0})
    assert first.current_sample == second.current_sample
    assert first.current_sample.profile == "exact"
    o = _observation()
    family_actions = first.act({f"unit_{i}": o for i in range(5)})
    v4_actions = StrategicHeuristicV4().act({f"unit_{i}": o for i in range(5)})
    for name in family_actions:
        np.testing.assert_allclose(family_actions[name], v4_actions[name])


def test_general_mixture_can_force_nested_near_v4_family():
    mixture = HeuristicPolicyMixture(
        seed=5, weights={"strategic_v4_near": 1.0}, perturb=True
    )
    assert mixture.current_sample.policy_id == "strategic_v4_near"
    assert mixture.current_sample.parameters["profile"] in {
        "exact", "balanced", "responsive", "cautious"
    }
    assert 8 <= mixture.current_sample.parameters["replan_interval"] <= 12


def test_phase_director_switches_to_closeout_and_labels_roles():
    policy = StrategicHeuristicV10(mode_confirm_ticks=1)
    o = _observation()
    # A durable late lead turns the policy into a lead-preserving defender.  The first
    # observation starts hysteresis and the second confirms it.
    o["team_state"][:] = [0.40, 0.20, 0.40, 0.8]
    team = {f"unit_{i}": o for i in range(5)}
    policy.act(team)
    policy.act(team)
    assert policy.current_mode == "closeout_defend"
    assert policy.strategy_transitions
    assert all(role.startswith("closeout_defend:") for role in policy.role_assignments.values())


def test_raid_window_switches_only_when_visible_external_storage_has_value():
    policy = StrategicHeuristicV11(mode_confirm_ticks=1)
    o = _observation()
    o["graphic"][16:18, 16:18, 7] = 1  # external enemy storage, distant from enemy spawn
    o["graphic"][16, 16, 8] = 6 / 15
    o["team_state"][:] = [0.10, 0.30, 0.70, 0.20]
    team = {f"unit_{i}": o for i in range(5)}
    policy.act(team)
    policy.act(team)
    assert policy.current_mode == "raid_window"
    assert policy.strategy_transitions


def test_storage_siege_stays_on_exposed_enemy_storage_without_absorption_window():
    policy = StrategicHeuristicV13(mode_confirm_ticks=1, siege_min_value=1.0)
    o = _observation()
    o["graphic"][16:18, 16:18, 7] = 1  # external enemy storage
    o["graphic"][16, 16, 8] = 4 / 15
    # This is outside V11's ordinary raid window; V13 should still commit.
    o["team_state"][:] = [0.10, 0.10, 0.80, 0.99]
    team = {f"unit_{i}": o for i in range(5)}
    policy.act(team)
    policy.act(team)
    assert policy.current_mode == "raid_window"
    assert policy.style_label == "storage_siege"


def test_counter_raid_and_convoy_policies_keep_distinct_irreversible_roles():
    sentinel = StrategicHeuristicV14()
    convoy = StrategicHeuristicV15()
    o = _observation()
    team = {f"unit_{i}": o for i in range(5)}
    sentinel.act(team)
    convoy.act(team)
    assert sentinel.hunter_quota == 1 and sentinel.carrier_quota == 0
    assert convoy.hunter_quota == 0 and convoy.carrier_quota == 1
    assert sentinel.current_mode == "home_guard"
    assert convoy.current_mode == "convoy_rush"
    assert all(role.startswith("home_guard:") for role in sentinel.role_assignments.values())
    assert all(role.startswith("convoy_rush:") for role in convoy.role_assignments.values())


def test_v16_director_enters_counterplay_siege_mode_from_public_storage_value():
    policy = StrategicHeuristicV16(mode_confirm_ticks=1, siege_value=1.0)
    o = _observation()
    o["graphic"][16:18, 16:18, 7] = 1  # exposed enemy storage
    o["graphic"][16, 16, 8] = 4 / 15
    team = {f"unit_{i}": o for i in range(5)}
    policy.act(team)
    policy.act(team)
    assert policy.current_mode == "storage_siege"
    assert all(role.startswith("storage_siege:") for role in policy.role_assignments.values())
    # The mode must also pass through its specialised multi-unit task assignment, not merely
    # label the roles.  This is the path exercised in the real Unity tournament.
    assignments = policy._assign_economic_tasks(
        ["unit_1", "unit_2", "unit_3", "unit_4"],
        {f"unit_{i}": i for i in range(1, 5)},
        o["agent_states"], o["graphic"], o["team_state"], o["graphic"][..., 0] > 0.5,
    )
    assert assignments
    assert all(kind == "steal" for _, kind in assignments.values())


def test_fortress_mode_defends_home_instead_of_chasing_empty_enemy():
    policy = StrategicHeuristicV12(mode_confirm_ticks=1)
    o = _observation()
    o["team_state"][:] = [0.40, 0.20, 0.50, 0.8]
    team = {f"unit_{i}": o for i in range(5)}
    policy.act(team)
    policy.act(team)
    assert policy.current_mode == "fortress"
    # A Hunter with no cargo threat returns a home-storage patrol target.
    hunter = o["agent_states"][1].copy()
    hunter[9:12] = [0, 1, 0]
    target, kind = policy._choose_target(
        "hunter", hunter, o["agent_states"], o["graphic"], set(), 1, o["team_state"]
    )
    assert kind == "patrol_defend"
    assert target is not None


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
    target, can_deposit = StrategicHeuristicV3()._risk_aware_storage_target(
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


def test_v5_hunter_targets_feasible_point_ahead_of_cargo():
    def world(row, col, size=16):
        return np.array([(col + 0.5) * 2 / size - 1,
                         1 - (row + 0.5) * 2 / size], np.float32)

    policy = StrategicHeuristicV5(use_specialists=False)
    graphic = np.zeros((16, 16, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[8, 14, 7] = 1
    states = np.zeros((10, 12), dtype=np.float32)
    states[:, 3] = 1
    states[:, 9] = 1
    states[0, :2] = world(6, 10)
    states[0, 2] = 1
    states[0, 9] = 0
    states[0, 10] = 1  # Hunter
    states[5, :2] = world(8, 8)
    states[5, 2] = -1
    states[5, 3] = 0
    states[5, 4] = 6 / 15  # cargo
    states[1:5, 2] = 1
    states[6:, 2] = -1
    target = policy._best_cargo_intercept(states[0], states, graphic)
    assert target is not None
    assert target[1] > 8
    assert target != (8, 8)


def test_v5_enemy_velocity_memory_clears_on_reset():
    policy = StrategicHeuristicV5()
    states = np.zeros((10, 12), dtype=np.float32)
    states[5:, 2] = -1
    policy._update_enemy_motion(states)
    states[5, 0] = 0.01
    policy._update_enemy_motion(states)
    assert policy._enemy_velocities[5][0] > 0
    policy.reset()
    assert policy._enemy_velocities == {}


def test_v6_cargo_path_detours_around_hunter_risk():
    def world(row, col, size=11):
        return np.array([(col + 0.5) * 2 / size - 1,
                         1 - (row + 0.5) * 2 / size], np.float32)

    policy = StrategicHeuristicV6(use_specialists=False, risk_weight=8.0)
    walkable = np.ones((11, 11), dtype=bool)
    state = np.zeros(12, dtype=np.float32)
    state[:2] = world(5, 1)
    state[2] = 1
    state[4] = 5 / 15
    state[9] = 1
    states = np.stack([state] + [np.zeros(12, dtype=np.float32) for _ in range(9)])
    states[1:5, 2] = 1
    states[5:, 2] = -1
    states[5, :2] = world(5, 5)
    states[5, 10] = 1  # enemy Hunter
    action = policy._navigate(
        "unit_0", state, states, (5, 9), "deposit", walkable, (11, 11)
    )
    assert abs(float(action[1])) > 0.2  # leaves the direct horizontal collision corridor


def test_v6_numba_risk_astar_matches_python_paths_exactly():
    from blackout_env.heuristics import _native

    if not _native.NUMBA_AVAILABLE:
        return
    rng = np.random.default_rng(1)
    policy = StrategicHeuristicV6()
    yy, xx = np.indices((24, 24), dtype=np.float32)
    for trial in range(200):
        walkable = rng.random((24, 24)) > rng.uniform(0.05, 0.4)
        risk = np.zeros((24, 24), np.float32)
        for _ in range(rng.integers(0, 6)):
            ey, ex = rng.uniform(0, 24, 2)
            risk += np.exp(-np.sqrt((yy - ey) ** 2 + (xx - ex) ** 2) / 3.0).astype(np.float32)
        if trial % 3 == 0:
            risk[rng.random((24, 24)) < 0.2] = np.float32(0.15)  # the narrow-penalty threshold
        policy.risk_weight = float(rng.uniform(0.5, 4.2))
        policy._active_risk = risk
        start = tuple(int(v) for v in rng.integers(0, 24, 2))
        goal = tuple(int(v) for v in rng.integers(0, 24, 2))
        fast = policy._astar(walkable, start, goal)
        _native.NUMBA_AVAILABLE = False
        try:
            slow = policy._astar(walkable, start, goal)
        finally:
            _native.NUMBA_AVAILABLE = True
        assert fast == slow, (trial, start, goal)


def test_v7_delays_hunter_and_respects_distinct_specialist_quotas():
    policy = StrategicHeuristicV7(
        hunter_activation_time=0.4, hunter_activation_field_battery=0.0
    )
    o = _observation()
    policy.act({f"unit_{i}": o for i in range(5)})
    roles = list(policy.role_assignments.values())
    assert roles.count("carrier") == 1
    assert roles.count("hunter") == 0

    o["agent_states"][5, 3] = 0
    o["agent_states"][5, 4] = 5 / 15
    policy.act({f"unit_{i}": o for i in range(5)})
    roles = list(policy.role_assignments.values())
    assert roles.count("carrier") == 1
    assert roles.count("hunter") == 1


def test_v8_respec_requires_mutual_kill_target_and_enforces_cooldown():
    policy = StrategicHeuristicV8(
        hunter_activation_time=0.0,
        hunter_activation_field_battery=0.0,
        respec_inactivity_ticks=2,
        respec_score_gap=0.05,
        respec_min_field_battery=1.0,
        respec_cooldown_ticks=20,
    )
    o = _observation()
    o["team_state"][:2] = [0.2, 0.4]
    o["agent_states"][1, 9:12] = [0, 1, 0]
    o["agent_states"][5, 9:12] = [0, 1, 0]
    obs = {f"unit_{i}": o for i in range(5)}
    policy.act(obs)
    policy.act(obs)
    assert policy.respec_attempts == 1
    assert "respec" in policy.role_assignments.values()

    o["agent_states"][1, 9:12] = [1, 0, 0]
    policy.act(obs)
    assert policy.respec_completions == 1
    assert "hunter" not in policy.role_assignments.values()
    assert policy._hunter_cooldown > 0


def _unit_state(row, col, *, enemy=False, holding=False, cls=0, size=14):
    state = np.zeros(12, dtype=np.float32)
    state[0] = (col + 0.5) * 2.0 / size - 1.0
    state[1] = 1.0 - (row + 0.5) * 2.0 / size
    state[2] = -1.0 if enemy else 1.0
    state[4 if holding else 3] = 0.4 if holding else 1.0
    state[9 + cls] = 1.0
    return state


def _two_storage_home_graphic():
    graphic = np.zeros((14, 14, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[1, 1, 4] = 1        # own spawn
    graphic[1:3, 2:4, 6] = 1    # own storage inside the protected base
    graphic[9:11, 9:11, 6] = 1  # own storage the enemy can reach
    return graphic


def test_own_spawn_storage_is_protected():
    graphic = _two_storage_home_graphic()
    policy = StrategicHeuristic()
    assert policy._cached_protected_ally_storage_mask(graphic)[1:3, 2:4].all()
    defendable = policy._defendable_storage_mask(graphic)
    assert defendable[9:11, 9:11].all() and not defendable[1:3, 2:4].any()


def test_defend_patrol_skips_protected_storage_even_when_closer():
    graphic = _two_storage_home_graphic()
    target = StrategicHeuristic()._defend_patrol_target(graphic, (2, 5))
    assert 9 <= target[0] <= 10 and 9 <= target[1] <= 10


def test_defend_patrol_falls_back_when_all_storage_is_protected():
    graphic = np.zeros((14, 14, 13), dtype=np.float32)
    graphic[1, 1, 4] = 1
    graphic[1:3, 2:4, 6] = 1
    target = StrategicHeuristic()._defend_patrol_target(graphic, (8, 8))
    assert 1 <= target[0] <= 2 and 2 <= target[1] <= 3


def test_home_guard_ignores_cargo_next_to_protected_storage():
    graphic = _two_storage_home_graphic()
    policy = StrategicHeuristicV14()
    hunter = _unit_state(6, 6, cls=1)
    team_state = np.array([0, 0, 1, 1], dtype=np.float32)

    near_protected = _unit_state(3, 4, enemy=True, holding=True)
    target, kind = policy._choose_target("hunter", hunter, np.stack([hunter, near_protected]), graphic, set(), 0, team_state)
    assert kind == "patrol_defend" and 9 <= target[0] <= 10 and 9 <= target[1] <= 10

    near_defendable = _unit_state(8, 11, enemy=True, holding=True)
    _, kind = policy._choose_target("hunter", hunter, np.stack([hunter, near_defendable]), graphic, set(), 0, team_state)
    assert kind == "hunt"


def _v17_observation():
    """Open 24x24 arena with spawns in opposite corners, as in Prototype.unity."""
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[0, :, 1] = graphic[-1, :, 1] = graphic[:, 0, 1] = graphic[:, -1, 1] = 1
    graphic[1, 1, 4] = 1  # our spawn: base rows/cols 1-4
    graphic[22, 22, 5] = 1  # enemy spawn: base rows/cols 19-22
    graphic[3, 1, 3] = 1  # Carrier sanctuary inside our base
    graphic[11:13, 11:13, 2] = 1  # Hunter sanctuary at the centre
    graphic[1:3, 3, 6] = 1  # protected home storage
    graphic[8:10, 14:16, 6] = 1  # exposed storage
    graphic[20:22, 21, 7] = 1
    graphic[14:16, 8:10, 7] = 1
    for y, x in ((6, 6), (6, 16), (16, 6), (9, 9), (15, 15)):
        graphic[y, x, 8] = 8 / 15
    states = np.zeros((10, 12), dtype=np.float32)

    def world(row, col):
        return ((col + 0.5) * 2 / 24 - 1, 1 - (row + 0.5) * 2 / 24)

    for i in range(10):
        own = i < 5
        states[i, :2] = world(2, 2) if own else world(21, 21)
        states[i, 2] = 1 if own else -1
        states[i, 3] = 1
        states[i, 9] = 1
    return {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}


def test_v17_base_is_the_4x4_corner_square_around_spawn_and_camp_faces_the_centre():
    base = StrategicHeuristicV17._base_mask((22, 22), (24, 24))
    assert base.sum() == 16 and base[19, 19] and base[22, 22] and not base[18, 22]
    policy = StrategicHeuristicV17()
    o = _v17_observation()
    ctx = {"enemy_spawn": (22, 22), "shape": (24, 24), "walkable": o["graphic"][..., 1] < 0.5}
    slots = policy._camp_slots(ctx)
    assert slots[0] == (18, 18)
    assert all(not base[slot] for slot in slots)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2 for a, b in zip(slots, slots[1:]))


def test_v17_commits_one_carrier_and_three_hunters_from_the_opening():
    policy = StrategicHeuristicV17()
    o = _v17_observation()
    actions = policy.act({f"unit_{i}": o for i in range(5)})
    assert all(np.isfinite(a).all() for a in actions.values())
    roles = sorted(policy.role_assignments.values())
    assert roles == ["carrier", "collector", "hunter", "hunter", "hunter"]
    kinds = {kind for _, kind in policy.plan.values()}
    assert "transform" in kinds


def test_v17_camping_hunter_strikes_cargo_near_the_enemy_base_but_ignores_enemy_hunters():
    policy = StrategicHeuristicV17()
    o = _v17_observation()
    states = o["agent_states"]

    def world(row, col):
        return ((col + 0.5) * 2 / 24 - 1, 1 - (row + 0.5) * 2 / 24)

    states[0, :2] = world(18, 18)
    states[0, 9], states[0, 10] = 0, 1  # our Hunter, already on its camp slot
    states[5, :2] = world(16, 17)
    states[5, 3], states[5, 4] = 0, 6 / 15  # enemy Collector bringing 6 points home
    states[6, :2] = world(17, 19)
    states[6, 9], states[6, 10] = 0, 1  # enemy Hunter right next to the slot
    policy.act({f"unit_{i}": o for i in range(5)})
    target, kind = policy.plan["unit_0"]
    assert kind == "hunt" and target == (16, 17)
    # Nothing but an enemy Hunter in reach: hold the slot instead of trading.
    states[5, :2] = world(8, 3)
    policy.act({f"unit_{i}": o for i in range(5)})
    assert policy.plan["unit_0"] == ((18, 18), "patrol_defend")


def test_v18_routes_sanctuary_walkers_around_field_batteries():
    policy = StrategicHeuristicV18()
    o = _v17_observation()
    graphic = o["graphic"]
    graphic[3, 3, 8] = 5 / 15  # a battery on the direct diagonal from (2, 2) toward the centre
    policy._last_graphic = graphic
    walkable = graphic[..., 1] < 0.5
    captured = {}
    original = StrategicHeuristicV17._navigate

    def spy(self, name, state, states, target, kind, walkable, shape):
        captured["walkable"] = walkable
        return original(self, name, state, states, target, kind, walkable, shape)

    StrategicHeuristicV17._navigate = spy
    try:
        policy._navigate("unit_1", o["agent_states"][1], o["agent_states"], (11, 11), "transform",
                         walkable, (24, 24))
        assert not captured["walkable"][3, 3]
        policy._navigate("unit_1", o["agent_states"][1], o["agent_states"], (6, 6), "battery",
                         walkable, (24, 24))
        assert captured["walkable"][3, 3]  # only the race to the sanctuary detours
    finally:
        StrategicHeuristicV17._navigate = original


def test_v18_exit_guard_trades_with_an_enemy_hunter_at_our_exit():
    policy = StrategicHeuristicV18()
    o = _v17_observation()
    states = o["agent_states"]

    def world(row, col):
        return ((col + 0.5) * 2 / 24 - 1, 1 - (row + 0.5) * 2 / 24)

    for row, cell in ((0, (18, 18)), (1, (9, 9))):  # camper (rank 0) and guard (rank 1)
        states[row, :2] = world(*cell)
        states[row, 9], states[row, 10] = 0, 1
    states[5, :2] = world(6, 5)
    states[5, 9], states[5, 10] = 0, 1  # V17's camper arriving at our exit
    policy.act({f"unit_{i}": o for i in range(5)})
    assert policy.plan["unit_0"][0] == (18, 18)
    assert policy.plan["unit_1"] == ((6, 5), "hunt")
    states[5, :2] = world(18, 5)  # far from our exit: the guard goes back to its post
    policy.act({f"unit_{i}": o for i in range(5)})
    assert policy.plan["unit_1"] == ((5, 5), "patrol_defend")
