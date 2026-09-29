import numpy as np

from blackout_env.heuristics import StrategicHeuristicV17, StrategicHeuristicV17Teach


def _observation(classes=(0, 0, 0, 0, 0), batteries=((18, 6, 8), (6, 18, 9), (12, 3, 10))):
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[2:4, 5:7, 3] = 1       # carrier sanctuary
    graphic[11:13, 11:13, 2] = 1   # hunter sanctuary
    graphic[1:4, 1:4, 6] = 1       # own storage
    graphic[20:23, 20:23, 7] = 1   # enemy storage
    graphic[0, 0, 4] = 1           # own spawn
    graphic[23, 23, 5] = 1         # enemy spawn
    for y, x, amount in batteries:
        graphic[y, x, 8] = amount / 15
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        states[i, :2] = (-0.5 + 0.1 * i, 0.4 - 0.1 * i) if i < 5 else (0.6, -0.6 + 0.05 * i)
        states[i, 2] = 1 if i < 5 else -1
        states[i, 3] = 1
        states[i, 9 + (classes[i] if i < 5 else 0)] = 1
    return {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}


def _team(o):
    return {f"unit_{i}": o for i in range(5)}


def test_labels_are_a_function_of_the_observation():
    o = _observation()
    teacher = StrategicHeuristicV17Teach()
    first = teacher.label_masks(_team(o))
    teacher.label_masks(_team(_observation(classes=(1, 1, 0, 2, 0))))
    again = teacher.label_masks(_team(o))
    fresh = StrategicHeuristicV17Teach().label_masks(_team(o))
    for name in first:
        assert np.array_equal(first[name], again[name])
        assert np.array_equal(first[name], fresh[name])


def test_labels_lead_to_v17_role_ratio():
    o = _observation()
    teacher = StrategicHeuristicV17Teach()
    labels = teacher.teach(_team(o))
    assert teacher.role_counts(labels, _team(o)) == {"collector": 1, "hunter": 3, "carrier": 1}
    assert sum(label.kind == "transform" for label in labels.values()) == 4
    # plain V17 plans the same composition in this state
    v17 = StrategicHeuristicV17()
    v17.act(_team(o))
    assert sorted(v17.role_assignments.values()) == ["carrier", "collector", "hunter", "hunter", "hunter"]


def test_composition_counts_current_classes_and_skips_carrier_on_an_empty_field():
    o = _observation(classes=(1, 1, 0, 0, 0))
    teacher = StrategicHeuristicV17Teach()
    labels = teacher.teach(_team(o))
    assert teacher.role_counts(labels, _team(o)) == {"collector": 1, "hunter": 3, "carrier": 1}
    poor = _observation(batteries=((18, 6, 8),))  # 8 points on the field < 25: no new Carrier
    labels = StrategicHeuristicV17Teach().teach(_team(poor))
    assert StrategicHeuristicV17Teach().role_counts(labels, _team(poor)) == {"collector": 2, "hunter": 3, "carrier": 0}


def test_camper_is_the_hunter_nearest_the_enemy_base_not_the_lowest_slot():
    for hunters in ((0, 3, 4), (1, 2, 4)):
        classes = tuple(1 if i in hunters else 0 for i in range(5))
        teacher = StrategicHeuristicV17Teach()
        labels = teacher.teach(_team(_observation(classes=classes)))
        assert teacher.campers == [f"unit_{max(hunters)}"]  # higher index sits closer to (23, 23) here
        assert labels[teacher.campers[0]].kind in {"patrol_defend", "hunt"}


def test_following_the_labels_does_not_crash_and_holds_the_team_plan():
    o = _observation()
    teacher = StrategicHeuristicV17Teach()
    actions = teacher.act(_team(o))
    assert set(actions) == {f"unit_{i}" for i in range(5)}
    for vector in actions.values():
        assert vector.shape == (2,) and np.isfinite(vector).all()
