import numpy as np

from blackout_env.heuristics import StrategicHeuristicV4, StrategicHeuristicV4Teach
from blackout_env.heuristics.teach import DIRECTIONS, MOVES
from blackout_env.model.my_policy import DIRECTION_VECTORS


def _observation(classes=(0, 0, 0, 0, 0)):
    graphic = np.zeros((24, 24, 13), dtype=np.float32)
    graphic[..., 0] = 1
    graphic[2:4, 2:4, 3] = 1       # carrier sanctuary
    graphic[11:13, 11:13, 2] = 1   # hunter sanctuary
    graphic[1:4, 1:4, 6] = 1       # own storage
    graphic[20:23, 20:23, 7] = 1   # enemy storage
    graphic[0, 0, 4] = 1           # own spawn
    graphic[23, 23, 5] = 1         # enemy spawn
    graphic[18, 6, 8] = 8 / 15     # one field battery
    states = np.zeros((10, 12), dtype=np.float32)
    for i in range(10):
        states[i, :2] = (-0.2 + 0.05 * i, 0.2 - 0.05 * i)
        states[i, 2] = 1 if i < 5 else -1
        states[i, 3] = 1
        states[i, 9 + (classes[i] if i < 5 else 0)] = 1
    return {"graphic": graphic, "team_state": np.array([0, 0, 1, 1], np.float32), "agent_states": states}


def _team(o):
    return {f"unit_{i}": o for i in range(5)}


def test_compass_moves_match_direction_vectors():
    for (dr, dc), vector in zip(MOVES, DIRECTION_VECTORS):
        assert np.allclose(np.array([dc, -dr], np.float32) / np.hypot(dr, dc), vector, atol=1e-6)
    assert np.allclose(DIRECTIONS, DIRECTION_VECTORS)


def test_labels_are_a_function_of_the_observation():
    o = _observation()
    teacher = StrategicHeuristicV4Teach()
    first = teacher.label_masks(_team(o))
    other = _observation(classes=(1, 0, 0, 2, 0))
    teacher.label_masks(_team(other))  # a different state in between must not leak in
    again = teacher.label_masks(_team(o))
    fresh = StrategicHeuristicV4Teach().label_masks(_team(o))
    for name in first:
        assert np.array_equal(first[name], again[name])
        assert np.array_equal(first[name], fresh[name])


def test_class_role_never_labels_a_transformation():
    o = _observation()
    labels = StrategicHeuristicV4Teach().teach(_team(o))
    assert all(label.kind != "transform" for label in labels.values())
    # plain V4 sends slot 0 and 1 Collectors to their sanctuaries in the same state
    v4 = StrategicHeuristicV4()
    v4.act(_team(o))
    assert {v4._memory[f"unit_{i}"].target_kind for i in (0, 1)} == {"transform"}


def test_close_races_are_shared_and_straight_diagonal_ties():
    o = _observation()
    teacher = StrategicHeuristicV4Teach()
    labels = teacher.teach(_team(o))
    # one battery, units within a second of each other: the race is left alone, all head for it
    assert all(label.valid and label.target == (18, 6) for label in labels.values())
    o["graphic"][18, 6, 8] = 0
    o["graphic"][18, 9, 8] = 8 / 15  # off the diagonal of unit 4's tile: "south" and "south-west" tie
    label = StrategicHeuristicV4Teach().teach(_team(o))["unit_4"]
    south, south_west = 6, 5
    assert label.mask[south] and label.mask[south_west] and label.mask.sum() == 2


def test_target_tile_finishes_the_approach_then_goes_invalid_not_east():
    o = _observation()
    teacher = StrategicHeuristicV4Teach()
    walkable = o["graphic"][..., 1] < 0.5
    state = o["agent_states"][0].copy()
    tile = teacher._to_pixel(state[:2], (24, 24))
    # tile centre (row r, col c) is x = (c + 0.5) / 12 - 1, y = 1 - (r + 0.5) / 12
    centre = np.array([(tile[1] + 0.5) / 12 - 1, 1 - (tile[0] + 0.5) / 12], np.float32)
    state[:2] = centre + np.array([0.3 / 12, 0.0], np.float32)  # 0.3 tiles east of centre
    goals = (((tuple(tile),), "battery"),)
    label = teacher._label(state, o["agent_states"], goals, walkable, (24, 24))
    assert label.valid and label.mask.sum() == 1
    assert np.argmax(label.mask) == 4  # compass index 4 = west, back toward the centre
    state[:2] = centre
    label = teacher._label(state, o["agent_states"], goals, walkable, (24, 24))
    assert not label.valid and not label.mask.any()


def _two_batteries(unit_col):
    """Two equal batteries left and right of one Collector (others parked far away as Hunters)."""
    o = _observation(classes=(0, 1, 1, 1, 1))
    g = o["graphic"]
    g[..., 8] = 0
    g[12, 4, 8] = g[12, 18, 8] = 5 / 15
    s = o["agent_states"]
    s[0, :2] = ((unit_col + 0.5) / 12 - 1, 1 - (12 + 0.5) / 12)
    for i in range(1, 10):
        s[i, :2] = (0.9, -0.9 + 0.01 * i) if i >= 5 else (-0.9, 0.9 - 0.01 * i)
    return o


def test_following_the_label_keeps_the_target():
    teacher = StrategicHeuristicV4Teach()
    col = 10.0  # slightly nearer the right battery (col 18 is 8 away, col 4 is 6 away -> left)
    targets = []
    for _ in range(12):
        o = _two_batteries(col)
        label = teacher.teach(_team(o))["unit_0"]
        assert label.valid
        targets.append(label.target)
        step = DIRECTIONS[label.mask].mean(axis=0)
        col += 0.16 * float(np.sign(step[0])) * 4  # one tile per call along the label
        if not 1 <= col <= 22:
            break
    assert len(set(targets)) == 1


def test_near_ties_are_unioned_but_opposite_ones_are_no_label():
    o = _two_batteries(11.0)  # exactly midway: the teacher has no preference -> no label, not a coin flip
    label = StrategicHeuristicV4Teach().teach(_team(o))["unit_0"]
    assert len(label.goals) == 2 and not label.valid and not label.mask.any()
    o = _observation(classes=(0, 1, 1, 1, 1))
    g = o["graphic"]
    g[..., 8] = 0
    g[18, 12, 8] = g[17, 14, 8] = 5 / 15  # same side, 6.0 vs 5.83 tiles away: within the value band
    s = o["agent_states"]
    s[0, :2] = ((12 + 0.5) / 12 - 1, 1 - (12 + 0.5) / 12)
    label = StrategicHeuristicV4Teach().teach(_team(o))["unit_0"]
    assert label.valid and len(label.goals) == 2


def test_a_clearly_faster_teammate_takes_a_shared_best_task():
    o = _observation(classes=(0, 0, 1, 1, 1))
    g = o["graphic"]
    g[..., 8] = 0
    g[12, 12, 8] = 8 / 15   # the best battery for both Collectors
    g[20, 2, 8] = 3 / 15    # a worse one, far from both
    s = o["agent_states"]
    s[0, :2] = ((12 + 0.5) / 12 - 1, 1 - (10 + 0.5) / 12)   # 2 tiles from the big battery
    s[1, :2] = ((12 + 0.5) / 12 - 1, 1 - (2 + 0.5) / 12)    # 10 tiles: 2 s later than unit 0
    labels = StrategicHeuristicV4Teach().teach(_team(o))
    assert labels["unit_0"].target == (12, 12)
    assert labels["unit_1"].target == (20, 2)
    s[1, :2] = ((12 + 0.5) / 12 - 1, 1 - (7 + 0.5) / 12)    # 5 tiles: 0.75 s later, inside the margin
    labels = StrategicHeuristicV4Teach().teach(_team(o))
    assert labels["unit_0"].target == labels["unit_1"].target == (12, 12)
