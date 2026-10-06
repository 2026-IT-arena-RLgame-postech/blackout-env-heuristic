import math

import numpy as np
import pytest

from blackout_env.heuristics import StrategicHeuristicV21, StrategicHeuristicV21AnyAngle
from blackout_env.heuristics import any_angle as aa
from blackout_env.heuristics._native import distance_map_numba
from blackout_env.heuristics.teach import label_action

from .test_teach_heuristic import _observation, _team


def _map():
    walkable = np.ones((24, 24), dtype=bool)
    walkable[0, :] = walkable[-1, :] = walkable[:, 0] = walkable[:, -1] = False
    walkable[5:15, 10] = False       # a long wall to go round
    walkable[18, 3:9] = False        # and a short one
    return walkable


def _dense_min_clearance(walkable, start, end, samples=400):
    t = np.linspace(0.0, 1.0, samples)
    return min(aa.point_clearance(walkable, (start[0] + (end[0] - start[0]) * s, start[1] + (end[1] - start[1]) * s))
               for s in t[1:])


def test_point_clearance():
    walkable = _map()
    assert aa.point_clearance(walkable, (3.0, 3.0)) == pytest.approx(0.5)  # capped: open floor
    assert aa.point_clearance(walkable, (10.0, 9.2)) == pytest.approx(0.3)  # wall square starts at col 9.5
    assert aa.point_clearance(walkable, (14.8, 10.8)) == pytest.approx(math.hypot(0.3, 0.3))  # corner (14.5, 10.5)
    assert aa.point_clearance(walkable, (1.0, 1.0)) == pytest.approx(0.5)  # map edge counts as blocked


@pytest.mark.parametrize("radius", sorted(aa.UNIT_RADIUS.values()))
def test_waypoints_keep_clearance_and_headings_are_unit(radius):
    walkable = _map()
    rng = np.random.default_rng(0)
    cells = np.argwhere(walkable)
    checked = 0
    for _ in range(60):
        goal = tuple(int(v) for v in cells[rng.integers(len(cells))])
        start_cell = cells[rng.integers(len(cells))]
        start = (float(start_cell[0]) + rng.uniform(-0.15, 0.15), float(start_cell[1]) + rng.uniform(-0.15, 0.15))
        distance = distance_map_numba(walkable, goal[0], goal[1])
        waypoint = aa.any_angle_waypoint(walkable, distance, start, radius)
        if waypoint is None:
            continue
        required = min(radius + aa.CLEARANCE_MARGIN, aa.point_clearance(walkable, start))
        assert _dense_min_clearance(walkable, start, waypoint) >= required - 0.01
        # progress: the waypoint is on a shorter path than where the unit stands
        here = (int(round(start[0])), int(round(start[1])))
        assert distance[waypoint] <= distance[here] + 1e-6
        heading = aa.any_angle_heading(walkable, distance, start, radius)
        assert heading.dtype == np.float32 and float(np.linalg.norm(heading)) == pytest.approx(1.0, abs=1e-5)
        checked += 1
    assert checked > 40


def test_visible_goal_is_taken_directly():
    walkable = _map()
    distance = distance_map_numba(walkable, 3, 20)
    start = (8.3, 13.0)  # open ground east of the long wall
    assert aa.any_angle_waypoint(walkable, distance, start, 0.275) == (3, 20)
    heading = aa.any_angle_heading(walkable, distance, start, 0.275)
    expected = np.array([20 - 13.0, -(3 - 8.3)])
    assert np.allclose(heading, expected / np.linalg.norm(expected), atol=1e-5)
    assert abs(math.degrees(math.atan2(heading[1], heading[0])) % 45) > 1  # not a compass direction


def test_goal_behind_wall_heads_for_the_corner_not_through_it():
    walkable = _map()
    distance = distance_map_numba(walkable, 10, 15)
    start = (10.0, 5.0)
    waypoint = aa.any_angle_waypoint(walkable, distance, start, 0.325)
    assert waypoint is not None and waypoint[0] in (4, 15)  # round one end of the wall
    assert _dense_min_clearance(walkable, start, waypoint) >= 0.325


def test_adjacent_goal_and_degenerate_cases():
    walkable = _map()
    distance = distance_map_numba(walkable, 3, 4)
    heading = aa.any_angle_heading(walkable, distance, (3.0, 3.0), 0.275)  # goal next door
    assert np.allclose(heading, [1.0, 0.0], atol=1e-6)
    # standing on the goal centre: nothing to head for
    assert aa.any_angle_heading(walkable, distance, (3.0, 4.0), 0.275) is None
    # just off the goal centre: head back to it
    assert np.allclose(aa.any_angle_heading(walkable, distance, (3.0, 4.2), 0.275), [-1.0, 0.0], atol=1e-6)
    # no path at all
    unreachable = np.full((24, 24), np.inf, dtype=np.float32)
    assert aa.any_angle_heading(walkable, unreachable, (3.0, 3.0), 0.275) is None
    # boxed in: every neighbour blocked
    boxed = np.zeros((24, 24), dtype=bool)
    boxed[3, 3] = True
    boxed_distance = np.full((24, 24), np.inf, dtype=np.float32)
    boxed_distance[3, 3] = 0.0
    assert aa.any_angle_heading(boxed, boxed_distance, (3.0, 3.0), 0.275) is None
    # standing on a cell the distance field does not reach (e.g. a soft-blocked danger cell)
    assert aa.any_angle_heading(walkable, unreachable, (8.0, 8.0), 0.275) is None


def test_labels_unchanged_and_moves_are_unit_vectors():
    o = _observation((0, 0, 1, 0, 2))
    labels = StrategicHeuristicV21AnyAngle().teach(_team(o))
    reference = StrategicHeuristicV21().teach(_team(o))
    assert labels.keys() == reference.keys()
    for name in labels:
        assert np.array_equal(labels[name].mask, reference[name].mask)
        assert labels[name].valid == reference[name].valid
    moves = StrategicHeuristicV21AnyAngle().act(_team(o))
    for name, move in moves.items():
        norm = float(np.linalg.norm(move))
        assert norm == pytest.approx(1.0, abs=1e-5) or (norm == 0.0 and not labels[name].valid)


def test_falls_back_to_the_8_direction_move(monkeypatch):
    o = _observation((0, 0, 1, 0, 2))
    monkeypatch.setattr(aa, "any_angle_heading", lambda *args, **kwargs: None)
    monkeypatch.setattr(aa, "_waypoint_blocked", lambda *args, **kwargs: None)  # the mixin's path (O1)
    player = StrategicHeuristicV21AnyAngle(exact_hunt=False)
    moves = player.act(_team(o))
    reference = StrategicHeuristicV21().teach(_team(o))
    for name, move in moves.items():
        assert np.allclose(move, label_action(reference[name]))
    assert "path" not in player.anyangle_stats
