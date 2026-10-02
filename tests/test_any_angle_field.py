"""Any-angle distance field (``any_angle.anyangle_distance_field``) and strategic_v21_anyangle_ed."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV21AnyAngleED
from blackout_env.heuristics import any_angle as aa
from blackout_env.heuristics._native import distance_map_numba
from blackout_env.heuristics.teach import TeachLabeller

from .test_teach_heuristic import _observation, _team

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from anyangle_optimality import VisibilityGraph  # noqa: E402


def _rooms():
    """Walls, a long wall, a short one and a pocket: corners in every orientation."""
    walkable = np.ones((24, 24), dtype=bool)
    walkable[0, :] = walkable[-1, :] = walkable[:, 0] = walkable[:, -1] = False
    walkable[5:15, 10] = False
    walkable[18, 3:9] = False
    walkable[8, 14:20] = False
    walkable[8:13, 19] = False
    walkable[14:16, 15:17] = False
    walkable[20, 15:21] = False
    walkable[19, 15] = False  # pocket at (19, 16) open only downward-right
    return walkable


def _random_blocks(seed):
    rng = np.random.default_rng(seed)
    walkable = np.ones((24, 24), dtype=bool)
    walkable[0, :] = walkable[-1, :] = walkable[:, 0] = walkable[:, -1] = False
    for _ in range(14):
        r, c = rng.integers(1, 22, size=2)
        walkable[r:r + rng.integers(1, 4), c:c + rng.integers(1, 4)] = False
    return walkable


@pytest.mark.parametrize("walkable", [_rooms(), _random_blocks(0), _random_blocks(1)], ids=["rooms", "blocks0", "blocks1"])
def test_field_matches_the_visibility_graph_optimum(walkable):
    graph = VisibilityGraph(walkable, aa.FIELD_CLEARANCE)
    cells = np.argwhere(walkable)
    rng = np.random.default_rng(0)
    field_ratio, octile_ratio = [], []
    for _ in range(150):
        s, g = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        optimal = graph.shortest((float(s[0]), float(s[1])), (int(g[0]), int(g[1])))
        if not math.isfinite(optimal) or optimal < 2.0:
            continue
        field = aa.anyangle_distance_field(walkable, [tuple(g)])
        octile = distance_map_numba(walkable, int(g[0]), int(g[1]))
        field_ratio.append(float(field[tuple(s)]) / optimal)
        octile_ratio.append(float(octile[tuple(s)]) / optimal)
    field_ratio, octile_ratio = np.array(field_ratio), np.array(octile_ratio)
    assert len(field_ratio) > 80
    # the optimum's nodes and the field's coincide, so they agree up to sampling tolerance
    assert field_ratio.mean() == pytest.approx(1.0, abs=2e-3)
    assert field_ratio.min() > 0.995 and field_ratio.max() < 1.02
    assert octile_ratio.mean() > field_ratio.mean() + 0.02  # what the field fixes


def test_open_floor_is_straight_line_distance():
    walkable = np.zeros((24, 24), dtype=bool)
    walkable[1:-1, 1:-1] = True
    field = aa.anyangle_distance_field(walkable, [(12, 12)])
    rows, cols = np.mgrid[:24, :24]
    euclid = np.hypot(rows - 12, cols - 12)
    # a corner-free room: every cell centre keeps 0.5 from the walls, so all see the source
    assert np.allclose(field[walkable], euclid[walkable], atol=1e-4)
    assert np.isinf(field[~walkable]).all()


@pytest.mark.parametrize("seed", range(6))
def test_reachability_bound_and_progress_moves(seed):
    """Finite exactly where the octile field is, never longer, and always a compass move that
    makes progress (TeachLabeller labels stay valid) -- also on masks with Hunter discs removed."""
    rng = np.random.default_rng(seed)
    walkable = _rooms() if seed % 2 else _random_blocks(seed)
    rows, cols = np.mgrid[:24, :24]
    for _ in range(int(seed // 2)):
        r, c = np.argwhere(walkable)[rng.integers(walkable.sum())]
        walkable = walkable & ~((rows - r) ** 2 + (cols - c) ** 2 <= 9)
    cells = np.argwhere(walkable)
    for gi in rng.choice(len(cells), 6, replace=False):
        goal = tuple(int(v) for v in cells[gi])
        field = aa.anyangle_distance_field(walkable, [goal])
        octile = distance_map_numba(walkable, goal[0], goal[1])
        assert np.array_equal(np.isfinite(field), np.isfinite(octile))
        finite = np.isfinite(octile)
        assert (field[finite] <= octile[finite] + 1e-3).all()  # octile uses 1.4142 < sqrt 2
        for cell in cells:
            cell = tuple(int(v) for v in cell)
            if math.isfinite(field[cell]) and field[cell] > 0.0:
                assert TeachLabeller._progress_moves(field, cell, walkable, walkable.shape).any(), (goal, cell)


def test_multi_source_is_the_minimum_and_blocked_sources_work():
    walkable = _rooms()
    tiles = [(3, 3), (3, 4), (4, 3), (20, 12), (12, 16)]
    multi = aa.anyangle_distance_field(walkable, tiles)
    single = np.minimum.reduce([aa.anyangle_distance_field(walkable, [t]) for t in tiles])
    assert np.allclose(multi, single, atol=1e-3)
    # a unit standing on a cell the mask excludes still gets a field (as distance_map_numba)
    blocked = (10, 10)
    assert not walkable[blocked]
    field = aa.anyangle_distance_field(walkable, [blocked])
    octile = distance_map_numba(walkable, *blocked)
    assert field[blocked] == 0.0
    assert np.array_equal(np.isfinite(field), np.isfinite(octile))


def test_heuristic_is_registered_ranks_by_the_field_and_moves():
    assert POLICY_REGISTRY["strategic_v21_anyangle_ed"] is StrategicHeuristicV21AnyAngleED
    o = _observation((0, 0, 1, 0, 2))
    player = StrategicHeuristicV21AnyAngleED()
    moves = player.act(_team(o))
    labels = StrategicHeuristicV21AnyAngleED().teach(_team(o))
    assert moves.keys() == labels.keys()
    for name, move in moves.items():
        norm = float(np.linalg.norm(move))
        assert norm == pytest.approx(1.0, abs=1e-5) or (norm == 0.0 and not labels[name].valid)
    assert any(label.valid for label in labels.values())
    # every per-unit distance it ranks by is the any-angle field
    walkable = o["graphic"][..., 1] < 0.5
    assert np.array_equal(player._cached_distance_map(walkable, (5, 5)),
                          aa.anyangle_distance_field(walkable, [(5, 5)]))
    assert player._distance_map(walkable, (5, 5))[5, 17] == pytest.approx(12.0, abs=1e-4)


def _min_clearance(walkable, start, end, samples=400):
    t = np.linspace(0.0, 1.0, samples)[1:]
    return min(aa.point_clearance(walkable, (start[0] + (end[0] - start[0]) * s, start[1] + (end[1] - start[1]) * s))
               for s in t)


@pytest.mark.parametrize("radius", sorted(aa.UNIT_RADIUS.values()))
def test_field_waypoints_keep_clearance_and_every_class_arrives(radius, monkeypatch):
    """Cell-or-corner-node waypoints keep the controller's clearance, and walking them on the
    Unity physics replica (examples/anyangle_probe.py) reaches every goal -- no dithering."""
    import anyangle_probe as probe

    walkable = _rooms()
    cells = np.argwhere(walkable)
    rng = np.random.default_rng(1)
    speed = 6.0 if radius != aa.UNIT_RADIUS[0] else 4.0
    arrived = 0
    for _ in range(40):
        goal = tuple(int(v) for v in cells[rng.integers(len(cells))])
        s = cells[rng.integers(len(cells))]
        start = (float(s[0]) + rng.uniform(-0.15, 0.15), float(s[1]) + rng.uniform(-0.15, 0.15))
        field, nodes, node_g = aa.anyangle_distance_field(walkable, [goal], with_nodes=True)
        if not math.isfinite(field[tuple(s)]) or field[tuple(s)] < 2.0:
            continue
        waypoint = aa.field_waypoint(walkable, field, nodes, node_g, start, radius)
        assert waypoint is not None
        required = min(radius + aa.CLEARANCE_MARGIN, aa.point_clearance(walkable, start))
        assert _min_clearance(walkable, start, waypoint) >= required - 0.01

        def heading(walkable_, distance, position, radius_, **kwargs):
            wp = aa.field_waypoint(walkable_, distance, nodes, node_g, position, radius_)
            return None if wp is None else aa.world_heading(wp[0] - position[0], wp[1] - position[1])

        monkeypatch.setattr(probe, "any_angle_heading", heading)
        limit = int((3.0 * float(field[tuple(s)]) / speed + 4.0) / probe.TICK)
        assert probe.walk(walkable, field, start, goal, radius, speed, "any", limit) is not None, (start, goal)
        arrived += 1
    assert arrived > 25


def test_ablations_off_is_strategic_v21_anyangle():
    from blackout_env.heuristics import StrategicHeuristicV21AnyAngle

    o = _observation((0, 0, 1, 0, 2))
    off = StrategicHeuristicV21AnyAngleED(field_ranking=False, field_goals=False).act(_team(o))
    old = StrategicHeuristicV21AnyAngle().act(_team(o))
    assert off.keys() == old.keys()
    for name in off:
        assert np.array_equal(off[name], old[name])
