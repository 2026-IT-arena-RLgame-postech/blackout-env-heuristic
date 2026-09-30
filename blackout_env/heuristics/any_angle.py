"""Any-angle movement for the label-following heuristics (opt-in).

Every heuristic moves in the 8 compass directions of its grid A* / distance maps, but Unity's
``Unit.Move`` takes ``input.normalized * deltaTime * MoveSpeed``: any heading is possible (only
the zero vector means "stand").  This module turns a goal's shortest-path distance field into a
continuous heading, and ``AnyAngleMixin`` applies it to a ``TeachLabeller`` heuristic without
changing anything it decides: the 8-direction labels (``teach``) are untouched, only the move
``act`` returns is replaced.

Unity constraints the heading respects (``UnitMovementSystem.ClampPosition``,
``CollisionUtils.GetClampedPosition``):

* a unit is a circle -- radius ``UNIT_RADIUS`` per class, from ``UnitData.CollisionBound``
  (``Unit.Move`` passes the class bound, not the ``UnitSize``-scaled one) -- and every
  non-walkable cell is a 1x1 square (walls, map edge, the enemy base's floor);
* motion is resolved per axis (X then Y), pushing the circle out of each blocked square along the
  closest-point normal, so a unit that touches a wall slides along it and one that clips a convex
  corner is deflected round it (losing part of that tick's displacement) rather than stopped.

So a straight segment is safe when it keeps ``radius + margin`` from every blocked square.  A
unit already closer than that (pressed against a wall) may keep its current clearance: sliding
along a wall costs nothing.  Since ``radius + margin < 0.5``, only the 3x3 cells around a point
can be that close, which keeps the clearance test local.

Heading choice (a one-step, field-guided Theta*): among cells ``c`` visible from the continuous
position ``p`` with that clearance, take the one minimising ``|p - c| + D(c)`` (``D`` = the
8-neighbour distance field to the goal, ties to the farther cell) and head for its centre.  ``D``
never under-estimates the remaining path, so a visible goal is always taken directly, and a goal
round a corner is approached along the tangent to the corner's clearance circle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .strategic import CARRIER, COLLECTOR, HUNTER
from .teach import DIRECTIONS, TeachLabel, label_action

# UnitData/*.asset CollisionBound.Width / 2 (circles).
UNIT_RADIUS = {COLLECTOR: 0.275, CARRIER: 0.225, HUNTER: 0.325}
CLEARANCE_MARGIN = 0.05  # extra distance from walls on a planned segment (tiles)
SAMPLE_STEP = 0.05  # segment sampling step (tiles); the test adds half of it to the clearance
MAX_TURN_COS = 0.0  # a heading more than 90 degrees off the 8-direction label falls back to it

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a project dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _point_clearance(blocked, y, x):
    """Distance from (row y, col x) to the nearest blocked 1x1 square, capped at 0.5.

    Cells outside the grid count as blocked (Unity: no tile = not walkable)."""
    h, w = blocked.shape
    cy = int(math.floor(y + 0.5))
    cx = int(math.floor(x + 0.5))
    best = 0.5
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny = cy + dy
            nx = cx + dx
            if 0 <= ny < h and 0 <= nx < w and not blocked[ny, nx]:
                continue
            # square [ny-0.5, ny+0.5] x [nx-0.5, nx+0.5]
            qy = min(max(y, ny - 0.5), ny + 0.5)
            qx = min(max(x, nx - 0.5), nx + 0.5)
            d = math.sqrt((y - qy) * (y - qy) + (x - qx) * (x - qx))
            if d < best:
                best = d
    return best


@njit(cache=True)
def _segment_clear(blocked, y0, x0, y1, x1, required, step):
    """Every sampled point of the segment keeps ``required`` from the blocked squares."""
    length = math.sqrt((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0))
    n = int(math.ceil(length / step))
    if n < 1:
        n = 1
    for i in range(1, n + 1):
        t = i / n
        if _point_clearance(blocked, y0 + (y1 - y0) * t, x0 + (x1 - x0) * t) < required:
            return False
    return True


@njit(cache=True)
def _best_waypoint(blocked, distance, y, x, radius, margin, step):
    """(row, col) of the visible cell minimising |p - c| + D(c); (-1, -1) if none.

    Candidates are cells closer to the goal than the unit's own cell, plus that cell's centre
    (the way back to the grid when nothing ahead is visible from an off-centre position)."""
    h, w = distance.shape
    here_y = min(max(int(math.floor(y + 0.5)), 0), h - 1)
    here_x = min(max(int(math.floor(x + 0.5)), 0), w - 1)
    d_here = distance[here_y, here_x]
    if not math.isfinite(d_here):
        return -1, -1
    required = min(radius + margin, _point_clearance(blocked, y, x) - 1e-3) + 0.5 * step
    if required < 0.0:
        required = 0.0
    n = 0
    values = np.empty(h * w, dtype=np.float64)
    far = np.empty(h * w, dtype=np.float64)
    cells = np.empty(h * w, dtype=np.int64)
    for cy in range(h):
        for cx in range(w):
            d = distance[cy, cx]
            if not math.isfinite(d) or blocked[cy, cx]:
                continue
            if d >= d_here and not (cy == here_y and cx == here_x):
                continue  # only cells closer to the goal (or this cell's own centre)
            e = math.sqrt((cy - y) * (cy - y) + (cx - x) * (cx - x))
            if e < 0.05:  # (almost) there: no heading to take
                continue
            values[n] = e + d
            far[n] = e
            cells[n] = cy * w + cx
            n += 1
    if n == 0:
        return -1, -1
    # ascending value, ties (within 1e-4) to the farther cell
    keys = np.round(values[:n] * 1e4) * 1e3 - far[:n]
    order = np.argsort(keys)
    for k in range(n):
        i = order[k]
        cy = cells[i] // w
        cx = cells[i] % w
        if _segment_clear(blocked, y, x, float(cy), float(cx), required, step):
            return cy, cx
    return -1, -1


def pixel_position(state: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """Continuous (row, col) of a unit, cell centres at integers (as ``_to_pixel_float``)."""
    return ((1.0 - float(state[1])) * 0.5 * shape[0] - 0.5, (float(state[0]) + 1.0) * 0.5 * shape[1] - 0.5)


def world_heading(d_row: float, d_col: float) -> np.ndarray | None:
    """Unit world-frame action (+x = column +1, +y = row -1) for a grid offset; None if zero."""
    norm = math.hypot(d_row, d_col)
    if norm < 1e-6:
        return None
    return np.array([d_col / norm, -d_row / norm], dtype=np.float32)


def any_angle_waypoint(
    walkable: np.ndarray,
    distance: np.ndarray,
    position: tuple[float, float],
    radius: float,
    *,
    margin: float = CLEARANCE_MARGIN,
    step: float = SAMPLE_STEP,
) -> tuple[int, int] | None:
    """The cell to head for from continuous ``position`` (row, col), or None (no visible cell)."""
    blocked = ~np.ascontiguousarray(walkable, dtype=bool)
    cy, cx = _best_waypoint(blocked, np.ascontiguousarray(distance, dtype=np.float32),
                            float(position[0]), float(position[1]), float(radius), float(margin), float(step))
    return None if cy < 0 else (int(cy), int(cx))


def any_angle_heading(
    walkable: np.ndarray,
    distance: np.ndarray,
    position: tuple[float, float],
    radius: float,
    **kwargs,
) -> np.ndarray | None:
    """World-frame unit heading toward ``any_angle_waypoint``, or None (caller falls back)."""
    waypoint = any_angle_waypoint(walkable, distance, position, radius, **kwargs)
    if waypoint is None:
        return None
    return world_heading(waypoint[0] - position[0], waypoint[1] - position[1])


def segment_clear(walkable: np.ndarray, start: tuple[float, float], end: tuple[float, float],
                  required: float, step: float = SAMPLE_STEP) -> bool:
    """Python entry to the clearance test (tests, diagnostics)."""
    blocked = ~np.ascontiguousarray(walkable, dtype=bool)
    return bool(_segment_clear(blocked, float(start[0]), float(start[1]), float(end[0]), float(end[1]),
                               float(required), float(step)))


def point_clearance(walkable: np.ndarray, point: tuple[float, float]) -> float:
    blocked = ~np.ascontiguousarray(walkable, dtype=bool)
    return float(_point_clearance(blocked, float(point[0]), float(point[1])))


@dataclass(frozen=True)
class AnyAngleLabel(TeachLabel):
    """A TeachLabel (the 8-direction mask is unchanged) plus the continuous heading to play."""

    heading: np.ndarray | None = None


class AnyAngleMixin:
    """Mix in before a ``TeachLabeller`` heuristic: same labels, any-angle moves.

    * following a path: head for the best visible waypoint of the goal's distance field;
    * on the goal tile: head exactly for its centre (the base label rounds to a compass move);
    * ``exact_hunt``: a Hunter whose chase target is visible heads for the enemy's continuous
      position rather than its tile;
    * ``exact_evasion``: threat repulsion plays its continuous vector, not the nearest compass move.

    Anything else -- no visible waypoint, a unit standing on a cell the planning map excludes, a
    heading more than 90 degrees off the label -- plays the 8-direction label (``label_action``).
    ``anyangle_stats`` counts which case each move took.
    """

    def __init__(self, *, exact_hunt: bool = True, exact_evasion: bool = False,
                 clearance_margin: float = CLEARANCE_MARGIN, **kwargs):
        super().__init__(**kwargs)
        self.exact_hunt = bool(exact_hunt)
        self.exact_evasion = bool(exact_evasion)
        self.clearance_margin = float(clearance_margin)
        self.anyangle_stats: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.anyangle_stats[key] = self.anyangle_stats.get(key, 0) + 1

    def act(self, obs):
        out = {}
        for name, label in self.teach(obs).items():
            heading = getattr(label, "heading", None)
            out[name] = heading if heading is not None and label.valid else label_action(label)
        return out

    def _label(self, state, states, goals, walkable, shape) -> TeachLabel:
        label = super()._label(state, states, goals, walkable, shape)
        if not label.valid:
            return label
        if label.evading and not self.exact_evasion:
            self._count("evade_8dir")
            return label
        heading = self._any_angle(label, state, states, goals, walkable, shape)
        if heading is None:
            self._count("fallback")
            return label
        return AnyAngleLabel(label.mask, label.valid, label.kind, label.target, label.goals, label.evading, heading)

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        position = pixel_position(state, shape)
        cls = self._class_id(state)
        radius = UNIT_RADIUS.get(cls, 0.325)
        base = DIRECTIONS[label.mask].mean(axis=0)
        base = base / max(float(np.linalg.norm(base)), 1e-6)

        if label.evading:
            vector = self._evasion_vector(state, states, base)
            self._count("evade")
            return vector

        pos = self._to_pixel(state[:2], shape)
        if not (0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]) or not walkable[pos]:
            return None

        if self.exact_hunt and cls == HUNTER:
            heading = self._hunt_heading(state, states, goals, walkable, shape, position, radius)
            if heading is not None:
                self._count("hunt")
                return heading

        if any(pos in tiles for tiles, _ in goals):  # final approach to the goal tile's centre
            self._count("approach")
            return world_heading(pos[0] - position[0], pos[1] - position[1])

        tiles = tuple(dict.fromkeys(tile for goal_tiles, _ in goals for tile in goal_tiles))
        distance = self._distance_to(walkable, tiles)
        heading = any_angle_heading(walkable, distance, position, radius, margin=self.clearance_margin)
        if heading is None or float(heading @ base) <= MAX_TURN_COS:
            return None
        self._count("path")
        return heading

    def _hunt_heading(self, state, states, goals, walkable, shape, position, radius) -> np.ndarray | None:
        targets = {tile for tiles, kind in goals if kind == "hunt" for tile in tiles}
        if not targets:
            return None
        required = min(radius + self.clearance_margin, point_clearance(walkable, position) - 1e-3) + 0.5 * SAMPLE_STEP
        best = None
        for enemy in states[states[:, 2] < 0]:
            if self._to_pixel(enemy[:2], shape) not in targets:
                continue
            where = pixel_position(enemy, shape)
            gap = math.hypot(where[0] - position[0], where[1] - position[1])
            if best is not None and gap >= best[0]:
                continue
            if segment_clear(walkable, position, where, max(required, 0.0)):
                best = (gap, where)
        if best is None:
            return None
        return world_heading(best[1][0] - position[0], best[1][1] - position[1])

    def _evasion_vector(self, state, states, delta) -> np.ndarray | None:
        """``TeachLabeller._evasion``'s summed repulsion, unrounded."""
        cls = self._class_id(state)
        total = delta.astype(np.float32).copy()
        for enemy in states[states[:, 2] < 0]:
            if not (self._class_id(enemy) == HUNTER or cls == CARRIER):
                continue
            away = state[:2] - enemy[:2]
            dist = float(np.sqrt(away.dot(away)))
            if 1e-5 < dist < self.threat_radius:
                total += (away / dist) * ((self.threat_radius - dist) / self.threat_radius) * 9.0
        norm = float(np.linalg.norm(total))
        return None if norm < 1e-6 else (total / norm).astype(np.float32)


from .v21_safe_transform import StrategicHeuristicV21  # noqa: E402  (after the mixin it uses)


class StrategicHeuristicV21AnyAngle(AnyAngleMixin, StrategicHeuristicV21):
    """V21 with any-angle moves: identical goals and 8-direction labels, continuous headings."""
