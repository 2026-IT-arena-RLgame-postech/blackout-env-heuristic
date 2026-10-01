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
def _best_waypoint(blocked, distance, y, x, radius, margin, step, far_bias):
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
            values[n] = e + d - far_bias * e
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


# ----------------------------------------------------------------------- any-angle distance field
#
# ``distance_map_numba`` is the 8-neighbour (octile) path length.  It over-estimates the shortest
# path by up to 8% on open floor (1/cos 22.5 deg) and by ~6% on average over real pairs
# (``examples/anyangle_optimality.py``), and by how much depends on the direction: targets ranked
# by octile length are ranked by a distance the any-angle controller does not travel.
#
# ``anyangle_distance_field`` is the shortest path for a point that keeps ``FIELD_CLEARANCE`` from
# every blocked 1x1 square (as the optimum in ``anyangle_optimality.py``), within 0.1% of it on the
# real map (``tests/test_any_angle_field.py``) at ~0.4 ms a map (octile: 0.02 ms):
#
# 1. corner nodes -- one just outside every convex corner of a blocked square, offset by the
#    clearance along both axes -- and their mutual visibility (cached per walkable mask);
# 2. per query, an exact visibility-graph Dijkstra over the nodes from the source cells;
# 3. a Lazy-Theta*-style sweep over cell centres in Dijkstra order: each cell takes its parent
#    *point* (a source, a corner node, or a cell centre) from a neighbour, the node a corner seeds
#    into its own cell, and is checked for line of sight to it when expanded.  If that fails it
#    falls back to the best (parent-of-neighbour, or neighbour itself) among its expanded grid
#    neighbours.  Grid edges between cell centres keep 0.5, so the field is finite exactly where
#    the octile field is and never longer than it (beyond its 1.4142 rounding of sqrt 2).
#
# ``field_waypoint`` lets the controller head for corner nodes as well as cell centres: the field's
# paths bend at the nodes, and chasing the nearest centres instead zigzags round the taut line.

# Clearance of the field's segments: the largest any controller needs (Hunter radius 0.325 + margin
# 0.05 + half the controller's 0.05 sampling step), so every field path is one the controller of
# any class can follow.  With the Collector's 0.325, a Hunter's own cell centre could undercut every
# visible cell ahead round a corner it cannot pass that close to, and it dithered on the centre.
FIELD_CLEARANCE = CLEARANCE_MARGIN + max(UNIT_RADIUS.values()) + 0.5 * SAMPLE_STEP
FIELD_STEP = 0.1  # sampling step of the field's line-of-sight test
SEED_RADIUS = 2  # a corner node seeds the cells within this many rows/cols of it
_FIELD_TOLERANCE = 1e-3  # nodes sit at exactly the clearance: test against clearance - tolerance


@njit(cache=True)
def _corner_nodes(walkable, clearance):
    """[K, 2] (row, col) points just outside the convex corners of blocked squares (in-grid ones)."""
    h, w = walkable.shape
    blocked = np.empty((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            blocked[y, x] = not walkable[y, x]
    offset = 0.5 + clearance + _FIELD_TOLERANCE
    out = np.empty((4 * h * w, 2), dtype=np.float64)
    n = 0
    for y in range(h):
        for x in range(w):
            if walkable[y, x]:
                continue
            for dy in (-1, 1):
                for dx in (-1, 1):
                    ay, ax = y + dy, x
                    by, bx = y, x + dx
                    cy, cx = y + dy, x + dx
                    if not (0 <= ay < h and 0 <= cx < w and 0 <= cy < h and 0 <= bx < w):
                        continue
                    if not (walkable[ay, ax] and walkable[by, bx] and walkable[cy, cx]):
                        continue
                    py = y + dy * offset
                    px = x + dx * offset
                    if _point_clearance(blocked, py, px) < clearance - _FIELD_TOLERANCE:
                        continue
                    dup = False
                    for k in range(n):
                        if abs(out[k, 0] - py) < 1e-6 and abs(out[k, 1] - px) < 1e-6:
                            dup = True
                            break
                    if not dup:
                        out[n, 0] = py
                        out[n, 1] = px
                        n += 1
    return out[:n].copy()


@njit(cache=True)
def _node_adjacency(walkable, nodes, required, step):
    h, w = walkable.shape
    blocked = np.empty((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            blocked[y, x] = not walkable[y, x]
    k = nodes.shape[0]
    adj = np.full((k, k), np.inf, dtype=np.float64)
    for i in range(k):
        adj[i, i] = 0.0
        for j in range(i + 1, k):
            if _segment_clear(blocked, nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1], required, step):
                dy = nodes[i, 0] - nodes[j, 0]
                dx = nodes[i, 1] - nodes[j, 1]
                d = math.sqrt(dy * dy + dx * dx)
                adj[i, j] = d
                adj[j, i] = d
    return adj


@njit(cache=True, inline="always")
def _heap_push(heap_cost, heap_y, heap_x, size, cost, y, x):
    j = size
    heap_cost[j] = cost
    heap_y[j] = y
    heap_x[j] = x
    while j > 0:
        parent = (j - 1) // 2
        if heap_cost[parent] <= heap_cost[j]:
            break
        heap_cost[parent], heap_cost[j] = heap_cost[j], heap_cost[parent]
        heap_y[parent], heap_y[j] = heap_y[j], heap_y[parent]
        heap_x[parent], heap_x[j] = heap_x[j], heap_x[parent]
        j = parent
    return size + 1


@njit(cache=True)
def anyangle_field_numba(walkable, nodes, adj, sources_y, sources_x, required, step):
    """Any-angle path length from every cell to the nearest source cell (float32, inf = unreachable).

    ``nodes`` / ``adj`` come from ``_corner_nodes`` / ``_node_adjacency`` on the same mask.  Sources
    may be non-walkable (a unit standing on an excluded cell), as in ``distance_map_numba``."""
    h, w = walkable.shape
    blocked = np.empty((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            blocked[y, x] = not walkable[y, x]
    k = nodes.shape[0]
    n_src = len(sources_y)

    # 1. visibility-graph Dijkstra over the corner nodes
    node_g = np.full(k, np.inf, dtype=np.float64)
    for s in range(n_src):
        sy = float(sources_y[s])
        sx = float(sources_x[s])
        for v in range(k):
            dy = nodes[v, 0] - sy
            dx = nodes[v, 1] - sx
            d = math.sqrt(dy * dy + dx * dx)
            if d < node_g[v] and _segment_clear(blocked, sy, sx, nodes[v, 0], nodes[v, 1], required, step):
                node_g[v] = d
    done = np.zeros(k, dtype=np.bool_)
    for _ in range(k):
        best = np.inf
        u = -1
        for v in range(k):
            if not done[v] and node_g[v] < best:
                best = node_g[v]
                u = v
        if u < 0:
            break
        done[u] = True
        for v in range(k):
            if not done[v]:
                c = best + adj[u, v]
                if c < node_g[v]:
                    node_g[v] = c

    # 2. cell sweep; each cell's parent is a point (py, px) at path length pg from the sources
    g = np.full((h, w), np.inf, dtype=np.float64)
    par_y = np.zeros((h, w), dtype=np.float64)
    par_x = np.zeros((h, w), dtype=np.float64)
    par_g = np.zeros((h, w), dtype=np.float64)
    checked = np.zeros((h, w), dtype=np.bool_)
    # best verified node seed per cell: an optimistic relaxation may overwrite it, a repair restores it
    seed_g = np.full((h, w), np.inf, dtype=np.float64)
    seed_y = np.zeros((h, w), dtype=np.float64)
    seed_x = np.zeros((h, w), dtype=np.float64)
    seed_pg = np.zeros((h, w), dtype=np.float64)
    closed = np.zeros((h, w), dtype=np.bool_)
    capacity = h * w * 8 + n_src + k * (2 * SEED_RADIUS + 1) ** 2 + 1
    heap_cost = np.empty(capacity, dtype=np.float64)
    heap_y = np.empty(capacity, dtype=np.int64)
    heap_x = np.empty(capacity, dtype=np.int64)
    size = 0
    for s in range(n_src):
        sy = sources_y[s]
        sx = sources_x[s]
        if sy < 0 or sy >= h or sx < 0 or sx >= w or g[sy, sx] == 0.0:
            continue
        g[sy, sx] = 0.0
        par_y[sy, sx] = sy
        par_x[sy, sx] = sx
        par_g[sy, sx] = 0.0
        checked[sy, sx] = True
        size = _heap_push(heap_cost, heap_y, heap_x, size, 0.0, sy, sx)
    for v in range(k):
        if not math.isfinite(node_g[v]):
            continue
        # seed the node into the cells round it (a corner's own pocket may see only this node)
        vy = int(math.floor(nodes[v, 0] + 0.5))
        vx = int(math.floor(nodes[v, 1] + 0.5))
        for cy in range(vy - SEED_RADIUS, vy + SEED_RADIUS + 1):
            for cx in range(vx - SEED_RADIUS, vx + SEED_RADIUS + 1):
                if cy < 0 or cy >= h or cx < 0 or cx >= w or not walkable[cy, cx]:
                    continue
                dy = nodes[v, 0] - cy
                dx = nodes[v, 1] - cx
                c = node_g[v] + math.sqrt(dy * dy + dx * dx)
                if c < g[cy, cx] - 1e-9 and _segment_clear(blocked, nodes[v, 0], nodes[v, 1], float(cy), float(cx),
                                                           required, step):
                    g[cy, cx] = c
                    par_y[cy, cx] = nodes[v, 0]
                    par_x[cy, cx] = nodes[v, 1]
                    par_g[cy, cx] = node_g[v]
                    checked[cy, cx] = True
                    seed_g[cy, cx] = c
                    seed_y[cy, cx] = nodes[v, 0]
                    seed_x[cy, cx] = nodes[v, 1]
                    seed_pg[cy, cx] = node_g[v]
                    size = _heap_push(heap_cost, heap_y, heap_x, size, c, cy, cx)

    dys = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
    dxs = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
    root2 = math.sqrt(2.0)
    while size > 0:
        cost = heap_cost[0]
        y = heap_y[0]
        x = heap_x[0]
        size -= 1
        heap_cost[0] = heap_cost[size]
        heap_y[0] = heap_y[size]
        heap_x[0] = heap_x[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < size and heap_cost[left] < heap_cost[smallest]:
                smallest = left
            if right < size and heap_cost[right] < heap_cost[smallest]:
                smallest = right
            if smallest == i:
                break
            heap_cost[i], heap_cost[smallest] = heap_cost[smallest], heap_cost[i]
            heap_y[i], heap_y[smallest] = heap_y[smallest], heap_y[i]
            heap_x[i], heap_x[smallest] = heap_x[smallest], heap_x[i]
            i = smallest
        if closed[y, x] or cost > g[y, x] + 1e-9:
            continue
        fy = float(y)
        fx = float(x)
        if not checked[y, x] and not _segment_clear(blocked, par_y[y, x], par_x[y, x], fy, fx, required, step):
            # no line of sight from the inherited parent: best of this cell's node seed and its
            # expanded grid neighbours, through their parent (if visible) or the neighbour itself
            best = seed_g[y, x]
            by = seed_y[y, x]
            bx = seed_x[y, x]
            bg = seed_pg[y, x]
            for d in range(8):
                ny = y + dys[d]
                nx = x + dxs[d]
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not closed[ny, nx]:
                    continue
                diagonal = dys[d] != 0 and dxs[d] != 0
                if diagonal and (not walkable[y, nx] or not walkable[ny, x]):
                    continue
                c = g[ny, nx] + (root2 if diagonal else 1.0)
                if c < best:
                    best = c
                    by = float(ny)
                    bx = float(nx)
                    bg = g[ny, nx]
                qy = par_y[ny, nx]
                qx = par_x[ny, nx]
                ey = qy - fy
                ex = qx - fx
                c = par_g[ny, nx] + math.sqrt(ey * ey + ex * ex)
                if c < best - 1e-9 and _segment_clear(blocked, qy, qx, fy, fx, required, step):
                    best = c
                    by = qy
                    bx = qx
                    bg = par_g[ny, nx]
            g[y, x] = best
            par_y[y, x] = by
            par_x[y, x] = bx
            par_g[y, x] = bg
        closed[y, x] = True
        py = par_y[y, x]
        px = par_x[y, x]
        pg = par_g[y, x]
        for d in range(8):
            ny = y + dys[d]
            nx = x + dxs[d]
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not walkable[ny, nx] or closed[ny, nx]:
                continue
            if dys[d] != 0 and dxs[d] != 0 and (not walkable[y, nx] or not walkable[ny, x]):
                continue
            ey = py - ny
            ex = px - nx
            cand = pg + math.sqrt(ey * ey + ex * ex)  # optimistic: checked when ``n`` is expanded
            if cand >= g[ny, nx] - 1e-9:
                continue
            g[ny, nx] = cand
            par_y[ny, nx] = py
            par_x[ny, nx] = px
            par_g[ny, nx] = pg
            checked[ny, nx] = False
            size = _heap_push(heap_cost, heap_y, heap_x, size, cand, ny, nx)
    return g.astype(np.float32), node_g


_FIELD_GRAPHS: dict = {}


def _field_graph(walkable: np.ndarray, clearance: float, step: float):
    """(nodes, adjacency) for a mask, cached by its bytes (static masks hit every tick)."""
    key = (walkable.shape, walkable.tobytes(), clearance, step)
    graph = _FIELD_GRAPHS.get(key)
    if graph is None:
        nodes = _corner_nodes(walkable, clearance)
        graph = (nodes, _node_adjacency(walkable, nodes, clearance - _FIELD_TOLERANCE, step))
        if len(_FIELD_GRAPHS) >= 256:
            _FIELD_GRAPHS.pop(next(iter(_FIELD_GRAPHS)))
        _FIELD_GRAPHS[key] = graph
    return graph


def anyangle_distance_field(
    walkable: np.ndarray,
    sources,
    *,
    clearance: float = FIELD_CLEARANCE,
    step: float = FIELD_STEP,
    with_nodes: bool = False,
):
    """Any-angle shortest-path length from every cell to the nearest of ``sources`` ((row, col) cells).

    ``with_nodes``: also return the corner nodes ``[K, 2]`` and their path lengths ``[K]``."""
    walkable = np.ascontiguousarray(walkable, dtype=np.bool_)
    pts = np.asarray(list(sources), dtype=np.int64).reshape(-1, 2)
    nodes, adj = _field_graph(walkable, float(clearance), float(step))
    field, node_g = anyangle_field_numba(walkable, nodes, adj, np.ascontiguousarray(pts[:, 0]),
                                         np.ascontiguousarray(pts[:, 1]), float(clearance) - _FIELD_TOLERANCE,
                                         float(step))
    return (field, nodes, node_g) if with_nodes else field


@njit(cache=True)
def _best_node(blocked, nodes, node_g, d_here, y, x, required, step):
    """(index, value) of the visible corner node minimising |p - v| + D(v) among nodes closer to the
    goal than the unit's cell; (-1, inf) if none.  Ties (within 1e-4) to the farther node."""
    best = -1
    best_value = np.inf
    best_far = -1.0
    for v in range(nodes.shape[0]):
        d = node_g[v]
        if not math.isfinite(d) or d >= d_here:
            continue
        dy = nodes[v, 0] - y
        dx = nodes[v, 1] - x
        e = math.sqrt(dy * dy + dx * dx)
        if e < 0.05:
            continue
        value = e + d
        if value > best_value + 1e-4 or (value > best_value - 1e-4 and e <= best_far):
            continue
        if _segment_clear(blocked, y, x, nodes[v, 0], nodes[v, 1], required, step):
            best = v
            best_value = value
            best_far = e
    return best, best_value


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
    far_bias: float = 0.0,
) -> tuple[int, int] | None:
    """The cell to head for from continuous ``position`` (row, col), or None (no visible cell)."""
    blocked = ~np.ascontiguousarray(walkable, dtype=bool)
    cy, cx = _best_waypoint(blocked, np.ascontiguousarray(distance, dtype=np.float32),
                            float(position[0]), float(position[1]), float(radius), float(margin), float(step),
                            float(far_bias))
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


def field_waypoint(
    walkable: np.ndarray,
    distance: np.ndarray,
    nodes: np.ndarray,
    node_g: np.ndarray,
    position: tuple[float, float],
    radius: float,
    *,
    margin: float = CLEARANCE_MARGIN,
    step: float = SAMPLE_STEP,
) -> tuple[float, float] | None:
    """``any_angle_waypoint`` over cell centres *and* the field's corner nodes: the visible point
    minimising |p - q| + D(q).  An exact field's shortest path bends at corner nodes, not at cell
    centres, so without them the controller zigzags between centres near the taut line."""
    blocked = ~np.ascontiguousarray(walkable, dtype=bool)
    h, w = distance.shape
    y, x = float(position[0]), float(position[1])
    here = (min(max(int(math.floor(y + 0.5)), 0), h - 1), min(max(int(math.floor(x + 0.5)), 0), w - 1))
    d_here = float(distance[here])
    if not math.isfinite(d_here):
        return None
    best = None  # (value, length, point)
    cell = any_angle_waypoint(walkable, distance, position, radius, margin=margin, step=step)
    if cell is not None:
        e = math.hypot(cell[0] - y, cell[1] - x)
        best = (e + float(distance[cell]), e, (float(cell[0]), float(cell[1])))
    if len(nodes):
        required = max(min(radius + margin, float(_point_clearance(blocked, y, x)) - 1e-3) + 0.5 * step, 0.0)
        v, value = _best_node(blocked, nodes, node_g, d_here, y, x, required, step)
        if v >= 0:
            e = math.hypot(nodes[v, 0] - y, nodes[v, 1] - x)
            if best is None or value < best[0] - 1e-4 or (value <= best[0] + 1e-4 and e > best[1]):
                best = (value, e, (float(nodes[v, 0]), float(nodes[v, 1])))
    return None if best is None else best[2]


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
        heading = self._path_heading(walkable, tiles, position, radius)
        if heading is None or float(heading @ base) <= MAX_TURN_COS:
            return None
        self._count("path")
        return heading

    def _path_heading(self, walkable, tiles, position, radius) -> np.ndarray | None:
        """Heading along the goal cluster's distance field (None: no visible waypoint)."""
        distance = self._distance_to(walkable, tiles)
        return any_angle_heading(walkable, distance, position, radius, margin=self.clearance_margin)

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


class AnyAngleFieldMixin:
    """Mix in first: every path length the heuristic ranks by is the any-angle field.

    All of V21's distances go through ``_distance_map`` (one start; per-unit ETAs, storage
    fields, hunt/camp/economy scores) or ``TeachLabeller._distance_to`` (a goal cluster:
    composition candidates, storage choice, the labels and the any-angle controller's ``D``), so
    overriding those two moves target selection, labels and the controller onto the same field.

    Ablations: ``field_ranking=False`` keeps ``_distance_map`` octile; ``field_goals=False`` keeps
    ``_distance_to`` (goal-cluster choices, labels and the controller) octile."""

    def __init__(self, *, field_ranking: bool = True, field_goals: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.field_ranking = bool(field_ranking)
        self.field_goals = bool(field_goals)
        if not self.field_ranking:
            self._distance_map = super()._distance_map  # the octile staticmethod
        self._field_nodes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # id(field) -> nodes

    @staticmethod
    def _distance_map(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        """Any-angle path length from one cell (``StrategicHeuristicV2._distance_map``'s contract)."""
        return anyangle_distance_field(walkable, (tuple(start),))

    def _distance_to(self, walkable: np.ndarray, tiles) -> np.ndarray:
        if not self.field_goals:  # TeachLabeller._distance_to on the octile map
            if tiles and isinstance(tiles[0], (int, np.integer)):
                tiles = (tuple(tiles),)
            key = ("octile", walkable.tobytes(), tuple(sorted(tiles)))
            cached = self._teach_distance_cache.get(key)
            if cached is None:
                octile = super(AnyAngleFieldMixin, self)._distance_map
                cached = np.minimum.reduce([octile(walkable, tile) for tile in key[2]])
                if len(self._teach_distance_cache) > 4096:
                    self._teach_distance_cache.clear()
                self._teach_distance_cache[key] = cached
            return cached
        if tiles and isinstance(tiles[0], (int, np.integer)):
            tiles = (tuple(tiles),)
        key = (walkable.tobytes(), tuple(sorted(tiles)))
        cached = self._teach_distance_cache.get(key)
        if cached is None:
            cached, nodes, node_g = anyangle_distance_field(walkable, key[1], with_nodes=True)  # one sweep
            if len(self._teach_distance_cache) > 4096:
                self._teach_distance_cache.clear()
            self._teach_distance_cache[key] = cached
            self._field_nodes[id(cached)] = (cached, nodes, node_g)
            if len(self._field_nodes) > 4096:
                self._field_nodes.clear()
        return cached

    def _path_heading(self, walkable, tiles, position, radius) -> np.ndarray | None:
        """Head for the best visible cell centre or corner node of the goal cluster's field."""
        if not self.field_goals:
            return super()._path_heading(walkable, tiles, position, radius)
        distance = self._distance_to(walkable, tiles)
        entry = self._field_nodes.get(id(distance))
        if entry is None or entry[0] is not distance:
            entry = (distance, *anyangle_distance_field(walkable, tiles, with_nodes=True)[1:])
        waypoint = field_waypoint(walkable, distance, entry[1], entry[2], position, radius, margin=self.clearance_margin)
        return None if waypoint is None else world_heading(waypoint[0] - position[0], waypoint[1] - position[1])


class StrategicHeuristicV21AnyAngleED(AnyAngleFieldMixin, StrategicHeuristicV21AnyAngle):
    """V21-anyangle whose targets, roles and threats are ranked by any-angle (Euclidean shortest
    path) length instead of 8-neighbour grid length; the controller follows the same field."""
