"""Exact taut-path controller for the opening race (docs/policy_v2_design_ko.md §28).

The race to the central Hunter sanctuary is decided by single decisions: both sides' walkers
cover ~15.84 tiles and the first one whose centre enters a sanctuary tile wins (a tie lets both
transform).  The S3 search wins races V23 ties by shaving a few thousandths of a tile off the
walk -- the plain walk ends ~0.005 tiles short of the sanctuary boundary at the decision before it
enters (``examples/race_analysis.py``).  V23's controller (``AnyAngleFieldMixin``) bends at the
any-angle field's corner nodes, which sit ``0.5 + FIELD_CLEARANCE`` (0.9 tiles) from the blocked
cell's centre along *both* axes -- 0.4 tiles off each wall, i.e. 0.57 tiles from the convex corner
-- and tests segments at ``radius + 0.025``; the unit only needs ``radius`` (0.275 for a Collector:
``UnitMovementSystem`` pushes a circle out of each blocked square, sliding costs nothing).

This module plans on the exact geometry instead: the configuration space of a circle of radius
``r`` among unit squares is bounded by straight edges and quarter-circle arcs of radius ``r``
round convex corners, so the shortest path consists of tangent segments and those arcs.  Nodes
are points on each convex corner's arc at radius ``(r + delta) / cos(step / 2)`` (chords between
neighbours keep ``r + delta``), edges are segments that keep ``r + delta / 2`` from every blocked
square (exact segment-square distance, no sampling), and the goal is the union of the goal tiles'
squares (centre inside = entered).  ``RacePlanner.heading(p)`` heads for the visible point that
minimises ``|p - q| + G(q)`` -- the goal's nearest point when visible, else a node.

Coordinates are the heuristics' pixel frame: (row, col), cell centres at integers, a blocked cell
is the square [row - 0.5, row + 0.5] x [col - 0.5, col + 0.5]; cells outside the grid are blocked.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]

ARC_STEP_DEG = 7.5  # angular spacing of the nodes on a corner arc
DELTA = 2e-3  # clearance beyond the unit radius kept by node chords (tiles)


@njit(cache=True, inline="always")
def _pt_box(y, x, cy, cx, h):
    qy = min(max(y, cy - h), cy + h)
    qx = min(max(x, cx - h), cx + h)
    return math.sqrt((y - qy) * (y - qy) + (x - qx) * (x - qx))


@njit(cache=True, inline="always")
def _pt_seg(py, px, y0, x0, y1, x1):
    dy, dx = y1 - y0, x1 - x0
    L2 = dy * dy + dx * dx
    t = 0.0 if L2 <= 0.0 else ((py - y0) * dy + (px - x0) * dx) / L2
    t = min(max(t, 0.0), 1.0)
    qy, qx = y0 + t * dy, x0 + t * dx
    return math.sqrt((py - qy) * (py - qy) + (px - qx) * (px - qx))


@njit(cache=True, inline="always")
def _seg_hits_box(y0, x0, y1, x1, cy, cx, h):
    """Liang-Barsky: does the segment intersect the closed square?"""
    t0, t1 = 0.0, 1.0
    dy, dx = y1 - y0, x1 - x0
    for k in range(4):
        if k == 0:
            p, q = -dx, x0 - (cx - h)
        elif k == 1:
            p, q = dx, (cx + h) - x0
        elif k == 2:
            p, q = -dy, y0 - (cy - h)
        else:
            p, q = dy, (cy + h) - y0
        if p == 0.0:
            if q < 0.0:
                return False
        else:
            t = q / p
            if p < 0.0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
    return True


@njit(cache=True)
def seg_box_distance(y0, x0, y1, x1, cy, cx, h):
    """Exact distance between a segment and an axis-aligned square (centre, half width)."""
    if _seg_hits_box(y0, x0, y1, x1, cy, cx, h):
        return 0.0
    d = min(_pt_box(y0, x0, cy, cx, h), _pt_box(y1, x1, cy, cx, h))
    for sy in (-1.0, 1.0):
        for sx in (-1.0, 1.0):
            d = min(d, _pt_seg(cy + sy * h, cx + sx * h, y0, x0, y1, x1))
    return d


@njit(cache=True)
def _blocked_at(blocked, y, x):
    h, w = blocked.shape
    return y < 0 or x < 0 or y >= h or x >= w or blocked[y, x]


@njit(cache=True)
def seg_clear_exact(blocked, y0, x0, y1, x1, required):
    """The segment keeps ``required`` (< 1) from every blocked square (off-grid cells blocked)."""
    lo_y = int(math.floor(min(y0, y1) - required + 0.5)) - 1
    hi_y = int(math.floor(max(y0, y1) + required + 0.5)) + 1
    lo_x = int(math.floor(min(x0, x1) - required + 0.5)) - 1
    hi_x = int(math.floor(max(x0, x1) + required + 0.5)) + 1
    for cy in range(lo_y, hi_y + 1):
        for cx in range(lo_x, hi_x + 1):
            if not _blocked_at(blocked, cy, cx):
                continue
            if seg_box_distance(y0, x0, y1, x1, float(cy), float(cx), 0.5) < required:
                return False
    return True


@njit(cache=True)
def point_clearance_exact(blocked, y, x):
    """Distance to the nearest blocked square (capped at 1.5)."""
    cy0 = int(math.floor(y + 0.5))
    cx0 = int(math.floor(x + 0.5))
    best = 1.5
    for cy in range(cy0 - 2, cy0 + 3):
        for cx in range(cx0 - 2, cx0 + 3):
            if _blocked_at(blocked, cy, cx):
                best = min(best, _pt_box(y, x, float(cy), float(cx), 0.5))
    return best


@njit(cache=True)
def _arc_nodes(blocked, radius, delta, n_arc):
    h, w = blocked.shape
    rho = (radius + delta) / math.cos(0.5 * (0.5 * math.pi) / n_arc)
    out = np.empty((4 * (h + 2) * (w + 2) * (n_arc + 1), 2), dtype=np.float64)
    n = 0
    for y in range(-1, h + 1):
        for x in range(-1, w + 1):
            if not _blocked_at(blocked, y, x):
                continue
            for sy in (-1, 1):
                for sx in (-1, 1):
                    # convex corner of this square: the 3 cells round the corner point are free
                    if _blocked_at(blocked, y + sy, x) or _blocked_at(blocked, y, x + sx) or _blocked_at(blocked, y + sy, x + sx):
                        continue
                    py, px = y + 0.5 * sy, x + 0.5 * sx
                    for k in range(n_arc + 1):
                        a = 0.5 * math.pi * k / n_arc
                        ny = py + sy * rho * math.cos(a)
                        nx = px + sx * rho * math.sin(a)
                        if point_clearance_exact(blocked, ny, nx) < radius + delta - 1e-9:
                            continue
                        out[n, 0] = ny
                        out[n, 1] = nx
                        n += 1
    return out[:n].copy()


@njit(cache=True)
def _adjacency(blocked, nodes, required):
    k = nodes.shape[0]
    adj = np.full((k, k), np.inf)
    for i in range(k):
        adj[i, i] = 0.0
        for j in range(i + 1, k):
            if seg_clear_exact(blocked, nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1], required):
                d = math.sqrt((nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2)
                adj[i, j] = d
                adj[j, i] = d
    return adj


@njit(cache=True)
def _goal_point(y, x, boxes, inset):
    """Nearest point of the union of goal squares (inset), and its distance."""
    best, by, bx = np.inf, y, x
    for i in range(boxes.shape[0]):
        cy, cx, hh = boxes[i, 0], boxes[i, 1], boxes[i, 2] - inset
        qy = min(max(y, cy - hh), cy + hh)
        qx = min(max(x, cx - hh), cx + hh)
        d = math.sqrt((y - qy) ** 2 + (x - qx) ** 2)
        if d < best:
            best, by, bx = d, qy, qx
    return best, by, bx


@njit(cache=True)
def _query(blocked, nodes, node_g, boxes, inset, y, x, required):
    """(value, target y, target x) of the best visible point; value inf if none."""
    req = min(required, point_clearance_exact(blocked, y, x) - 1e-6)
    if req < 0.0:
        req = 0.0
    d, gy, gx = _goal_point(y, x, boxes, inset)
    if d <= 0.0:
        return 0.0, y, x
    if seg_clear_exact(blocked, y, x, gy, gx, req):
        return d, gy, gx
    best, ty, tx, far = np.inf, y, x, -1.0
    for v in range(nodes.shape[0]):
        g = node_g[v]
        if not math.isfinite(g):
            continue
        e = math.sqrt((nodes[v, 0] - y) ** 2 + (nodes[v, 1] - x) ** 2)
        if e < 1e-4:
            continue
        val = e + g
        if val > best + 1e-9 or (val > best - 1e-9 and e <= far):
            continue
        if seg_clear_exact(blocked, y, x, nodes[v, 0], nodes[v, 1], req):
            best, ty, tx, far = val, nodes[v, 0], nodes[v, 1], e
    return best, ty, tx


@njit(cache=True)
def _clamp_axis_px(blocked, y, x, dy, dx, r):
    """UnitMovementSystem.ClampAxis in the pixel frame: move by (dy, dx) (one of them 0), then push
    the circle out of every blocked neighbour square on the side of the motion (start cell's)."""
    if dy * dy + dx * dx < 1e-10:
        return y, x
    qy, qx = y + dy, x + dx
    cy0 = int(math.floor(y + 0.5))
    cx0 = int(math.floor(x + 0.5))
    for ddy in (-1, 0, 1):
        for ddx in (-1, 0, 1):
            if ddy == 0 and ddx == 0:
                continue
            a, b = ddy * dy, ddx * dx
            if a >= 0.0 and b >= 0.0 and a + b > 0.0 and _blocked_at(blocked, cy0 + ddy, cx0 + ddx):
                cy, cx = float(cy0 + ddy), float(cx0 + ddx)
                py = min(max(qy, cy - 0.5), cy + 0.5)
                px = min(max(qx, cx - 0.5), cx + 0.5)
                ey, ex = qy - py, qx - px
                d2 = ey * ey + ex * ex
                if d2 < r * r:
                    if d2 <= 0.0:
                        ey, ex, d2 = -1.0, 0.0, 1.0
                    d = math.sqrt(d2)
                    qy, qx = py + ey / d * r, px + ex / d * r
    return qy, qx


@njit(cache=True)
def emulate_move(blocked, y, x, hy, hx, tick_step, ticks, r):
    """Position after ``ticks`` movement ticks along unit heading (hy, hx) (pixel frame), X axis
    (columns) resolved before Y as Unity does."""
    for _ in range(ticks):
        y, x = _clamp_axis_px(blocked, y, x, 0.0, hx * tick_step, r)
        y, x = _clamp_axis_px(blocked, y, x, hy * tick_step, 0.0, r)
    return y, x


@njit(cache=True)
def _best_heading(blocked, nodes, node_g, boxes, inset, required, y, x, base, span, n, tick_step, ticks, r):
    """Heading angle (radians, pixel frame atan2(dy, dx)) within +-span of ``base`` minimising the
    exact remaining path length after the emulated move."""
    best_a, best_v = base, np.inf
    for i in range(n):
        a = base - span + 2.0 * span * i / (n - 1) if n > 1 else base
        ny, nx = emulate_move(blocked, y, x, math.sin(a), math.cos(a), tick_step, ticks, r)
        v = _query(blocked, nodes, node_g, boxes, inset, ny, nx, required)[0]
        if v < best_v - 1e-12:
            best_v, best_a = v, a
    return best_a, best_v


class RacePlanner:
    """Exact shortest paths of a circle of ``radius`` from anywhere to a set of goal cells."""

    def __init__(self, walkable: np.ndarray, goal_cells, radius: float = 0.275, *, delta: float = DELTA,
                 arc_step_deg: float = ARC_STEP_DEG, inset: float = 1e-3, query_slack: float = 0.0):
        self.blocked = ~np.ascontiguousarray(walkable, dtype=bool)
        self.radius, self.delta, self.inset = float(radius), float(delta), float(inset)
        n_arc = max(1, int(round(90.0 / arc_step_deg)))
        self.nodes = _arc_nodes(self.blocked, self.radius, self.delta, n_arc)
        self.required = self.radius + 0.5 * self.delta
        # a query segment from the unit may come within ``radius - query_slack`` of a wall (the clamp
        # slides a unit pressed against a corner round it at the radius)
        self.q_required = self.radius + 0.5 * self.delta - float(query_slack)
        adj = _adjacency(self.blocked, self.nodes, self.required)
        cells = np.asarray(list(goal_cells), dtype=np.float64).reshape(-1, 2)
        self.boxes = np.concatenate([cells, np.full((len(cells), 1), 0.5)], 1)
        k = len(self.nodes)
        g = np.full(k, np.inf)
        for v in range(k):  # nodes that see the goal directly
            d, gy, gx = _goal_point(self.nodes[v, 0], self.nodes[v, 1], self.boxes, self.inset)
            if seg_clear_exact(self.blocked, self.nodes[v, 0], self.nodes[v, 1], gy, gx, self.required):
                g[v] = d
        heap = [(g[v], v) for v in range(k) if math.isfinite(g[v])]
        heapq.heapify(heap)
        done = np.zeros(k, bool)
        while heap:
            d, v = heapq.heappop(heap)
            if done[v]:
                continue
            done[v] = True
            row = adj[v]
            for u in np.nonzero(np.isfinite(row) & ~done)[0]:
                nd = d + row[u]
                if nd < g[u] - 1e-12:
                    g[u] = nd
                    heapq.heappush(heap, (nd, int(u)))
        self.node_g = g

    def distance(self, position) -> float:
        """Shortest-path length from ``position`` (row, col) to the goal region."""
        return float(_query(self.blocked, self.nodes, self.node_g, self.boxes, self.inset,
                            float(position[0]), float(position[1]), self.q_required)[0])

    def target(self, position):
        """The point to head for, or None (no visible node / already inside)."""
        val, ty, tx = _query(self.blocked, self.nodes, self.node_g, self.boxes, self.inset,
                             float(position[0]), float(position[1]), self.q_required)
        if not math.isfinite(val) or val <= 0.0:
            return None
        return ty, tx

    def heading(self, position, *, lookahead: bool = True, tick_step: float = 0.08, ticks: int = 2,
                span_deg: float = 20.0, n: int = 161):
        """World-frame unit move (+x = col + 1, +y = row - 1), or None.  Without ``lookahead``: toward
        ``target``.  With it: of ``n`` headings within ``span_deg`` of that direction, the one whose
        emulated move (``ticks`` x ``tick_step`` tiles, wall clamps) leaves the shortest exact path."""
        t = self.target(position)
        if t is None:
            return None
        dr, dc = t[0] - position[0], t[1] - position[1]
        if math.hypot(dr, dc) < 1e-9:
            return None
        a = math.atan2(dr, dc)
        if lookahead:
            a, _ = _best_heading(self.blocked, self.nodes, self.node_g, self.boxes, self.inset, self.q_required,
                                 float(position[0]), float(position[1]), a, math.radians(span_deg), int(n),
                                 float(tick_step), int(ticks), self.radius)
        return np.array([math.cos(a), -math.sin(a)], dtype=np.float32)
