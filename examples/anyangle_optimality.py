"""How far are the 8-direction and any-angle controllers from the true shortest path?

Same setup as ``examples/anyangle_probe.py`` (real map, Unity-exact physics, random start/goal
pairs per class), plus the optimum: the shortest path for a circle of radius ``c = r + margin``
among the blocked 1x1 squares, from a visibility graph whose nodes sit outside every convex corner
of a blocked square, offset by ``c`` along both axes (so a segment along the wall between two
corners keeps ``c``; nodes not that clear of every other square are dropped), plus the start and
the goal centre.  Squares inflated to squares rather than rounded ones: the optimum is slightly
over-estimated round each corner (by at most ~0.3 c), which favours the controllers.

Reported per class: each controller's travelled time x speed divided by the optimal length (1.00 =
optimal; physics losses at wall contacts count against the controller), and the octile (8-way
grid) path length over the optimum for reference.

    .venv/bin/python examples/anyangle_optimality.py --pairs 300
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anyangle_probe import CLASSES, TICK, load_walkable, walk  # noqa: E402

from blackout_env.heuristics._native import distance_map_numba
from blackout_env.heuristics.any_angle import CLEARANCE_MARGIN, UNIT_RADIUS, _segment_clear, _point_clearance

TOLERANCE = 1e-3  # clearance slack for nodes placed exactly at the required distance


def corner_nodes(walkable: np.ndarray, clearance: float) -> np.ndarray:
    """[N, 2] (row, col) points just outside the convex corners of blocked squares."""
    blocked = ~walkable
    h, w = walkable.shape
    offset = clearance + TOLERANCE
    nodes = set()
    for r, c in np.argwhere(blocked):
        for dr in (-1, 1):
            for dc in (-1, 1):
                # a convex corner: both side neighbours and the diagonal one are free (or off-map for walls only)
                def free(rr, cc):
                    return 0 <= rr < h and 0 <= cc < w and walkable[rr, cc]
                if free(r + dr, c) and free(r, c + dc) and free(r + dr, c + dc):
                    nodes.add((r + dr * (0.5 + offset), c + dc * (0.5 + offset)))
    pts = np.array(sorted(nodes), dtype=np.float64)
    blocked_c = np.ascontiguousarray(blocked)
    keep = [i for i, (y, x) in enumerate(pts) if _point_clearance(blocked_c, y, x) >= clearance - TOLERANCE]
    return pts[keep]


class VisibilityGraph:
    def __init__(self, walkable: np.ndarray, clearance: float):
        self.blocked = np.ascontiguousarray(~walkable)
        self.required = clearance - TOLERANCE
        self.nodes = corner_nodes(walkable, clearance)
        n = len(self.nodes)
        self.adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._clear(self.nodes[i], self.nodes[j]):
                    d = float(np.hypot(*(self.nodes[i] - self.nodes[j])))
                    self.adj[i].append((j, d))
                    self.adj[j].append((i, d))

    def _clear(self, a, b) -> bool:
        return bool(_segment_clear(self.blocked, float(a[0]), float(a[1]), float(b[0]), float(b[1]), self.required, 0.05))

    def shortest(self, start, goal) -> float:
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if self._clear(start, goal):
            return float(np.hypot(*(start - goal)))
        n = len(self.nodes)
        to_goal = {i: float(np.hypot(*(self.nodes[i] - goal))) for i in range(n) if self._clear(self.nodes[i], goal)}
        dist = np.full(n, np.inf)
        heap = []
        for i in range(n):
            if self._clear(start, self.nodes[i]):
                dist[i] = float(np.hypot(*(start - self.nodes[i])))
                heapq.heappush(heap, (dist[i], i))
        best = np.inf
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i] or d >= best:
                continue
            if i in to_goal:
                best = min(best, d + to_goal[i])
            for j, w in self.adj[i]:
                if d + w < dist[j]:
                    dist[j] = d + w
                    heapq.heappush(heap, (dist[j], j))
        return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", default="reports/v19_analysis/shards/*.npz")
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    walkable = load_walkable(args.shards)
    cells = np.argwhere(walkable)
    rng = np.random.default_rng(args.seed)
    print(f"map {walkable.shape}, {len(cells)} walkable cells")
    for cls_name, (cls, speed) in CLASSES.items():
        radius = UNIT_RADIUS[cls]
        graph = VisibilityGraph(walkable, radius + CLEARANCE_MARGIN)
        ratios = {"8dir": [], "any": [], "octile": []}
        detour = []  # optimal / straight line: how obstructed the pairs are
        n = 0
        while n < args.pairs:
            s, g = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
            goal = (int(g[0]), int(g[1]))
            distance = distance_map_numba(walkable, goal[0], goal[1])
            d = float(distance[s[0], s[1]])
            if not math.isfinite(d) or d < args.min_distance:
                continue
            start = (float(s[0]) + rng.uniform(-0.15, 0.15), float(s[1]) + rng.uniform(-0.15, 0.15))
            optimal = graph.shortest(start, goal)
            if not math.isfinite(optimal) or optimal < 1.0:
                continue
            limit = int((3.0 * d / speed + 4.0) / TICK)
            times = {c: walk(walkable, distance, start, goal, radius, speed, c, limit) for c in ("8dir", "any")}
            if any(t is None for t in times.values()):
                continue
            n += 1
            for c, t in times.items():
                ratios[c].append(t * TICK * speed / optimal)
            ratios["octile"].append(d / optimal)
            detour.append(optimal / math.hypot(start[0] - goal[0], start[1] - goal[1]))
        stats = {k: np.array(v) for k, v in ratios.items()}
        fmt = lambda a: f"{a.mean():.4f} +- {a.std(ddof=1) / np.sqrt(len(a)):.4f} (p90 {np.percentile(a, 90):.3f})"
        print(f"{cls_name:9s} r={radius:.3f}  pairs={n}  graph nodes={len(graph.nodes)}  optimal/straight {np.mean(detour):.3f}")
        print(f"  travelled / optimal:  8dir {fmt(stats['8dir'])}   any {fmt(stats['any'])}")
        print(f"  octile grid path / optimal: {fmt(stats['octile'])}")
        gap8, gapa = stats["8dir"] - 1, stats["any"] - 1
        print(f"  (below 1.00 = within the optimum's own over-estimate round corners and the 0.22-tile arrival radius)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
