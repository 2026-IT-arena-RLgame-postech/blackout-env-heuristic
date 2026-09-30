"""Offline movement probe: 8-direction label following vs any-angle, on a replica of Unity's physics.

No Unity.  ``UnitMovementSystem.ClampPosition`` is reproduced exactly (per-axis X then Y, each
axis clamped against the blocked neighbours of the current cell that lie in its direction, circle
vs 1x1 square pushed out along the closest-point normal), with ``decisionPeriod`` 2 x 0.02 s ticks.
Random (start, goal) pairs on the real map (walls from a recorded shard, the enemy base blocked)
are walked by

* ``8dir``   -- V21's path following: ``TeachLabeller._progress_moves`` on the current tile, mean
                of the tied compass moves; on the goal tile the compass move nearest its centre;
* ``any``    -- ``any_angle_heading`` (falling back to ``8dir``); on the goal tile its exact centre;

until the unit is within the 0.22-tile arrival radius of the goal centre.  Reported: time to
arrive, its ratio (any / 8dir) with SE, the ratio to the 8-neighbour path length, and failures
(not arrived within 3x the path time + 4 s).

    .venv/bin/python examples/anyangle_probe.py --pairs 400
"""

from __future__ import annotations

import argparse
import glob
import math

import numpy as np

from blackout_env.heuristics._native import distance_map_numba
from blackout_env.heuristics.any_angle import UNIT_RADIUS, any_angle_heading
from blackout_env.heuristics.teach import DIRECTIONS, TeachLabeller
from blackout_env.heuristics.v17_planner import StrategicHeuristicV17

TICK = 0.02
DECISION_TICKS = 2
ARRIVAL = 0.22
NEIGHBOUR_DELTA = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # (x, y)
CLASSES = {"collector": (0, 4.0), "hunter": (1, 6.0), "carrier": (2, 6.0)}


def load_walkable(path_glob: str) -> np.ndarray:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise SystemExit(f"no shard matches {path_glob}")
    graphic = np.load(files[0])["graphic"][0]
    walkable = graphic[..., 1] < 0.5
    enemy_spawn = np.argwhere(graphic[..., 5] > 0.5)
    if len(enemy_spawn):
        walkable &= ~StrategicHeuristicV17._base_mask(tuple(int(v) for v in enemy_spawn[0]), walkable.shape)
    return walkable


class UnityPhysics:
    """World frame: x = col, y = -row; cell centres at integers; cell = floor(v + 0.5)."""

    def __init__(self, walkable: np.ndarray, radius: float) -> None:
        self.walkable, self.radius = walkable, radius

    def blocked(self, cx: int, cy: int) -> bool:
        row, col = -cy, cx
        h, w = self.walkable.shape
        return not (0 <= row < h and 0 <= col < w and self.walkable[row, col])

    def _clamp_square(self, px, py, cx, cy):
        qx = min(max(px, cx - 0.5), cx + 0.5)
        qy = min(max(py, cy - 0.5), cy + 0.5)
        dx, dy = px - qx, py - qy
        d2 = dx * dx + dy * dy
        if d2 < self.radius * self.radius:
            if d2 <= 1e-12:
                dx, dy, d2 = 0.0, 1.0, 1.0
            d = math.sqrt(d2)
            return qx + dx / d * self.radius, qy + dy / d * self.radius
        return px, py

    def _axis(self, x, y, dx, dy):
        if dx == 0.0 and dy == 0.0:
            return x, y
        px, py = x + dx, y + dy
        cx, cy = math.floor(x + 0.5), math.floor(y + 0.5)
        for ddx, ddy in NEIGHBOUR_DELTA:
            xx, yy = ddx * dx, ddy * dy
            if xx >= 0 and yy >= 0 and xx + yy > 0 and self.blocked(cx + ddx, cy + ddy):
                px, py = self._clamp_square(px, py, cx + ddx, cy + ddy)
        return px, py

    def move(self, x, y, vx, vy, step):
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return x, y
        dx, dy = vx / norm * step, vy / norm * step
        x, y = self._axis(x, y, dx, 0.0)
        return self._axis(x, y, 0.0, dy)


def eight_dir(walkable, distance, row, col, goal):
    pos = (int(round(row)), int(round(col)))
    if pos == goal:
        toward = np.array([goal[1] - col, row - goal[0]], dtype=np.float32)
        return DIRECTIONS[int(np.argmax(DIRECTIONS @ toward))]
    mask = TeachLabeller._progress_moves(distance, pos, walkable, walkable.shape)
    if not mask.any():
        return None
    v = DIRECTIONS[mask].mean(axis=0)
    return v / np.linalg.norm(v)


def any_dir(walkable, distance, row, col, goal, radius):
    pos = (int(round(row)), int(round(col)))
    if pos == goal:
        v = np.array([goal[1] - col, row - goal[0]], dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-6)
    heading = any_angle_heading(walkable, distance, (row, col), radius)
    if heading is None:
        return eight_dir(walkable, distance, row, col, goal)
    return heading


def walk(walkable, distance, start, goal, radius, speed, controller, limit_ticks):
    phys = UnityPhysics(walkable, radius)
    row, col = start
    step = speed * TICK
    for tick in range(0, limit_ticks, DECISION_TICKS):
        if math.hypot(row - goal[0], col - goal[1]) <= ARRIVAL:
            return tick
        if controller == "8dir":
            v = eight_dir(walkable, distance, row, col, goal)
        else:
            v = any_dir(walkable, distance, row, col, goal, radius)
        if v is None:
            return None
        for _ in range(DECISION_TICKS):
            x, y = phys.move(col, -row, float(v[0]), float(v[1]), step)
            col, row = x, -y
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", default="reports/v19_analysis/shards/*.npz")
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    walkable = load_walkable(args.shards)
    cells = np.argwhere(walkable)
    rng = np.random.default_rng(args.seed)
    for cls_name, (cls, speed) in CLASSES.items():
        radius = UNIT_RADIUS[cls]
        times = {"8dir": [], "any": []}
        path_ticks, fails = [], {"8dir": 0, "any": 0}
        n = 0
        while n < args.pairs:
            s, g = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
            goal = (int(g[0]), int(g[1]))
            distance = distance_map_numba(walkable, goal[0], goal[1])
            d = float(distance[s[0], s[1]])
            if not math.isfinite(d) or d < args.min_distance:
                continue
            start = (float(s[0]) + rng.uniform(-0.15, 0.15), float(s[1]) + rng.uniform(-0.15, 0.15))
            limit = int((3.0 * d / speed + 4.0) / TICK)
            result = {c: walk(walkable, distance, start, goal, radius, speed, c, limit) for c in times}
            n += 1
            for c, t in result.items():
                if t is None:
                    fails[c] += 1
            if all(t is not None for t in result.values()):
                for c, t in result.items():
                    times[c].append(t * TICK)
                path_ticks.append(d / speed)
        a, b = np.array(times["any"]), np.array(times["8dir"])
        ratio = a / b
        print(f"{cls_name:9s} r={radius:.3f} v={speed:.0f}  pairs={n}  failures 8dir={fails['8dir']} any={fails['any']}")
        print(f"  time 8dir {b.mean():.3f}s  any {a.mean():.3f}s  any/8dir {ratio.mean():.4f} +- "
              f"{ratio.std(ddof=1) / np.sqrt(len(ratio)):.4f} (SE)  median {np.median(ratio):.4f}"
              f"  saved {(b - a).mean():.3f} +- {(b - a).std(ddof=1) / np.sqrt(len(a)):.3f} s/trip")
        p = np.array(path_ticks)
        print(f"  time / (octile path / speed): 8dir {np.mean(b / p):.4f}  any {np.mean(a / p):.4f}"
              f"   share of trips faster {np.mean(a < b - 1e-9):.0%}, slower {np.mean(a > b + 1e-9):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
