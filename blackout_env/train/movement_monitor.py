"""
Per-unit movement-failure diagnostics: tracks how long each unit spends idle (no meaningful
action, no movement) or blocked (strong commanded movement, no actual movement -- i.e. walking
into a wall/obstacle) over an episode.

Originally written for examples/benchmark_heuristics.py's heuristic-reliability tournaments;
factored out here so QMIXTrainer/offline_pretrain.py's periodic in-training eval (see
periodic_eval.py) can reuse the exact same idle/blocked definitions instead of re-deriving them,
and benchmark_heuristics.py now imports from here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FailureRuns:
    idle: list[int] = field(default_factory=list)
    blocked: list[int] = field(default_factory=list)
    unit_ticks: int = 0

    @staticmethod
    def _summarize(runs: list[int], threshold: int, unit_ticks: int) -> dict[str, float | int]:
        incidents = [length for length in runs if length >= threshold]
        return {
            "incidents": len(incidents),
            "per_1000_unit_ticks": 1000.0 * len(incidents) / max(1, unit_ticks),
            "worst_ticks": max(runs, default=0),
            "incident_ticks": sum(incidents),
        }

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            "idle_6s": self._summarize(self.idle, 300, self.unit_ticks),
            "blocked_0.24s": self._summarize(self.blocked, 12, self.unit_ticks),
        }


class MovementMonitor:
    """Track consecutive actionless and commanded-but-motionless unit ticks."""

    def __init__(self):
        self.result = FailureRuns()
        self._idle = {}
        self._blocked = {}

    def observe(self, names, before, after, actions):
        for name in names:
            if name not in actions:
                continue
            row = int(name.split("_")[1])
            action_norm = float(np.linalg.norm(actions[name]))
            movement = float(np.linalg.norm(after[row, :2] - before[row, :2]))
            # Respawn/teleport and class/cargo transitions delimit a run rather than count as
            # successful navigation; this prevents unrelated episodes being joined together.
            transition = movement > 0.15 or not np.array_equal(before[row, 3:], after[row, 3:])
            self.result.unit_ticks += 1
            self._update(name, "idle", action_norm <= 0.05 and movement <= 2e-4 and not transition)
            self._update(name, "blocked", action_norm >= 0.35 and movement <= 2e-4 and not transition)

    def _update(self, name, kind, active):
        current = self._idle if kind == "idle" else self._blocked
        runs = self.result.idle if kind == "idle" else self.result.blocked
        if active:
            current[name] = current.get(name, 0) + 1
        elif current.get(name, 0):
            runs.append(current.pop(name))

    def finish(self):
        self.result.idle.extend(self._idle.values())
        self.result.blocked.extend(self._blocked.values())
        self._idle.clear()
        self._blocked.clear()


def aggregate(results: list[FailureRuns]) -> dict[str, dict[str, float | int]]:
    merged = FailureRuns()
    for result in results:
        merged.idle.extend(result.idle)
        merged.blocked.extend(result.blocked)
        merged.unit_ticks += result.unit_ticks
    return merged.summary()
