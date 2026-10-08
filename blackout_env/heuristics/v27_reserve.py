"""V25 that stages a reserve walker behind its race walkers (Q6, docs/policy_v2_design_ko.md §33).

Q6 (search vs plain on the same seeds): against V23 the search's gain after the opening is
almost all in race-tied matches, and there it is the *second race*.  A tied race (both sides'
walkers enter at the same decision) ends in Hunter-for-Hunter trades on the sanctuary (both die
and respawn as Collectors at their spawns, ~16 tiles away), and whoever next turns a Collector
into a Hunter owns the sanctuary: the other side's walkers see the Hunter and turn away (threat
evasion) and are caught -- the same monopoly as winning the first race.  The next walker is not
a respawned one but one of the two Collectors that did not walk the opening.  The search keeps
its nearest such Collector ~1 tile closer to the sanctuary at the race decision (median 6.0 vs
7.1 tiles, plain and V23 both 7.1) and wins the second race 44/52 vs plain's 31/52.

``reserve_stage`` (tiles, 0 = off): while the race is on (``_racing``), no enemy Hunter is out
and some other Collector of ours is walking to the Hunter sanctuary ahead of it, the empty
Collector (no item) that is nearest the sanctuary among those without a transform goal walks
the exact race path too (``race_walk``) but stands still once its exact distance to the
sanctuary is at most ``reserve_stage``.  When the walkers have transformed (and traded), it is
the nearest Collector and V25's own composition sends it in; when an enemy Hunter is out it is
released to V25's rules (threat evasion, ``race_commit``).  ``reserve_count`` such Collectors.

Default off (``reserve_stage=0``) = V25 exactly (``tests/test_v27_reserve.py``).  Every rule is a function of the observation.
"""

from __future__ import annotations

import numpy as np

from ..env.constants import unit_index
from .any_angle import UNIT_RADIUS, pixel_position
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .v25_race import StrategicHeuristicV25


class StrategicHeuristicV27(StrategicHeuristicV25):
    """V25 + a staged reserve walker for the second race (``reserve_stage``)."""

    def __init__(self, *, reserve_stage: float = 0.0, reserve_count: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.reserve_stage = float(reserve_stage)
        self.reserve_count = int(reserve_count)
        self.staged: list[str] = []  # for inspection: reserve walkers this tick
        self._staged_xy: set[tuple[float, float]] = set()

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        self.staged = []
        self._staged_xy = set()
        if self.reserve_stage <= 0 or self.reserve_count <= 0 or not self._race_now:
            return goals
        ctx = self._teach_ctx
        if any(e.cls == HUNTER and not e.in_base for e in ctx["enemies"]):
            return goals
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)  # sorted (row-major)
        if not tiles:
            return goals
        shape = graphic.shape[:2]
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        tile_set = set(tiles)

        def to_sanctuary(g):
            return any(kind == "transform" and set(t) & tile_set for t, kind in g)

        walkers = [planner.distance(pixel_position(states[row_for[n]], shape))
                   for n in controlled if n in goals and to_sanctuary(goals[n])]
        if not walkers:
            return goals
        lead = min(walkers)
        cands = []
        for name in controlled:
            state = states[row_for[name]]
            if name in goals or self._class_id(state) != COLLECTOR or self._is_holding(state):
                continue
            d = planner.distance(pixel_position(state, shape))
            if np.isfinite(d) and d > lead:
                cands.append((d, unit_index(name), name))
        for _, _, name in sorted(cands)[:self.reserve_count]:
            goals[name] = ((tiles, "transform"),)
            self.staged.append(name)
            self._staged_xy.add((float(states[row_for[name]][0]), float(states[row_for[name]][1])))
        return goals

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        if (self._staged_xy and label.kind == "transform" and self._class_id(state) == COLLECTOR
                and (float(state[0]), float(state[1])) in self._staged_xy):
            tiles = tuple(sorted({t for g_tiles, kind in goals if kind == "transform" for t in g_tiles}))
            plan_map = getattr(self, "_teach_walkable", walkable)
            planner = self._planner(np.ascontiguousarray(plan_map, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
            if planner.distance(pixel_position(state, shape)) <= self.reserve_stage:
                self._count("reserve_hold")
                return np.zeros(2, dtype=np.float32)
        return super()._any_angle(label, state, states, goals, walkable, shape)
