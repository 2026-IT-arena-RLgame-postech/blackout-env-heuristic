"""V28 whose staged reserve joins a tied race as a fourth walker (Q10, docs/policy_v2_design_ko.md §41).

Q10 (student P17 vs its teacher V28 on the same seeds): P17 cannot stand still either, and its reserve does not
hold at 0.5 tiles but creeps on to a median 0.04 tiles by decision 98 -- in 112 of 128 matches against V28 it
turns into a Hunter at decision 99 together with the three walkers (four Hunters against three).  The three
Hunter-for-Hunter trades on the sanctuary leave P17 with the only Hunter on the map, i.e. the first race won
outright instead of a second race at ~decision 102.  Against an opponent whose walkers are a decision late
(V23: first Hunter at 100), the race is already won and a fourth Hunter is a Collector less on the economy.

``reserve_join`` (default ``"off"`` = V28 exactly): while the reserve is staged (V27's ``reserve_stage``) it does
not stand at the stage distance but walks the exact race path into the sanctuary --
``"always"``: always; ``"tie"``: only while the race looks tied, i.e. the enemy's nearest Collector outside
its base is at most ``join_margin`` tiles (exact race distance) behind our lead walker; ``"reserve"``: tied and the
enemy stages a close reserve of its own (more of its Collectors within ``join_reach`` tiles of its lead walker than
we have walkers) -- the opponents that would otherwise tie or win the second race.  When the enemy is further
behind (we win the race anyway), or keeps no close reserve (V28's reserve wins the second race), the reserve holds
as in V28.  Observation-only.
"""

from __future__ import annotations

import numpy as np

from .any_angle import UNIT_RADIUS, pixel_position
from .strategic import COLLECTOR, SITE_HUNTER
from .v25_race import StrategicHeuristicV25
from .v28_trade_hold import StrategicHeuristicV28

JOIN_MODES = ("off", "always", "tie", "reserve")


class StrategicHeuristicV29(StrategicHeuristicV28):
    """V28 + the staged reserve walks into a tied race (``reserve_join``)."""

    def __init__(self, *, reserve_join: str = "off", join_margin: float = 0.08, join_reach: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if reserve_join not in JOIN_MODES:
            raise ValueError(f"reserve_join must be one of {JOIN_MODES}, got {reserve_join!r}")
        self.reserve_join = reserve_join
        self.join_margin = float(join_margin)
        self.join_reach = float(join_reach)
        self.joining = False  # for inspection: the staged reserve walks in this tick

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        self.joining = False
        if self.reserve_join == "off" or not self._staged_xy:
            return goals
        if self.reserve_join == "always":
            self.joining = True
            return goals
        tiles = tuple(sorted((int(y), int(x)) for y, x in np.argwhere((graphic[..., SITE_HUNTER] > 0.5) & walkable)))
        if not tiles:
            return goals
        shape = graphic.shape[:2]
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        tile_set = set(tiles)
        walkers = [planner.distance(pixel_position(states[row_for[n]], shape)) for n in controlled
                   if n in goals and n not in self.staged
                   and any(kind == "transform" and set(t) & tile_set for t, kind in goals[n])]
        lead = min(walkers, default=np.inf)
        ctx = self._teach_ctx
        theirs = [planner.distance(pixel_position(states[e.row], shape)) for e in ctx["enemies"]
                  if e.cls == COLLECTOR and not e.in_base]
        theirs = [d for d in theirs if np.isfinite(d)]
        self.joining = bool(theirs) and np.isfinite(lead) and min(theirs) <= lead + self.join_margin
        if self.joining and self.reserve_join == "reserve":
            # the enemy stages a reserve of its own: more Collectors within join_reach of its lead than our walkers
            self.joining = sum(d <= min(theirs) + self.join_reach for d in theirs) > len(walkers)
        return goals

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        if (self.joining and label.kind == "transform" and self._class_id(state) == COLLECTOR
                and (float(state[0]), float(state[1])) in self._staged_xy):
            self._count("reserve_join")
            return StrategicHeuristicV25._any_angle(self, label, state, states, goals, walkable, shape)
        return super()._any_angle(label, state, states, goals, walkable, shape)
