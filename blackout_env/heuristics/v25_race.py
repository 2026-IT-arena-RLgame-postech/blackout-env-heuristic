"""V23 whose race walkers follow the exact taut path (``race_path.RacePlanner``).

H9 (docs/policy_v2_design_ko.md §28): the S3 search turns tied races (both sides' first Hunter
at decision 100) into wins by entering the sanctuary at decision 99.  Its walkers' paths differ
from plain r5's by ~0.01 tiles: the plain walk ends ~0.005 tiles short of the sanctuary boundary
at decision 99.  V23 walks the same race (it ties r5 and P11 at 100): its controller bends at the
any-angle field's corner nodes (0.4 tiles off the walls) and keeps radius + 0.025 from them.  The
exact configuration-space path (radius 0.275, arcs round the convex corners) is 15.68 tiles,
reachable at decision 99 with ~0.13 tiles to spare.

One decision ahead is decisive against V23-like opponents: their walkers, a hair from the
sanctuary, see our new Hunter and turn away (threat evasion) instead of stepping in, and are
caught.  ``race_walk`` changes nothing else: while ``_racing`` a Collector whose label is a
transform toward the Hunter sanctuary heads along ``RacePlanner.heading`` on the planning map
(the same map V23's controller uses, field items not blocked: ``race_cargo`` keeps a walker
walking with a battery).  Every rule stays a function of the observation.
"""

from __future__ import annotations

import numpy as np

from .any_angle import UNIT_RADIUS, pixel_position
from .race_path import RacePlanner
from .strategic import COLLECTOR, SITE_HUNTER
from .v23_opening import StrategicHeuristicV23


class StrategicHeuristicV25(StrategicHeuristicV23):
    """V23 + exact race walk (``race_walk``); ``race_lookahead`` picks headings by an emulated move."""

    def __init__(self, *, race_walk: bool = True, race_lookahead: bool = False, race_delta: float = 2e-3, **kwargs):
        super().__init__(**kwargs)
        self.race_walk = bool(race_walk)
        self.race_lookahead = bool(race_lookahead)
        self.race_delta = float(race_delta)
        self._planners: dict = {}

    def _planner(self, walkable: np.ndarray, tiles, radius: float) -> RacePlanner:
        key = (walkable.shape, walkable.tobytes(), tiles, radius)
        pl = self._planners.get(key)
        if pl is None:
            if len(self._planners) > 16:
                self._planners.clear()
            pl = self._planners[key] = RacePlanner(walkable, tiles, radius, delta=self.race_delta)
        return pl

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        if self.race_walk and self._race_now and label.kind == "transform" and self._class_id(state) == COLLECTOR:
            ctx = getattr(self, "_teach_ctx", None)
            graphic = ctx["graphic"] if ctx is not None else None
            tiles = tuple(sorted({t for g_tiles, kind in goals if kind == "transform" for t in g_tiles}))
            if tiles and graphic is not None and all(graphic[t][SITE_HUNTER] > 0.5 for t in tiles):
                plan_map = getattr(self, "_teach_walkable", walkable)
                heading = self._planner(np.ascontiguousarray(plan_map, dtype=bool), tiles,
                                        UNIT_RADIUS[COLLECTOR]).heading(pixel_position(state, shape),
                                                                        lookahead=self.race_lookahead)
                if heading is not None:
                    self._count("race_walk")
                    return heading
        return super()._any_angle(label, state, states, goals, walkable, shape)
