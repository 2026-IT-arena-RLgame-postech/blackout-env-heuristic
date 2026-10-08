"""V27 whose staged reserve does not back off from Hunters that are about to trade (Q7, docs/policy_v2_design_ko.md §36).

Q7 (student P16 vs its teacher V27(1.5) on the same seeds): P16's whole gain is the second race.  P16 cannot stand
still (no stop action), so its reserve Collector does not hold at 1.5 tiles but creeps to a median 0.62 tiles from
the sanctuary by decision 98; it then turns into a Hunter ~4 decisions before V27's reserve, whose walker sees the
new Hunter and turns away.  The closer stage is the gain (V27 ``reserve_stage`` 0.5 > 1.5); the arms race for the
stage distance flattens below ~0.5 because at decision 99 both sides' walkers transform and every staged reserve
*evades* the enemy Hunters (one step back) although those Hunters trade with ours on the next decision.

``trade_hold`` (default off = V27 exactly): while every enemy Hunter outside its base is within ``trade_radius``
tiles of one of our Hunters and we have at least as many Hunters (the trade on the sanctuary is pending), our empty
Collectors within ``trade_reach`` (exact race distance) of the Hunter sanctuary do not evade them:
``"hold"`` stands still, ``"advance"`` keeps walking the exact race path until ``trade_floor`` tiles and then stands.
After the trade (no enemy Hunter out) V25's own composition sends the nearest Collector in.  Observation-only.
"""

from __future__ import annotations

import math

import numpy as np

from .any_angle import UNIT_RADIUS, pixel_position
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .v27_reserve import StrategicHeuristicV27

MODES = ("off", "hold", "advance")


class StrategicHeuristicV28(StrategicHeuristicV27):
    """V27 + no evasion of Hunters whose trade on the sanctuary is pending (``trade_hold``)."""

    def __init__(self, *, trade_hold: str = "off", trade_radius: float = 1.0, trade_reach: float = 1.0,
                 trade_floor: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if trade_hold not in MODES:
            raise ValueError(f"trade_hold must be one of {MODES}, got {trade_hold!r}")
        self.trade_hold = trade_hold
        self.trade_radius = float(trade_radius)
        self.trade_reach = float(trade_reach)
        self.trade_floor = float(trade_floor)
        self.trade_held: list[str] = []  # for inspection
        self._held_xy: dict[tuple[float, float], float] = {}

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        self.trade_held = []
        self._held_xy = {}
        if self.trade_hold == "off":
            return goals
        ctx = self._teach_ctx
        shape = graphic.shape[:2]
        theirs = [pixel_position(states[e.row], shape) for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not theirs:
            return goals
        ours = [pixel_position(states[u.row], shape) for u in ctx["own"] if u.cls == HUNTER]
        if len(ours) < len(theirs):
            return goals
        if any(min(math.dist(t, o) for o in ours) > self.trade_radius for t in theirs):
            return goals
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)  # sorted (row-major)
        if not tiles:
            return goals
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        for name in controlled:
            state = states[row_for[name]]
            if self._class_id(state) != COLLECTOR or self._is_holding(state):
                continue
            d = planner.distance(pixel_position(state, shape))
            if 0.0 < d <= self.trade_reach:
                goals[name] = ((tiles, "transform"),)
                self.trade_held.append(name)
                self._held_xy[(float(state[0]), float(state[1]))] = d
        return goals

    def _evasion(self, state, states, delta):
        if self._held_xy and (float(state[0]), float(state[1])) in self._held_xy:
            return None
        return super()._evasion(state, states, delta)

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        key = (float(state[0]), float(state[1]))
        if self._held_xy and key in self._held_xy and self._class_id(state) == COLLECTOR:
            d = self._held_xy[key]
            if self.trade_hold == "advance" and d > self.trade_floor:
                tiles = tuple(sorted({t for g_tiles, kind in goals if kind == "transform" for t in g_tiles}))
                plan_map = getattr(self, "_teach_walkable", walkable)
                heading = self._planner(np.ascontiguousarray(plan_map, dtype=bool), tiles,
                                        UNIT_RADIUS[COLLECTOR]).heading(pixel_position(state, shape))
                if heading is not None:
                    self._count("trade_advance")
                    return heading
            self._count("trade_hold")
            return np.zeros(2, dtype=np.float32)
        return super()._any_angle(label, state, states, goals, walkable, shape)
