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
caught.  ``race_commit`` is the defence against that: an own Collector within ``commit_reach``
tiles of the Hunter sanctuary (exact distance to the region) that enters before any enemy Hunter
running straight at it could touch it is sent in and committed (no evasion) -- once a Hunter, a
contact is a trade, not three lost walkers.

``race_walk`` changes nothing else: while ``_racing`` a Collector whose label is a transform
toward the Hunter sanctuary heads along ``RacePlanner.heading`` on the planning map
(the same map V23's controller uses, field items not blocked: ``race_cargo`` keeps a walker
walking with a battery).  Every rule stays a function of the observation.
"""

from __future__ import annotations

import math

import numpy as np

from .any_angle import UNIT_RADIUS, pixel_position
from . import _shared
from .race_path import RacePlanner
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .v22_teacher import CONTACT_RADIUS
from .v23_opening import StrategicHeuristicV23



def _gap(a, b) -> float:
    """``math.hypot(*np.subtract(a, b))`` for two (row, col) float tuples, without the array."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class StrategicHeuristicV25(StrategicHeuristicV23):
    """V23 + exact race walk (``race_walk``) + last-step commit (``race_commit``);
    ``race_lookahead`` picks race headings by an emulated move (no gain measured, off)."""

    def __init__(self, *, race_walk: bool = True, race_lookahead: bool = False, race_delta: float = 2e-3,
                 race_commit: bool = True, commit_reach: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.race_commit = bool(race_commit)
        self.commit_reach = float(commit_reach)
        self.last_step: list[str] = []  # for inspection: walkers committed by race_commit this tick
        self.race_walk = bool(race_walk)
        self.race_lookahead = bool(race_lookahead)
        self.race_delta = float(race_delta)
        self._planners: dict = {}

    def _planner(self, walkable: np.ndarray, tiles, radius: float) -> RacePlanner:
        memo = self._tick_memo()
        mkey = ("planner", id(walkable), tiles, radius)
        hit = memo.get(mkey)
        if hit is not None and hit[0] is walkable:
            return hit[1]
        pl = self._planner_for(walkable, tiles, radius)
        memo[mkey] = (walkable, pl)
        return pl

    def _planner_for(self, walkable: np.ndarray, tiles, radius: float) -> RacePlanner:
        key = (walkable.shape, walkable.tobytes(), tiles, radius)
        pl = self._planners.get(key)
        if pl is None:
            if len(self._planners) > 16:
                self._planners.clear()
            shared_key = key + (self.race_delta,)
            pl = _shared.RACE_PLANNERS.get(shared_key)
            if pl is None:  # immutable once built: one per map, goal and radius for the whole process
                pl = _shared.RACE_PLANNERS.put(shared_key, RacePlanner(walkable, tiles, radius, delta=self.race_delta))
            self._planners[key] = pl
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

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        self.last_step = []
        if not self.race_commit:
            return goals
        ctx = self._teach_ctx
        hunters = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not hunters:
            return goals
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)  # sorted (row-major)
        if not tiles:
            return goals
        shape = graphic.shape[:2]
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        v_me = self._speed(ctx, COLLECTOR)
        v_them = self._speed(ctx, HUNTER, enemy=True)
        contact = CONTACT_RADIUS[HUNTER] * ctx["enemy_size"] + CONTACT_RADIUS[COLLECTOR] * ctx["own_size"]
        for name in controlled:
            state = states[row_for[name]]
            if self._class_id(state) != COLLECTOR or name in self.committed or float(state[5:9].max()) >= 0.5:
                continue  # not a Collector, already committed, or holding a special
            here = pixel_position(state, shape)
            d = planner.distance(here)
            if not (0.0 < d <= self.commit_reach):
                continue
            arrive = d / v_me + self.commit_margin
            caught = min((_gap(pixel_position(states[e.row], shape), here) - contact) / (v_them + v_me)
                         for e in hunters)
            if arrive < caught:
                goals[name] = ((tiles, "transform"),)
                self.committed.append(name)
                self._committed_xy.add((float(state[0]), float(state[1])))
                self.last_step.append(name)
        return goals
