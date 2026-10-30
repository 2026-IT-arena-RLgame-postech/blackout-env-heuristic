"""V34: V33 for the balance patch (max 2 living Hunters per team, spawn-side walls; docs: reports/bp).

Post-patch matches are decided in the first ~20 s by the Hunter war.  Both sides' two race walkers reach the
sanctuary on the same tick (~4 s) and trade Hunter for Hunter; the side whose *next* Collector enters first then owns
the sanctuary and kills ~10 Collectors.  So the count of Collectors committed near the sanctuary decides, and the
Carrier made at 0.4 s is one walker fewer (it can never become a Hunter).  New knobs:

* ``carrier_delay`` (s, 0 = off): no Carrier before this match time, unless ``carrier_on_monopoly`` and we hold a
  Hunter monopoly (a Hunter of ours out, none of theirs out of base) -- the would-be Carrier joins the Hunter war.
* ``reserve_count`` may be 3 (V27's staged reserve walkers).
* ``cap_sync`` (off): the sanctuary transforms a Collector only when it *enters* the region and only while the team
  has fewer than 2 Hunters.  A reserve walker that steps in on the same tick as our race walkers is refused (they
  are still alive) and then stands inside as a Collector until it is killed -- with stacked walkers that loses the
  second race (and gives team B, whose units move after A's, a large edge).  With ``cap_sync`` at most
  ``2 - our Hunters`` of our Collectors may come within ``cap_enter`` tiles of the sanctuary per decision (nearest
  first); the others hold.  A Collector already inside without having transformed steps out to re-enter.
* ``storm_units`` (0 = off): after the opening (``storm_after`` s), while the enemy has more Hunters out of base than
  we have, up to this many empty Collectors (nearest the sanctuary first) are *committed* walkers to the Hunter
  sanctuary: no threat evasion, plain-map path.  A Collector that reaches the sanctuary turns into a Hunter on entry
  and trades with the enemy sentry next to it -- a Collector for a Hunter, which breaks the enemy's monopoly.

Defaults = V33 exactly.
"""

from __future__ import annotations

import numpy as np

from ..env.constants import MAX_HUNTERS, unit_index
from .any_angle import UNIT_RADIUS, pixel_position, world_heading
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .v17_planner import MATCH_SECONDS
from .v33_specials import StrategicHeuristicV33


class StrategicHeuristicV34(StrategicHeuristicV33):
    def __init__(self, *, carrier_delay: float = 0.0, carrier_on_monopoly: bool = True, cap_sync: bool = False,
                 cap_enter: float = 0.3, storm_units: int = 0, storm_after: float = 6.0, **kwargs):
        super().__init__(**kwargs)
        self.storm_units = int(storm_units)
        self.storm_after = float(storm_after)
        self.cap_sync = bool(cap_sync)
        self.cap_enter = float(cap_enter)
        self._cap_hold: set[tuple[float, float]] = set()
        self._cap_out: dict[tuple[float, float], np.ndarray] = {}
        self._cap_allow: dict[tuple[float, float], int] = {}
        self.carrier_delay = float(carrier_delay)
        self.carrier_on_monopoly = bool(carrier_on_monopoly)

    def _composition_targets(self, graphic, states, row_for):
        hunters, carriers = super()._composition_targets(graphic, states, row_for)
        if carriers and self.carrier_delay > 0:
            ctx = self._teach_ctx
            if MATCH_SECONDS - ctx["remaining"] < self.carrier_delay:
                ours = sum(u.cls == HUNTER for u in ctx["own"])
                theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
                if not (self.carrier_on_monopoly and ours > 0 and theirs == 0):
                    carriers = 0
        return hunters, carriers

    # ------------------------------------------------------------------ cap_sync

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        if self.storm_units > 0:
            self._storm(controlled, row_for, states, graphic, walkable, goals)
        self._cap_hold, self._cap_out = set(), {}
        if not self.cap_sync:
            return goals
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return goals
        shape = graphic.shape[:2]
        tile_set = set(tiles)
        hunters = sum(self._class_id(states[row_for[n]]) == HUNTER for n in controlled)
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        centre = np.mean(np.asarray(tiles, np.float32), axis=0)
        near = []
        for n in controlled:
            st = states[row_for[n]]
            if self._class_id(st) != COLLECTOR or n not in goals:
                continue
            if not any(kind == "transform" and set(t) & tile_set for t, kind in goals[n]):
                continue
            p = pixel_position(st, shape)
            cell = (int(np.floor(p[0] + 0.5)), int(np.floor(p[1] + 0.5)))  # cell centres at integers
            xy = (float(st[0]), float(st[1]))
            if cell in tile_set:  # inside without having transformed: step out to re-enter
                away = np.asarray(p, np.float32) - centre
                norm = float(np.hypot(*away))
                self._cap_out[xy] = away / norm if norm > 1e-6 else np.array([1.0, 0.0], np.float32)
                continue
            d = planner.distance(p)
            if np.isfinite(d) and d <= self.cap_enter:
                near.append((d, unit_index(n), xy))
        slots = max(0, MAX_HUNTERS - hunters)
        # stacked walkers share a position: count, per position, how many of those there may still move
        self._cap_allow = {}
        for k, (_, _, xy) in enumerate(sorted(near)):
            self._cap_allow[xy] = self._cap_allow.get(xy, 0) + (1 if k < slots else 0)
            if k >= slots:
                self._cap_hold.add(xy)
        return goals

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        if (self._cap_hold or self._cap_out) and label.kind == "transform" and self._class_id(state) == COLLECTOR:
            xy = (float(state[0]), float(state[1]))
            if xy in self._cap_hold:
                if self._cap_allow.get(xy, 0) > 0:
                    self._cap_allow[xy] -= 1  # one of a stack that may enter
                else:
                    self._count("cap_hold")
                    return np.zeros(2, dtype=np.float32)
            out = self._cap_out.get(xy)
            if out is not None:
                self._count("cap_out")
                return world_heading(float(out[0]), float(out[1]))
        return super()._any_angle(label, state, states, goals, walkable, shape)

    # ------------------------------------------------------------------ storm

    def _storm(self, controlled, row_for, states, graphic, walkable, goals):
        ctx = self._teach_ctx
        if MATCH_SECONDS - ctx["remaining"] < self.storm_after:
            return
        ours = sum(u.cls == HUNTER for u in ctx["own"])
        theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
        if theirs <= ours:
            return
        tiles = self._site_tiles(graphic, SITE_HUNTER, walkable)
        if not tiles:
            return
        tile_set = set(tiles)
        shape = graphic.shape[:2]
        going = [n for n in controlled if n in goals
                 and any(kind == "transform" and set(t) & tile_set for t, kind in goals[n])]
        need = self.storm_units - len(going)
        if need <= 0:
            return
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        cands = []
        for n in controlled:
            st = states[row_for[n]]
            if n in going or self._class_id(st) != COLLECTOR or self._is_holding(st):
                continue
            d = planner.distance(pixel_position(st, shape))
            if np.isfinite(d):
                cands.append((d, unit_index(n), n))
        for _, _, n in sorted(cands)[:need]:
            goals[n] = ((tiles, "transform"),)
            if n not in self.committed:
                self.committed.append(n)
                st = states[row_for[n]]
                self._committed_xy.add((float(st[0]), float(st[1])))
            self._count("storm")
