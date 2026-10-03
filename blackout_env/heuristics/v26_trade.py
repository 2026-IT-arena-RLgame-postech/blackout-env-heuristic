"""V25 whose Hunters trade themselves off against the enemy's (P15, docs/policy_v2_design_ko.md §30).

Combat (``UnitInteractionSystem.ResolveCombat``, ``sim/core.py _combat``): a Hunter beats every
class including Hunters, so two Hunters that touch both die (``Die(null)``: no killer) and respawn
at once as Collectors at their spawns -- no cooldown, held items destroyed, no points for kills.
The Hunter sanctuary is not a unique-instance one: any number of Collectors of either team can
transform there, and a Collector that enters while an enemy Hunter stands on it transforms first
(``ChangePos`` -> ``OnUnitEnter``) and trades on the same move.

The idea (user, P15): when we lose the race the enemy Hunter hunts our Collectors; a late Hunter
of ours that seeks it out and annihilates with it removes that advantage.  V17's Hunters value an
enemy Hunter only near our spawn (12) or next to exposed storage (4) and the single Hunter is a
camper that ignores Hunters, so V25 rarely trades on purpose.

Rules (all off = V25 exactly, ``tests/test_v26_trade.py``):

* ``late_trade`` -- each enemy Hunter outside its base is matched (greedy, shortest path first) to
  one of our Hunters, which hunts it instead of its V17/V22 goal.  ``trade_mode``: ``"parity"``
  only while we do not have more Hunters than they do (the race-lost / deficit situation),
  ``"always"`` whenever an enemy Hunter is out.  ``trade_until`` seconds into the match (inf = any
  time; e.g. 20 = opening only).  ``trade_radius`` tiles of path at most (inf = any distance).
* ``trade_walk_reach`` -- while we have no Hunter and an enemy Hunter is out, the Collector
  (no special item) with the shortest exact path to the sanctuary within this many tiles is sent
  in and committed (no threat evasion, no Hunter-free detour) when it gets there before any
  enemy Hunter running straight at it could touch it -- V25's ``race_commit`` beyond its 1-tile
  reach, one walker at a time.  0 = off.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .any_angle import UNIT_RADIUS, pixel_position
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .v17_planner import MATCH_SECONDS
from .v22_teacher import CONTACT_RADIUS
from .v25_race import StrategicHeuristicV25

TRADE_MODES = ("parity", "always")


class StrategicHeuristicV26(StrategicHeuristicV25):
    """V25 + Hunter-for-Hunter trades (``late_trade``) + far commit of a walker (``trade_walk_reach``)."""

    def __init__(self, *, late_trade: bool = True, trade_mode: str = "parity", trade_until: float = math.inf,
                 trade_radius: float = math.inf, trade_walk_reach: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if trade_mode not in TRADE_MODES:
            raise ValueError(f"trade_mode {trade_mode!r} not in {TRADE_MODES}")
        self.late_trade = bool(late_trade)
        self.trade_mode = trade_mode
        self.trade_until = float(trade_until)
        self.trade_radius = float(trade_radius)
        self.trade_walk_reach = float(trade_walk_reach)
        self.traders: list[str] = []  # for inspection: our Hunters sent at enemy Hunters this tick
        self.far_committed: list[str] = []  # for inspection: walkers committed by trade_walk_reach

    @staticmethod
    def _enemy_hunters(ctx) -> list:
        return [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]

    def _hunter_goals(self, ctx, hunters, move_walkable):
        goals = super()._hunter_goals(ctx, hunters, move_walkable)
        self.traders = []
        if not self.late_trade or not hunters:
            return goals
        if MATCH_SECONDS - ctx["remaining"] > self.trade_until:
            return goals
        enemies = self._enemy_hunters(ctx)
        if not enemies:
            return goals
        if self.trade_mode == "parity" and len(hunters) > sum(e.cls == HUNTER for e in ctx["enemies"]):
            return goals
        pairs = []
        for h in hunters:
            field = self._cached_distance_map(ctx["walkable"], h.pos)
            for e in enemies:
                d = float(field[e.pos])
                if math.isfinite(d) and d <= self.trade_radius:
                    pairs.append((d, unit_index(h.name), h.name, e.pos))
        used_h, used_e = set(), set()
        for _, _, name, pos in sorted(pairs):
            if name in used_h or pos in used_e:
                continue
            used_h.add(name)
            used_e.add(pos)
            goals[name] = (((tuple(pos),), "hunt"),)
            self.traders.append(name)
        return goals

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        goals = super()._composition_goals(controlled, row_for, states, graphic, walkable)
        self.far_committed = []
        if self.trade_walk_reach <= self.commit_reach:
            return goals
        ctx = self._teach_ctx
        if any(u.cls == HUNTER for u in ctx["own"]):
            return goals
        hunters = self._enemy_hunters(ctx)
        if not hunters:
            return goals
        tiles = tuple(sorted((int(y), int(x)) for y, x in np.argwhere((graphic[..., SITE_HUNTER] > 0.5) & walkable)))
        if not tiles:
            return goals
        if any(kind == "transform" and set(t) & set(tiles) for g in goals.values() for t, kind in g):
            return goals  # a walker is already on its way in
        shape = graphic.shape[:2]
        planner = self._planner(np.ascontiguousarray(walkable, dtype=bool), tiles, UNIT_RADIUS[COLLECTOR])
        v_me = self._speed(ctx, COLLECTOR)
        v_them = self._speed(ctx, HUNTER, enemy=True)
        contact = CONTACT_RADIUS[HUNTER] * ctx["enemy_size"] + CONTACT_RADIUS[COLLECTOR] * ctx["own_size"]
        best = None
        for name in controlled:
            state = states[row_for[name]]
            if self._class_id(state) != COLLECTOR or name in self.committed or float(state[5:9].max()) >= 0.5:
                continue
            here = pixel_position(state, shape)
            d = planner.distance(here)
            if not (0.0 < d <= self.trade_walk_reach):
                continue
            arrive = d / v_me + self.commit_margin
            caught = min((math.hypot(*np.subtract(pixel_position(states[e.row], shape), here)) - contact) / (v_them + v_me)
                         for e in hunters)
            if arrive < caught and (best is None or (d, unit_index(name)) < best[:2]):
                best = (d, unit_index(name), name, state)
        if best is not None:
            _, _, name, state = best
            goals[name] = ((tiles, "transform"),)
            self.committed.append(name)
            self._committed_xy.add((float(state[0]), float(state[1])))
            self.far_committed.append(name)
        return goals
