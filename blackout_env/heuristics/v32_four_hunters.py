"""V31 with V19's four-Hunter allocation (V32, docs/policy_v2_design_ko.md §58).

OOD1 (§56) found the V30 lineage's one real weakness: ``hunter_quota=4`` on V30 ("h4") beats V31 by ~+18, P21e
by ~+8 and every older policy by +20..+80.  h4 always has four Hunters at decision 99; when our V30-family reserve
stays back -- the state (3 Hunters ours, 4 theirs) -- we lose the second Hunter race every time and the match ends
around -60 by 60 s; at (4, 4) it is even.  The enemy's fourth Hunter cannot be seen before decision 99, so the
answer is structural: field four Hunters ourselves.  V18/V19 did (``hunter_quota=4``) with a role mix around it
(camper + exit guard + sanctuary sentry + chaser, every Hunter sweeping the sanctuary first).  V32 = V31 + those
roles as knobs on the V22-teach goal labeller; ``hunter_quota`` itself is V17's parameter.  Defaults = V31 exactly:

* ``exit_guards`` (0): with at least ``guard_min_hunters`` Hunters, that many Hunters (nearest our base exit's
  first slot, after the sentry is set aside) hold our base exit and strike any enemy Hunter within
  ``guard_radius`` tiles of it -- V18's guard (``StrategicHeuristicV18._guard_plan``): a camping enemy Hunter is
  traded off the moment it arrives instead of killing our respawns;
* ``sweep`` (off): every Hunter except the sentry first strikes the enemy Collector nearest it that is outside its
  base, within ``sweep_radius`` tiles of the sanctuary centre and of this Hunter -- V19's sanctuary sweep
  (``StrategicHeuristicV19._sweep_target``), which catches the enemy's replacement walkers in the second race;
  only during the first ``sweep_until`` seconds of play (default: all match); ``sweep_camper=false`` leaves the
  Hunter nearest the enemy camp slot (V17's camper) on its post;
* ``cargo_transform`` (off): a Collector holding a battery stays a Hunter candidate all match (V19's
  ``transform_with_cargo``), not only during the opening race (V22's ``race_cargo``) -- a lost Hunter is
  replaced by the nearest Collector even if it carries a battery.

* ``late_hunter_quota`` (-1 = off): after ``late_after`` seconds the team refills to
  ``min(hunter_quota, max(late_hunter_quota, enemy Hunters out))`` Hunters -- four against a four-Hunter enemy, fewer
  (more economy) against one that fields fewer.  Hunters above the target are kept (a class only reverts by dying).

V19's late-game role order (camper, then exit guard, then sentry from the third Hunter) is a parameter set:
``solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2`` (V30's is sentry first).

Observation-only, as the rest of the V22-teach goals; no per-tick Python beyond a few distance lookups.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .strategic import COLLECTOR, HUNTER, SITE_HUNTER
from .teach import Goal
from .v17_planner import MATCH_SECONDS
from .v31_carrier_evade import StrategicHeuristicV31


class StrategicHeuristicV32(StrategicHeuristicV31):
    """V31 + V18's exit guard and V19's sanctuary sweep for a four-Hunter team."""

    def __init__(self, *, exit_guards: int = 0, guard_min_hunters: int = 4, guard_radius: float = 7.0,
                 sweep: bool = False, sweep_radius: float = 9.0, sweep_until: float = math.inf,
                 sweep_camper: bool = True,
                 cargo_transform: bool = False,
                 late_hunter_quota: int = -1, late_after: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.exit_guards = int(exit_guards)
        self.guard_min_hunters = int(guard_min_hunters)
        self.guard_radius = float(guard_radius)
        self.sweep = bool(sweep)
        self.sweep_radius = float(sweep_radius)
        self.sweep_until = float(sweep_until)
        self.sweep_camper = bool(sweep_camper)
        self.cargo_transform = bool(cargo_transform)
        self._composing = False
        self.late_hunter_quota = int(late_hunter_quota)
        self.late_after = float(late_after)
        self.guards: list[str] = []  # for inspection: this tick's exit guards
        self.sweepers: list[str] = []  # for inspection: Hunters sweeping the sanctuary this tick

    # ------------------------------------------------------------------ helpers

    def _sanctuary_centre(self, ctx):
        tiles = np.argwhere(ctx["graphic"][..., SITE_HUNTER] > 0.5)
        return None if not len(tiles) else (float(tiles[:, 0].mean()), float(tiles[:, 1].mean()))

    def _sentry_name(self, ctx, hunters, move_walkable):
        """The Hunter V22's ``_hunter_goals`` will post as the sentry (None if it posts none)."""
        if not self.sentry or len(hunters) < self.sentry_min_hunters:
            return None
        spawn, centre = ctx["enemy_spawn"], self._sanctuary_centre(ctx)
        if spawn is None or centre is None:
            return None
        post = self._sentry_post(move_walkable, centre, spawn)
        to_post = self._distance_to(move_walkable, (post,))
        return min(hunters, key=lambda h: (float(to_post[h.pos]), unit_index(h.name))).name

    def _guard_slots(self, ctx):
        return self._camp_slots(dict(ctx, enemy_spawn=ctx["ally_spawn"]))

    # ------------------------------------------------------------------ composition

    def _composition_targets(self, graphic, states, row_for):
        hunters, carriers = super()._composition_targets(graphic, states, row_for)
        ctx = self._teach_ctx
        if self.late_hunter_quota >= 0 and MATCH_SECONDS - ctx["remaining"] >= self.late_after:
            enemy = sum(e.cls == HUNTER for e in ctx["enemies"])
            hunters = min(hunters, max(self.late_hunter_quota, enemy))
        return hunters, carriers

    def _racing(self, ctx) -> bool:
        # V22's composition lets a battery-holding Collector walk to the sanctuary while racing (race_cargo)
        return (self._composing and self.cargo_transform) or super()._racing(ctx)

    def _composition_goals(self, controlled, row_for, states, graphic, walkable):
        self._composing = True
        try:
            return super()._composition_goals(controlled, row_for, states, graphic, walkable)
        finally:
            self._composing = False

    # ------------------------------------------------------------------ Hunters

    def _hunter_goals(self, ctx, hunters, move_walkable) -> dict[str, tuple[Goal, ...]]:
        self.guards, self.sweepers = [], []
        if not hunters or (self.exit_guards <= 0 and not self.sweep):
            return super()._hunter_goals(ctx, hunters, move_walkable)
        sentry = self._sentry_name(ctx, hunters, move_walkable)
        goals: dict[str, tuple[Goal, ...]] = {}
        rest = list(hunters)
        if self.sweep and MATCH_SECONDS - ctx["remaining"] < self.sweep_until:
            centre = self._sanctuary_centre(ctx)
            prey = [] if centre is None else [
                e for e in ctx["enemies"]
                if e.cls == COLLECTOR and not e.in_base and math.dist(e.pos, centre) <= self.sweep_radius]
            camper = None
            if not self.sweep_camper and prey:
                slots = self._camp_slots(ctx)
                others = [h for h in hunters if h.name != sentry]
                if slots and others and self.hunter_mode in ("camp", "mixed"):
                    to_camp = self._distance_to(move_walkable, (slots[0],))
                    camper = min(others, key=lambda h: (float(to_camp[h.pos]), unit_index(h.name))).name
            for hunter in hunters:
                if hunter.name in (sentry, camper) or not prey:
                    continue
                near = [(math.dist(e.pos, hunter.pos), e.pos) for e in prey]
                near = [x for x in near if x[0] <= self.sweep_radius]
                if near:
                    goals[hunter.name] = (((tuple(min(near)[1]),), "hunt"),)
                    rest.remove(hunter)
                    self.sweepers.append(hunter.name)
        if self.exit_guards > 0 and len(hunters) >= self.guard_min_hunters:
            slots = self._guard_slots(ctx)
            cands = [h for h in rest if h.name != sentry]
            if slots and cands:
                to_slot = self._distance_to(move_walkable, (slots[0],))
                cands.sort(key=lambda h: (float(to_slot[h.pos]), unit_index(h.name)))
                threats = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base
                           and math.dist(e.pos, slots[0]) <= self.guard_radius]
                for index, hunter in enumerate(cands[:self.exit_guards]):
                    if threats:
                        target = min(threats, key=lambda e: (math.dist(e.pos, hunter.pos), e.pos)).pos
                        goals[hunter.name] = (((tuple(target),), "hunt"),)
                    else:
                        goals[hunter.name] = (((slots[min(index, len(slots) - 1)],), "patrol_defend"),)
                    rest.remove(hunter)
                    self.guards.append(hunter.name)
        self.sentries = []
        if rest:
            goals.update(super()._hunter_goals(ctx, rest, move_walkable))
        return goals
