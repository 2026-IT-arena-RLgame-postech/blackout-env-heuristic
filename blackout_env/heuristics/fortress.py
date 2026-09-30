"""Fortress: a zone defence around our own base and storages, with a minimal-risk economy.

The pool's defenders are thin: V14 posts one Hunter by the home storage on V4's economy (Elo
~1320), V12/V10 turn defensive only on a late lead, and V18/V19's exit guard is one Hunter inside
an otherwise offensive planner.  Every strong policy attacks.  The fortress never sends a Hunter
into enemy ground and makes every trip of its economy as safe as it can:

* **zone Hunters** (``hunter_quota`` 2) each hold a post and engage only what comes within
  ``guard_radius`` of it: the first holds our base exit (trading with any Hunter that comes to
  camp it, and killing units that pass), the rest hold our exposed storages, richest first.
  Cargo, Carriers and Hunters come first, then empty thieves, all over the distance to them;
* **deposits** go to the protected base storage unless an exposed one is much closer
  (``exposure_cost`` 3.0; V17: 0.9) -- a zone Hunter beside an exposed storage halves its risk
  as in V17;
* **risk-free tasks only** -- field batteries, items and steals within ``danger_tiles`` of an enemy
  Hunter outside its base are not taken at all (V17 only discounts them), and steals are worth
  ``steal_weight`` 1.0 (V17: 1.8);
* **idle units go home** into our base instead of shadowing the enemy's storages.

The Carrier and movement (V6 threat-cost A*) are V17's.
"""

from __future__ import annotations

import math

from .strategic import CARRIER, HUNTER
from .v17_planner import StrategicHeuristicV17


class StrategicFortress(StrategicHeuristicV17):
    """V17 planner with zone-defending Hunters, protected deposits and Hunter-free tasks."""

    def __init__(
        self,
        *,
        hunter_quota: int = 2,
        exposure_cost: float = 3.0,
        steal_weight: float = 1.0,
        guard_radius: float = 6.0,
        danger_tiles: float = 5.0,
        **kwargs,
    ):
        super().__init__(hunter_quota=hunter_quota, exposure_cost=exposure_cost, steal_weight=steal_weight, **kwargs)
        self.guard_radius = float(guard_radius)
        self.danger_tiles = float(danger_tiles)

    # ------------------------------------------------------------------ economy

    def _economy_pairs(self, ctx, units):
        tasks, pairs, pickup = super()._economy_pairs(ctx, units)
        hunters = [e.pos for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not hunters:
            return tasks, pairs, pickup
        unsafe = {i for i, task in enumerate(tasks)
                  if min(math.dist(task[0], h) for h in hunters) <= self.danger_tiles}
        return tasks, [p for p in pairs if p[2] not in unsafe], pickup

    def _idle_plan(self, ctx, unit):
        spawn = ctx["ally_spawn"]
        if spawn is None:
            return super()._idle_plan(ctx, unit)
        return spawn, "patrol_defend"

    # ------------------------------------------------------------------ Hunters

    def _posts(self, ctx) -> list[tuple[int, int]]:
        posts = []
        exit_slots = self._camp_slots(dict(ctx, enemy_spawn=ctx["ally_spawn"]))
        if exit_slots:
            posts.append(exit_slots[0])
        exposed = sorted((s for s in ctx["ally_stores"] if not s.protected),
                         key=lambda s: -(s.battery + 6.0 * s.specials))
        posts += [s.center for s in exposed]
        return posts

    def _hunter_plan(self, ctx, hunter, claimed):
        posts = self._posts(ctx)
        if not posts:
            return self._hunt_plan(ctx, hunter, claimed)
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        post = posts[hunters.index(hunter.name) % len(posts)]
        best = None
        for enemy in ctx["enemies"]:
            if enemy.in_base or math.dist(enemy.pos, post) > self.guard_radius:
                continue
            distance = self._dist(ctx, hunter.pos, enemy.pos)
            if not math.isfinite(distance):
                continue
            value = 1.0 + (enemy.amount if enemy.slot == 1 else 6.0 if enemy.slot >= 2 else 0.0)
            value += 4.0 if enemy.cls in (CARRIER, HUNTER) else 0.0
            score = value / (distance + 1.0) * (0.5 if enemy.pos in claimed else 1.0)
            if best is None or score > best[0]:
                best = (score, enemy.pos)
        if best is not None:
            return best[1], "hunt"
        return post, "patrol_defend"
