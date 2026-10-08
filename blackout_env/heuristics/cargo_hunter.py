"""Cargo hunter: Hunters that let empty Collectors go and kill every load on its way home.

The Hunters of V4-V16 chase whatever is near, and V17-V21 spend one on camping the enemy base
exit, which kills units *before* they collect.  Nothing in the pool lets the opponent fill its
hands first.  Here all three Hunters (V17's ``hunter_quota``) ignore empty Collectors and enemy
Hunters (except one camping our spawn, V17's trade) and go only for cargo and Carriers -- the
units whose death destroys points:

* **intercept, don't chase** -- a load is headed for the enemy storage nearest it by path.  The
  Hunter aims at the first cell on that shortest route it can reach no later than the carrier
  (outside the enemy base, which we cannot enter), so it also catches speed-6 Carriers it could
  never run down from behind; it closes in directly once within ``strike_tiles``.  A load that
  reaches home before any Hunter can meet it is not chased at all;
* value is the cargo (a battery's amount, 6 for a special item, +3 for a Carrier) over the
  intercept time, and a load already claimed by another Hunter counts half;
* with no load on the map the Hunters wait on the routes loads must take: one outside the enemy
  base exit (the way to its protected storage), the others at the enemy's exposed storages,
  busiest first.

The economy, Carrier and deposits are V17's.
"""

from __future__ import annotations

import math

import numpy as np

from .strategic import CARRIER, HUNTER
from .v17_planner import StrategicHeuristicV17

INTERCEPT_SLACK = 0.2  # seconds we may arrive after the load and still count as meeting it
ROUTE_TOLERANCE = 1.5  # tiles: cells this close to a shortest route count as on it


class StrategicCargoHunter(StrategicHeuristicV17):
    """V17 whose Hunters only intercept cargo and Carriers, waiting on delivery routes otherwise."""

    def __init__(self, *, strike_tiles: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.strike_tiles = float(strike_tiles)

    def _hunter_plan(self, ctx, hunter, claimed):
        best = None
        for enemy in ctx["enemies"]:
            if enemy.in_base:
                continue
            value = self._load_value(ctx, enemy)
            if value <= 0:
                continue
            meet = self._meeting_point(ctx, hunter, enemy)
            if meet is None:
                continue
            cell, seconds = meet
            score = value / (seconds + 1.0) * (0.5 if enemy.pos in claimed else 1.0)
            if best is None or score > best[0]:
                best = (score, cell, enemy.pos)
        if best is not None:
            claimed.add(best[2])  # the caller claims the returned cell, which may be the meeting point
            return best[1], "hunt"
        return self._ambush_post(ctx, hunter), "patrol_defend"

    def _load_value(self, ctx, enemy) -> float:
        if enemy.cls == HUNTER:  # trade only with a Hunter camping our spawn (V17's value 12)
            spawn = ctx["ally_spawn"]
            return 12.0 if spawn is not None and math.dist(enemy.pos, spawn) <= 7.0 else 0.0
        value = enemy.amount if enemy.slot == 1 else 6.0 if enemy.slot >= 2 else 0.0
        return value + (3.0 if enemy.cls == CARRIER else 0.0)

    def _meeting_point(self, ctx, hunter, enemy):
        """(cell, our seconds) where this Hunter meets ``enemy`` on its way to the nearest enemy
        storage, or None when it gets home first."""
        walkable = ctx["walkable"]
        ours = self._cached_distance_map(walkable, hunter.pos)
        speed = self._speed(ctx, HUNTER)
        direct = float(ours[enemy.pos])
        if not math.isfinite(direct):
            return None
        if math.dist(hunter.pos, enemy.pos) <= self.strike_tiles or enemy.cls == HUNTER:
            return enemy.pos, direct / speed
        home = None
        for index, store in enumerate(ctx["enemy_stores"]):
            field = self._store_field(walkable, "enemy", index, store)
            if math.isfinite(float(field[enemy.pos])) and (home is None or field[enemy.pos] < home[enemy.pos]):
                home = field
        if home is None:  # nowhere to deliver: an ordinary chase, if we are faster
            chase = speed > self._speed(ctx, enemy.cls, enemy=True)
            return (enemy.pos, direct / speed) if chase else None
        theirs = self._cached_distance_map(walkable, enemy.pos)
        route = (theirs + home <= float(home[enemy.pos]) + ROUTE_TOLERANCE) & np.isfinite(ours)
        route &= ~self._cached_base_mask(ctx["enemy_spawn"], ctx["shape"])
        enemy_speed = self._speed(ctx, enemy.cls, enemy=True)
        route &= ours / speed <= theirs / enemy_speed + INTERCEPT_SLACK
        if not route.any():
            return None
        cells = np.argwhere(route)
        first = cells[int(np.argmin(theirs[route]))]
        cell = (int(first[0]), int(first[1]))
        return cell, float(ours[cell]) / speed

    def _ambush_post(self, ctx, hunter):
        """Wait where loads must pass: the enemy base exit, then its exposed storages by traffic."""
        posts = []
        slots = self._camp_slots(ctx)
        if slots:
            posts.append(slots[0])
        exposed = [s for s in ctx["enemy_stores"] if not s.protected]
        loaded = [e for e in ctx["enemies"] if e.holding]
        exposed.sort(key=lambda s: -(s.battery + sum(math.dist(e.pos, s.center) <= 8.0 for e in loaded)))
        posts += [s.center for s in exposed]
        if not posts:
            return self._defend_patrol_target(ctx["graphic"], hunter.pos)
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        return posts[hunters.index(hunter.name) % len(posts)]
