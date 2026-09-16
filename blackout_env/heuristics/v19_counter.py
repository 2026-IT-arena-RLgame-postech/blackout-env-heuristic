"""Nineteenth-generation heuristic: a dedicated counter to V18.

V18 against itself is decided at the centre sanctuary.  All four would-be Hunters walk there as
one clump; if the other team transforms first, a single fresh Hunter kills the whole clump, that
team gets no Hunter at all and its respawns die at the base exit.  In a 90-game V18 mirror a
quarter of the games went that way and the wiped side lost every one; transforming a few ticks
earlier without a wipe decided nothing (15-15).

V19 is V18 turned towards causing that wipe:

* Every V19 Hunter first kills enemy Collectors within ``sweep_radius`` of the sanctuary, before
  taking up its V18 post.  That catches V18's clump and every replacement walker.
* One Hunter stands as a sentry on the enemy side of the sanctuary.  No V18 rule attacks a Hunter
  there (its camper ignores Hunters, its guard only covers its own exit, its hunting Hunters only
  defend near their spawn), and every Hunter V18 loses in a trade must walk past it to come back.
* Walkers head for the sanctuary tile nearest by path; V18 aims at the centre cell and always
  detours around field items.  The item-free route is taken only when it is no longer, and a
  walker that picks up cargo anyway transforms with it instead of turning back to deposit.

Measured with 64s-truncated side-swapped games against V18 (128 per variant): V18 mirror 50%,
nearest-tile routing alone 48%, plus the sweep 59-61%, plus the sentry and a 9-cell sweep 66%;
without the exit guard 55-57%.  Spreading walkers over different tiles (36%) and retreating
walkers from Hunters (20%) lost and were left out.
"""

from __future__ import annotations

import math

import numpy as np

from .strategic import BATTERY, COLLECTOR, FIRST_SPECIAL, HUNTER, SITE_HUNTER, STORAGE_ALLY, STORAGE_ENEMY
from .v18_counter import StrategicHeuristicV18


class StrategicHeuristicV19(StrategicHeuristicV18):
    """V18 with a sanctuary sweep, a sanctuary sentry and nearest-tile walker routes."""

    def __init__(
        self,
        *,
        sanctuary_sweep: bool = True,
        sweep_radius: float = 9.0,
        sentry: bool = True,
        sentry_offset: float = 2.5,
        sentry_radius: float = 7.0,
        nearest_tile: bool = True,
        transform_with_cargo: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sanctuary_sweep = bool(sanctuary_sweep)
        self.sweep_radius = float(sweep_radius)
        self.sentry = bool(sentry)
        self.sentry_offset = float(sentry_offset)
        self.sentry_radius = float(sentry_radius)
        self.nearest_tile = bool(nearest_tile)
        self.transform_with_cargo = bool(transform_with_cargo)
        self._walker_detour: dict[str, bool] = {}

    def reset(self) -> None:
        super().reset()
        self._walker_detour.clear()

    @staticmethod
    def _sanctuary(ctx):
        tiles = [(int(y), int(x)) for y, x in zip(*np.nonzero(ctx["graphic"][..., SITE_HUNTER] > 0.5))]
        if not tiles:
            return tiles, None
        centre = (sum(t[0] for t in tiles) / len(tiles), sum(t[1] for t in tiles) / len(tiles))
        return tiles, centre

    # ------------------------------------------------------------------ Hunters

    def _hunter_plan(self, ctx, hunter, claimed):
        if self.sanctuary_sweep:
            target = self._sweep_target(ctx, hunter, self.sweep_radius, near_hunter=True)
            if target is not None:
                return target, "hunt"
        if self.sentry:
            hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
            if hunters.index(hunter.name) == 1 + self.exit_guards:
                return self._sentry_plan(ctx, hunter)
        return super()._hunter_plan(ctx, hunter, claimed)

    def _sweep_target(self, ctx, hunter, radius, *, near_hunter):
        """Nearest enemy Collector within ``radius`` of the sanctuary (and of this Hunter)."""
        _, centre = self._sanctuary(ctx)
        if centre is None:
            return None
        best = None
        for enemy in ctx["enemies"]:
            if enemy.cls != COLLECTOR or enemy.in_base or math.dist(enemy.pos, centre) > radius:
                continue
            distance = math.dist(enemy.pos, hunter.pos)
            if near_hunter and distance > radius:
                continue
            if best is None or distance < best[0]:
                best = (distance, enemy.pos)
        return None if best is None else best[1]

    def _sentry_plan(self, ctx, hunter):
        target = self._sweep_target(ctx, hunter, self.sentry_radius, near_hunter=False)
        if target is not None:
            return target, "hunt"
        _, centre = self._sanctuary(ctx)
        spawn = ctx["enemy_spawn"]
        if centre is None or spawn is None:
            return self._hunt_plan(ctx, hunter, set())
        direction = (spawn[0] - centre[0], spawn[1] - centre[1])
        norm = math.hypot(*direction) or 1.0
        post = (centre[0] + direction[0] / norm * self.sentry_offset,
                centre[1] + direction[1] / norm * self.sentry_offset)
        cell = min(zip(*np.nonzero(ctx["walkable"])), key=lambda c: math.dist(c, post))
        return (int(cell[0]), int(cell[1])), "patrol_defend"

    # ------------------------------------------------------------------ walkers

    def _update_roles(self, ctx) -> None:
        previous = dict(self._role_of)
        super()._update_roles(ctx)
        if not self.transform_with_cargo:
            return
        own = ctx["own"]
        roles = dict(self._role_of)
        for i, role in previous.items():
            if role != "hunter" or i >= len(own) or own[i].cls != COLLECTOR or not own[i].holding:
                continue
            if roles.get(i) == "hunter":
                continue
            # V17 dropped this walker for holding cargo and picked a fresh Collector instead.
            replacement = next((j for j, r in roles.items() if r == "hunter"
                                and own[j].cls == COLLECTOR and previous.get(j) != "hunter"), None)
            if replacement is None:
                continue
            roles[replacement] = "collector"
            roles[i] = "hunter"
        self._role_of = roles
        self.role_assignments = {u.name: roles[i] for i, u in enumerate(own)}

    @staticmethod
    def _item_mask(graphic):
        return ((graphic[..., BATTERY] > 1e-5) | np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1)) & (
            graphic[..., STORAGE_ALLY] < 0.5) & (graphic[..., STORAGE_ENEMY] < 0.5)

    def _plan(self, ctx):
        plan = super()._plan(ctx)
        if not self.nearest_tile:
            return plan
        tiles, _ = self._sanctuary(ctx)
        if not tiles:
            return plan
        walkable = ctx["walkable"]
        items = self._item_mask(ctx["graphic"])
        for i, unit in enumerate(ctx["own"]):
            if unit.cls != COLLECTOR or self._role_of.get(i) != "hunter":
                continue
            if unit.holding and not self.transform_with_cargo:
                continue
            direct = self._cached_distance_map(walkable, unit.pos)
            direct_tile = min(tiles, key=lambda t: float(direct[t]))
            if not np.isfinite(direct[direct_tile]):
                continue
            detour_walkable = walkable & ~items
            detour_walkable[unit.pos] = walkable[unit.pos]
            for tile in tiles:
                detour_walkable[tile] = walkable[tile]
            detour = self._distance_map(detour_walkable, unit.pos)
            detour_tile = min(tiles, key=lambda t: float(detour[t]))
            use_detour = float(detour[detour_tile]) <= float(direct[direct_tile]) + 1e-6
            self._walker_detour[unit.name] = use_detour
            plan[unit.name] = (detour_tile if use_detour else direct_tile, "transform")
        return plan

    def _navigate(self, name, state, states, target, kind, walkable, shape):
        if kind == "transform" and self._last_graphic is not None and name in self._walker_detour:
            if self._walker_detour[name]:
                walkable = walkable & ~self._item_mask(self._last_graphic)
                walkable[self._to_pixel(state[:2], shape)] = True
                if target is not None:
                    walkable[target] = True
            # The route was already chosen in _plan; skip V18's unconditional detour.
            return super(StrategicHeuristicV18, self)._navigate(
                name, state, states, target, kind, walkable, shape)
        return super()._navigate(name, state, states, target, kind, walkable, shape)
