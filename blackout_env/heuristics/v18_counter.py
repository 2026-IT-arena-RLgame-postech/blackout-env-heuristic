"""Eighteenth-generation heuristic: a dedicated counter to V17.

V18 is not meant to be strong in general.  Against V17 a match is a race: the team that first
gets a Hunter camping outside the other base wins, because every respawned unit walking out to
re-transform dies at that exit.  Two fixed V17 rules decide the race:

* V17's would-be Hunters walk the shortest path to the centre sanctuary and auto-pick any
  battery they step on; a V17 unit holding cargo must deposit before transforming, so it turns
  back and loses seconds.  V18's walkers route around field items instead.
* V17's camping Hunter never attacks Hunters, and its hunting Hunters ignore ours away from
  V17's spawn.  So a second V18 Hunter can stand at our own base exit and trade with V17's
  camper the moment it arrives, while V17 never removes ours the same way.

Measured with 64s-truncated, side-swapped games against V17 (96 games per variant; the game is
chaotic, one tick of reset jitter can flip a seed): V17 mirror 50%, V17 with four Hunters 62%,
V18 75-79%; without the item-avoiding route 53%, without the exit guard 70%.  Collectors
waiting inside the base, transforming with cargo, and camping beyond V17's 7-cell defence
radius measured no gain and were left out.
"""

from __future__ import annotations

import math

import numpy as np

from .strategic import BATTERY, FIRST_SPECIAL, HUNTER, STORAGE_ALLY, STORAGE_ENEMY
from .v17_planner import StrategicHeuristicV17


class StrategicHeuristicV18(StrategicHeuristicV17):
    """V17 plus item-avoiding sanctuary routes and an exit guard that trades off V17's camper."""

    def __init__(
        self,
        *,
        hunter_quota: int = 4,
        exit_guards: int = 1,
        guard_radius: float = 7.0,
        avoid_pickup_en_route: bool = True,
        **kwargs,
    ):
        super().__init__(hunter_quota=hunter_quota, **kwargs)
        self.exit_guards = int(exit_guards)
        self.guard_radius = float(guard_radius)
        self.avoid_pickup_en_route = bool(avoid_pickup_en_route)
        self._last_graphic: np.ndarray | None = None

    def act(self, obs):
        self._last_graphic = next(iter(obs.values()))["graphic"] if obs else None
        return super().act(obs)

    # ------------------------------------------------------------------ Hunters

    def _hunter_plan(self, ctx, hunter, claimed):
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        rank = hunters.index(hunter.name)
        if rank == 0:
            return self._camp_plan(ctx, hunter, 0, claimed)
        if rank <= self.exit_guards:
            return self._guard_plan(ctx, hunter, rank - 1, claimed)
        return self._hunt_plan(ctx, hunter, claimed)

    def _guard_plan(self, ctx, hunter, index, claimed):
        """Hold our own base exit; trade with any enemy Hunter that comes to camp it."""
        slots = self._camp_slots(dict(ctx, enemy_spawn=ctx["ally_spawn"]))
        if not slots:
            return self._hunt_plan(ctx, hunter, claimed)
        threats = [
            enemy for enemy in ctx["enemies"]
            if enemy.cls == HUNTER and not enemy.in_base and enemy.pos not in claimed
            and math.dist(enemy.pos, slots[0]) <= self.guard_radius
        ]
        if threats:
            target = min(threats, key=lambda enemy: math.dist(enemy.pos, hunter.pos))
            return target.pos, "hunt"
        return slots[min(index, len(slots) - 1)], "patrol_defend"

    # ------------------------------------------------------------------ opening race

    def _navigate(self, name, state, states, target, kind, walkable, shape):
        if kind == "transform" and self.avoid_pickup_en_route and self._last_graphic is not None:
            graphic = self._last_graphic
            field_items = (
                (graphic[..., BATTERY] > 1e-5) | np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1)
            ) & (graphic[..., STORAGE_ALLY] < 0.5) & (graphic[..., STORAGE_ENEMY] < 0.5)
            detour = walkable & ~field_items
            pos = self._to_pixel(state[:2], shape)
            detour[pos] = walkable[pos]
            if target is not None:
                detour[target] = walkable[target]
                if np.isfinite(self._distance_map(detour, pos)[target]):
                    walkable = detour
        return super()._navigate(name, state, states, target, kind, walkable, shape)
