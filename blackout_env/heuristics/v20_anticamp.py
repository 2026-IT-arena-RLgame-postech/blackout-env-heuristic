"""Twentieth-generation heuristic: V17-teach that gets out past a Hunter camping its base exit.

Enemy bases are closed to us, so a Hunter parked just outside our base exit can kill every
Collector and Carrier on its way out (Hunters kill both on contact and run at 6 tiles/s against a
Collector's 4, so a unit it chases does not escape).  V17 and V19 both do this.  Against V19,
V17-teach's units died 8.2 times per 1000 unit-ticks (V17: 1.0), 81% of them within 4 tiles of our
base, with our exit camped 32% of the time -- the whole of its -38.9 margin against V19 lives here.

V20 keeps V17-teach's roles, values and goals and changes only how Collectors and Carriers move
while an enemy Hunter camps our exit (within ``CAMP_TILES`` of our base):

* **route around** -- cells within ``DANGER_TILES`` of a camping Hunter are avoided whenever the
  goal can still be reached without them (our own base is never dangerous: enemies cannot enter);
* **wait, then break out together** -- a unit inside our base whose goal cannot be reached
  safely waits there while it is alone; once two or more such units are waiting they all leave
  at once, each on its own shortest route: the camper can chase only one of them;
* **run from a chaser** -- outside the base, empty Collectors now also take V4's threat repulsion
  (V17-teach applied it only to cargo and Carriers).

Our Hunters already go for a Hunter camping our spawn first (V17's value 12 for a trade there).

Everything is a function of the observation, so ``StrategicHeuristicV20Teach`` labels DAgger data
like V17-teach, and ``StrategicHeuristicV20`` -- the same class following its own labels -- is the
heuristic that plays it.
"""

from __future__ import annotations

import numpy as np

from .strategic import HUNTER
from .teach import Goal, TeachLabel, _invalid
from .v17_teach import StrategicHeuristicV17Teach

CAMP_TILES = 3.0  # an enemy Hunter this close to our base (outside its own) is camping the exit
DANGER_TILES = 3.0  # cells this close to a camping Hunter are avoided when a route around exists
BREAKOUT_UNITS = 2  # blocked units waiting in the base that leave together


_DISCS: dict = {}


def _disc(shape: tuple[int, int], r: int, c: int, radius: float) -> np.ndarray:
    """``(rows - r) ** 2 + (cols - c) ** 2 <= radius * radius`` over ``np.mgrid`` (read-only, cached)."""
    key = (shape, r, c, radius)
    disc = _DISCS.get(key)
    if disc is None:
        rows, cols = np.mgrid[: shape[0], : shape[1]]
        disc = (rows - r) ** 2 + (cols - c) ** 2 <= radius * radius
        disc.flags.writeable = False
        if len(_DISCS) >= 65536:
            _DISCS.clear()
        _DISCS[key] = disc
    return disc


class StrategicHeuristicV20Teach(StrategicHeuristicV17Teach):
    """V17-teach plus spawn-camp counterplay: route around, break out together, run from chasers."""

    evade_empty_collectors = True

    def __init__(self, *, camp_tiles: float = CAMP_TILES, danger_tiles: float = DANGER_TILES,
                 breakout_units: int = BREAKOUT_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.camp_tiles = float(camp_tiles)
        self.danger_tiles = float(danger_tiles)
        self.breakout_units = int(breakout_units)
        self.waiting: list[str] = []  # for inspection: units held in the base this tick

    def _labels(self, ctx, goals: dict[str, tuple[Goal, ...]], move_walkable: np.ndarray) -> dict[str, TeachLabel]:
        labels = super()._labels(ctx, goals, move_walkable)
        self.waiting = []
        base = self._cached_base_mask(ctx["ally_spawn"], ctx["shape"])
        campers = self._campers(ctx, base)
        if not campers:
            return labels

        danger = self._near(ctx["shape"], campers, self.danger_tiles) & ~base
        safe = move_walkable & ~danger
        states, shape = ctx["states"], ctx["shape"]
        blocked = []
        for unit in ctx["own"]:
            label = labels[unit.name]
            if unit.cls == HUNTER or not label.valid or danger[unit.pos]:
                continue  # Hunters trade; a unit already inside the zone keeps its route (and evades)
            around = self._label(states[unit.row], states, goals.get(unit.name, ()), safe, shape)
            if around.valid:
                labels[unit.name] = around
            elif base[unit.pos]:
                blocked.append(unit.name)
        if len(blocked) < self.breakout_units:
            for name in blocked:  # alone: wait inside, where nothing can reach it
                labels[name] = _invalid(goals.get(name, ()))
            self.waiting = blocked
        return labels

    def _campers(self, ctx, base: np.ndarray) -> list[tuple[int, int]]:
        cells = np.argwhere(base)
        if not len(cells):
            return []
        return [
            enemy.pos for enemy in ctx["enemies"]
            if enemy.cls == HUNTER and not enemy.in_base
            and float(np.sqrt(((cells - np.array(enemy.pos)) ** 2).sum(1)).min()) <= self.camp_tiles
        ]

    @staticmethod
    def _near(shape: tuple[int, int], points: list[tuple[int, int]], radius: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        for r, c in points:
            mask |= _disc(tuple(shape), r, c, radius)
        return mask


class StrategicHeuristicV20(StrategicHeuristicV20Teach):
    """V20 as a player: follows its own labels (TeachLabeller.act), standing still where it has none."""
