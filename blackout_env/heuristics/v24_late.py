"""Twenty-fourth-generation heuristic: V23 whose Hunters relieve a pinned Carrier.

V23 wins the opening race and is V22 after it.  Late in the match (S1.5 branch evaluation,
``reports/value_s1/branch``; docs/policy_v2_design_ko.md section 17) its commonest losing pattern
is a frozen position: our Carrier (or a Collector) holds a battery in a pocket next to our base,
an enemy Hunter stands 1.5-2 tiles from it, and nothing moves for the rest of the match.  The
Carrier's threat repulsion keeps it out of reach and away from every storage; the enemy Hunter
does not close in; our three or four Hunters camp the enemy base exits and the sanctuary, where
nothing ever comes.  The pinned Carrier is our only unit that can still collect, the field still
holds 50-90 points, and the score is frozen -- a loss whenever we trail.  In 17 of the 120 S1.5
roots after 30 s (14%) the position looked like this (cargo holder within 3 tiles of an enemy
Hunter, three or more own Hunters), and two seconds of V17 instead of V23 gained +23.8 points on
average (12 of the 17 by >= 5; the other 103 roots: -1.2).  V17's Hunters trade with the pinning
Hunter, or its Carrier runs and the frozen position dissolves.

V24 adds (switchable for ablations):

* ``relief`` -- an enemy Hunter outside its base within ``relief_radius`` tiles of one of our
  cargo holders or our Carrier is struck by our nearest Hunter (whatever its role: camper,
  sentry or chaser) if it gets there within ``relief_seconds``.  Hunter against Hunter kills
  both: theirs respawns in their base, ours as a Collector next to our storages -- in a frozen
  late game that is a unit back on the economy -- and the cargo is free to walk to a storage.
  At most one of our Hunters per enemy Hunter; only while we have ``relief_min_hunters`` Hunters,
  and only from ``late_after`` seconds on: before that the Hunters' denial decides the match
  (in the first 10-30 s a relief trade lost whole matches in the simulator).
* ``reenter`` -- deposits fire when a unit's centre *enters* a storage region
  (``MapRegion.OnUnitEnter``).  A unit that picks a battery up while standing inside our storage
  (a field battery touching the storage edge, picked up in the same tick as its deposit) is on
  its deposit goal and its label says "interacting": it stood there for the rest of the match
  (S1.5 root 950506_B: our Carrier from 17 s to the end).  A cargo holder standing on a storage
  tile that is one of its deposit goals steps out to a neighbouring cell outside that storage;
  from there its deposit goal walks it back in, and the entry deposits.

* ``base_exit`` -- after ``late_after``, an empty Collector inside our own base takes no threat
  repulsion (V20's "run from a chaser"): enemies cannot enter the base, and when it does walk
  into the Hunter camping our exit it loses nothing -- it respawns on the spawn it left.  With
  repulsion, four Collectors stood on the base edge facing the camper for the rest of the match.
* ``dash`` (off by default) -- after ``late_after``, a cargo holder within ``relief_radius`` of an
  enemy Hunter that no relief is coming for walks its route without repulsion (V17's Carrier ran
  and died in the S1.5 branches, which also unfroze the position).  Alone it gains in the late-
  root test, but with ``relief`` nothing: it stays off.

Late-game measurements (``examples/sim_late_roots_eval.py``, 286 roots after 30 s, V24 - V23
lead at +120 s): all rules +1.9+-0.6; alone: relief +1.7, dash +0.9, base_exit +0.2, reenter +0.1.

Every rule is a function of the observation (as V21-V23): a fresh instance plays the same moves.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .any_angle import pixel_position
from .strategic import CARRIER, COLLECTOR, HUNTER
from .teach import Goal, TeachLabel
from .v17_planner import MATCH_SECONDS
from .v23_opening import StrategicHeuristicV23

RELIEF_RADIUS = 4.0  # tiles: an enemy Hunter this close to our cargo / Carrier pins it
RELIEF_SECONDS = 6.0  # our Hunter must reach it within this many seconds
RELIEF_MIN_HUNTERS = 2  # own Hunters needed before one is spent on a trade
LATE_AFTER = 30.0  # seconds of play before the late-game rules (relief, base_exit, dash) start


class StrategicHeuristicV24(StrategicHeuristicV23):
    """V23 plus late-game rules (``relief``, ``reenter``, ``base_exit``; ``dash`` off)."""

    def __init__(self, *, relief: bool = True, reenter: bool = True, base_exit: bool = True, dash: bool = False, relief_radius: float = RELIEF_RADIUS,
                 relief_seconds: float = RELIEF_SECONDS, relief_min_hunters: int = RELIEF_MIN_HUNTERS,
                 late_after: float = LATE_AFTER, **kwargs):
        super().__init__(**kwargs)
        self.relief = bool(relief)
        self.reenter = bool(reenter)
        self.base_exit = bool(base_exit)
        self.dash = bool(dash)
        self.dashing: list[str] = []  # for inspection: pinned cargo holders walking through this tick
        self._no_evade_xy: set[tuple[float, float]] = set()
        self.reentering: list[str] = []  # for inspection: cargo holders stepping out of a storage this tick
        self.relief_radius = float(relief_radius)
        self.relief_seconds = float(relief_seconds)
        self.relief_min_hunters = int(relief_min_hunters)
        self.late_after = float(late_after)
        self.relievers: list[str] = []  # for inspection: Hunters sent to trade this tick

    # ------------------------------------------------------------------ relief

    def _pinning_hunters(self, ctx) -> list:
        """Enemy Hunters outside their base within ``relief_radius`` of our cargo or Carrier,
        nearest pin first."""
        guarded = [u for u in ctx["own"] if u.holding or u.cls == CARRIER]
        if not guarded:
            return []
        pins = []
        for enemy in ctx["enemies"]:
            if enemy.cls != HUNTER or enemy.in_base:
                continue
            gap = min(math.dist(enemy.pos, u.pos) for u in guarded)
            if gap <= self.relief_radius:
                pins.append((gap, enemy.row, enemy))
        return [e for _, _, e in sorted(pins, key=lambda p: (p[0], p[1]))]

    def _relief_goals(self, ctx, hunters) -> dict[str, tuple[Goal, ...]]:
        self.relievers = []
        if not self.relief or len(hunters) < self.relief_min_hunters:
            return {}
        if MATCH_SECONDS - ctx["remaining"] < self.late_after:
            return {}
        pins = self._pinning_hunters(ctx)
        if not pins:
            return {}
        speed = self._speed(ctx, HUNTER)
        free = list(hunters)
        goals: dict[str, tuple[Goal, ...]] = {}
        for enemy in pins:
            best = None
            for hunter in free:
                t = float(self._cached_distance_map(ctx["walkable"], hunter.pos)[enemy.pos]) / speed
                if math.isfinite(t) and t <= self.relief_seconds:
                    key = (t, unit_index(hunter.name))
                    if best is None or key < best[0]:
                        best = (key, hunter)
            if best is None:
                continue
            hunter = best[1]
            goals[hunter.name] = (((tuple(enemy.pos),), "hunt"),)
            free.remove(hunter)
            self.relievers.append(hunter.name)
        return goals

    def _hunter_goals(self, ctx, hunters, move_walkable) -> dict[str, tuple[Goal, ...]]:
        relief = self._relief_goals(ctx, hunters)
        if not relief:
            return super()._hunter_goals(ctx, hunters, move_walkable)
        rest = [h for h in hunters if h.name not in relief]
        goals = super()._hunter_goals(ctx, rest, move_walkable)
        goals.update(relief)
        return goals

    # ------------------------------------------------------------------ re-entering a storage

    def _labels(self, ctx, goals: dict[str, tuple[Goal, ...]], move_walkable: np.ndarray) -> dict[str, TeachLabel]:
        self._mark_no_evasion(ctx)
        labels = super()._labels(ctx, goals, move_walkable)
        self.reentering = []
        if not self.reenter:
            return labels
        states, shape = ctx["states"], ctx["shape"]
        for unit in ctx["own"]:
            if not unit.holding:
                continue
            inside = [tiles for tiles, kind in goals.get(unit.name, ()) if kind == "deposit" and unit.pos in tiles]
            if not inside:
                continue
            region = set(inside[0])
            r, c = unit.pos
            exits = tuple(
                (r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]
                and move_walkable[r + dr, c + dc] and (r + dr, c + dc) not in region
                and (not (dr and dc) or (move_walkable[r, c + dc] and move_walkable[r + dr, c]))
            )
            if not exits:
                continue
            here = pixel_position(states[unit.row], shape)
            nearest = min(exits, key=lambda t: (math.dist(t, here), t))  # one cell: a union could cancel
            label = self._label(states[unit.row], states, (((nearest,), "deposit"),), move_walkable, shape)
            if label.valid:
                labels[unit.name] = label
                self.reentering.append(unit.name)
        return labels

    # ------------------------------------------------------------------ when not to evade

    def _mark_no_evasion(self, ctx) -> None:
        """Units whose threat repulsion is switched off this tick (``base_exit``, ``dash``)."""
        self._no_evade_xy = set()
        self.dashing = []
        states = ctx["states"]
        late = MATCH_SECONDS - ctx["remaining"] >= self.late_after
        if self.base_exit and late:
            base = self._cached_base_mask(ctx["ally_spawn"], ctx["shape"])
            for unit in ctx["own"]:
                if unit.cls == COLLECTOR and not unit.holding and base[unit.pos]:
                    self._no_evade_xy.add((float(states[unit.row][0]), float(states[unit.row][1])))
        if self.dash and late and not self.relievers:
            hunters = [e.pos for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
            for unit in ctx["own"]:
                if unit.holding and hunters and min(math.dist(h, unit.pos) for h in hunters) <= self.relief_radius:
                    self._no_evade_xy.add((float(states[unit.row][0]), float(states[unit.row][1])))
                    self.dashing.append(unit.name)

    def _evasion(self, state, states, delta):
        if self._no_evade_xy and (float(state[0]), float(state[1])) in self._no_evade_xy:
            return None
        return super()._evasion(state, states, delta)
