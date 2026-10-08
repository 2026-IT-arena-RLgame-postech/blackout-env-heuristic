"""V17 as a relabelling teacher: V17's team composition and role tactics, labelled from the state.

V4-teach (``teach.py``) teaches V4's economy and, opt-in, a composition picked by hand (Run 16:
2 Hunters, 1 Carrier).  V17 is the strongest heuristic (``docs/heuristic_elo_20260918.md``) and
its strength *is* its composition -- 3 Hunters (one camping outside the enemy base, two hunting),
1 Carrier while the field is worth collecting, 1 Collector -- plus how each role plays.  Cloning
V17's actions directly taught the composition without the role play (Run 12: Hunters drifted to
2.9-3.6 and the model lost collection it could not replace with hunting).  This teacher labels
both: which empty Collectors should transform so the team reaches V17's role ratio, and what a
unit in each role should do, all as a function of the observation (see ``teach.py`` for why a
feed-forward learner needs that).

What changes against ``StrategicHeuristicV17``:

* composition: V17 keeps pending transformations in ``_role_of`` memory.  Here the missing roles
  (``hunter_quota`` Hunters; ``carrier_quota`` Carriers while field value >= 25, V17's rule) go
  to the empty Collectors with the shortest path to each sanctuary, as in V4-teach's
  composition labels.  A surplus is never labelled (a class only reverts by dying);
* economy: V17 scores every (unit, task) pair per second of round trip and matches greedily
  with a 1.12 bonus for each unit's previous target.  Here each unit keeps every task within
  ``SCORE_BAND`` (that same 1.12 ratio, as a band instead of memory) of its own best, minus
  tasks a teammate also wants and reaches ``EXCLUSION_MARGIN`` seconds earlier -- V4-teach's
  one-level coordination, which kept target-choice flips at V4's memory-stabilised level;
* idle collectors shadow the enemy's external storage like V17, keeping stores within
  ``IDLE_BAND`` of the best; V17's tick-rotated patrol fallback has no state-only answer and is
  invalid;
* deposits use V17's cost (travel seconds + expected raid loss before the next absorption), as
  whole storage clusters, keeping clusters within ``DEPOSIT_BAND`` seconds of the cheapest;
* Hunters: in ``mixed`` mode V17 makes the lowest-named Hunter camp.  Here the camper is the
  Hunter with the shortest path to the first camp slot (ties by unit index), so the role follows
  where the Hunters are, not which slot the learner happened to transform.  Chase targets keep
  V17's score and 0.35 threshold, unioned within ``SCORE_BAND``; a chase target a teammate reaches
  ``EXCLUSION_MARGIN`` seconds earlier is re-scored at V17's 0.5 "claimed" discount;
* movement: shortest-path progress sets with V4's threat repulsion for cargo and Carriers
  (``TeachLabeller._label``) in place of V17's threat-cost A* and stuck recovery, which depend
  on history and on a cached route.

``role_counts`` reports the composition a label set pushes toward, so a trainer can log how the
labels divide among roles.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .strategic import CARRIER, COLLECTOR, HUNTER, SITE_CARRIER
from .teach import Goal, TeachLabel, TeachLabeller
from .v17_planner import StrategicHeuristicV17

SCORE_BAND = 1.12  # V17's hysteresis bonus: tasks within this ratio of a unit's best are kept
EXCLUSION_MARGIN = 1.0  # seconds a teammate must arrive earlier to take a shared best task
DEPOSIT_BAND = 0.25  # V17 deposit cost units = seconds; about one tile for a Collector
IDLE_BAND = 0.3  # V17 idle score units; distance costs 0.15 per tile, so about two tiles
HUNT_THRESHOLD = 0.35  # StrategicHeuristicV17._hunt_plan
CARRIER_FIELD_VALUE = 25.0  # StrategicHeuristicV17._update_roles: no new Carrier below this


class StrategicHeuristicV17Teach(TeachLabeller, StrategicHeuristicV17):
    """V17's composition, values and role tactics as a memoryless set-valued action labeller."""

    campers: list[str] = []

    def teach(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, TeachLabel]:
        """Labels for every agent in ``obs`` (one team's observations, as ``act`` receives them)."""
        if not obs:
            return {}
        self._memory.clear()  # no target hysteresis: the answer depends on this state only
        ctx = self._context(obs)  # also handles the episode boundary and advances the tick
        graphic, states, shape = ctx["graphic"], ctx["states"], ctx["shape"]
        self._teach_ctx = ctx
        move_walkable = ctx["walkable"] & ~self._enemy_base_mask(graphic)
        self._teach_walkable = move_walkable

        own = ctx["own"]
        controlled = [unit.name for unit in own]
        row_for = {unit.name: unit.row for unit in own}
        transforms = self._composition_goals(controlled, row_for, states, graphic, move_walkable)

        goals: dict[str, tuple[Goal, ...]] = {}
        economic = []
        for unit in own:
            if unit.cls == HUNTER:
                continue
            if unit.holding:
                goals[unit.name] = self._deposit_goals(ctx, unit)
            elif unit.name in transforms:
                goals[unit.name] = transforms[unit.name]
            else:
                economic.append(unit)
        goals.update(self._economy_goals(ctx, economic))
        goals.update(self._hunter_goals(ctx, [unit for unit in own if unit.cls == HUNTER], move_walkable))
        return self._labels(ctx, goals, move_walkable)

    def _labels(self, ctx, goals: dict[str, tuple[Goal, ...]], move_walkable: np.ndarray) -> dict[str, TeachLabel]:
        """Each unit's goals as a label (subclasses change how units get there)."""
        states, shape = ctx["states"], ctx["shape"]
        return {
            unit.name: self._label(states[unit.row], states, goals.get(unit.name, ()), move_walkable, shape)
            for unit in ctx["own"]
        }

    def role_counts(self, labels: dict[str, TeachLabel], obs: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
        """Composition the labels lead to: current classes, with transform labels counted as the
        class they head for (the carrier sanctuary goal lies on the carrier site)."""
        sample = next(iter(obs.values()))
        states, graphic = sample["agent_states"], sample["graphic"]
        counts = {"collector": 0, "hunter": 0, "carrier": 0}
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        for j, name in enumerate(sorted(obs, key=unit_index)):
            row = unit_index(name)
            if row >= len(states) or states[row, 2] <= 0:
                row = own_rows[j]
            cls = self._class_id(states[row])
            label = labels.get(name)
            if label is not None and label.kind == "transform" and label.target is not None:
                cls = CARRIER if graphic[label.target][SITE_CARRIER] > 0.5 else HUNTER
            counts[{COLLECTOR: "collector", HUNTER: "hunter", CARRIER: "carrier"}[cls]] += 1
        return counts

    # ------------------------------------------------------------------------------ composition

    def _composition_targets(self, graphic, states, row_for) -> tuple[int, int]:
        ctx = self._teach_ctx
        carriers = self.carrier_quota if ctx["field_value"] >= CARRIER_FIELD_VALUE else 0
        return self.hunter_quota, min(1, carriers)

    # --------------------------------------------------------------------------------- economy

    def _economy_goals(self, ctx, units) -> dict[str, tuple[Goal, ...]]:
        if not units:
            return {}
        tasks, pairs, pickup = self._economy_pairs(ctx, units)
        by_unit: dict[str, list[tuple[float, int]]] = {}
        for score, name, task in pairs:
            by_unit.setdefault(name, []).append((score, task))

        def best(scored: list[tuple[float, int]]) -> list[int]:
            top = max(score for score, _ in scored)
            return [task for score, task in scored if score * SCORE_BAND >= top]

        wanted = {name: set(best(scored)) for name, scored in by_unit.items() if scored}
        goals: dict[str, tuple[Goal, ...]] = {}
        for unit in units:
            scored = [(score, task) for score, task in by_unit.get(unit.name, []) if score > 0.0]
            taken = {
                task
                for other, tasks_wanted in wanted.items() if other != unit.name
                for task in tasks_wanted
                if (unit.name, task) in pickup
                and pickup[(other, task)] + EXCLUSION_MARGIN <= pickup[(unit.name, task)]
            }
            free = [(score, task) for score, task in scored if task not in taken]
            if free:
                goals[unit.name] = tuple(
                    ((tuple(tasks[task][0]),), tasks[task][1]) for task in best(free))
            else:
                goals[unit.name] = self._idle_goals(ctx, unit)
        return goals

    def _idle_goals(self, ctx, unit) -> tuple[Goal, ...]:
        """StrategicHeuristicV17._idle_plan's storage shadowing, near-ties kept."""
        scored = []
        for index, store in enumerate(ctx["enemy_stores"]):
            if store.protected:
                continue
            field = self._store_field(ctx["walkable"], "enemy", index, store)
            incoming = sum(
                e.amount for e in ctx["enemies"] if e.slot == 1
                and np.isfinite(field[e.pos]) and field[e.pos] <= 8.0
            )
            distance = self._dist(ctx, unit.pos, store.center)
            if not math.isfinite(distance):
                continue
            scored.append((incoming * 2.0 + store.battery - 0.15 * distance, store.center))
        if not scored:
            return ()
        top = max(score for score, _ in scored)
        return tuple(((center,), "patrol_raid") for score, center in scored if score >= top - IDLE_BAND)

    def _deposit_goals(self, ctx, unit) -> tuple[Goal, ...]:
        costs = self._deposit_costs(ctx, unit)
        if not costs:
            target, can_deposit = self._storage_target(ctx["graphic"], ctx["states"][unit.row], unit.pos)
            return () if target is None else (((tuple(target),), "deposit" if can_deposit else "wait_storage"),)
        cheapest = min(cost for cost, _, _ in costs)
        return tuple((tuple(store.cells), "deposit") for cost, _, store in costs if cost <= cheapest + DEPOSIT_BAND)

    # --------------------------------------------------------------------------------- hunters

    def _hunter_goals(self, ctx, hunters, move_walkable) -> dict[str, tuple[Goal, ...]]:
        """Camper(s) first (``self.campers``, for inspection), then chasers."""
        if not hunters:
            self.campers = []
            return {}
        self.campers = []
        goals: dict[str, tuple[Goal, ...]] = {}
        chasers = list(hunters)
        claimed: set[tuple[int, int]] = set()
        slots = self._camp_slots(ctx)
        if slots and self.hunter_mode in ("camp", "mixed"):
            distance = self._distance_to(move_walkable, (slots[0],))
            ranked = sorted(hunters, key=lambda h: (float(distance[h.pos]), unit_index(h.name)))
            campers = ranked if self.hunter_mode == "camp" else ranked[:1]
            for rank, hunter in enumerate(campers):
                target, kind = self._camp_plan(ctx, hunter, rank, claimed)
                if target is not None:
                    goals[hunter.name] = (((tuple(target),), kind),)
                    if kind == "hunt":
                        claimed.add(tuple(target))
                chasers.remove(hunter)
                self.campers.append(hunter.name)

        speed = self._speed(ctx, HUNTER)
        choices = {hunter.name: self._chase_choice(ctx, hunter, claimed) for hunter in chasers}
        # arrival[name][target] = (field / speed)[target]: the same float32 division, per looked-up cell
        fields = {hunter.name: self._cached_distance_map(ctx["walkable"], hunter.pos) for hunter in chasers}
        speed32 = np.float32(speed)
        for hunter in chasers:
            chosen, kind = choices[hunter.name]
            if kind == "hunt":
                faster = {
                    target
                    for other in chasers if other.name != hunter.name and choices[other.name][1] == "hunt"
                    for target in choices[other.name][0]
                    if target in chosen
                    and float(fields[other.name][target] / speed32) + EXCLUSION_MARGIN
                    <= float(fields[hunter.name][target] / speed32)
                }
                if faster:
                    chosen, kind = self._chase_choice(ctx, hunter, claimed | faster)
            goals[hunter.name] = tuple(((target,), kind) for target in chosen)
        return goals

    def _chase_choice(self, ctx, hunter, claimed) -> tuple[tuple[tuple[int, int], ...], str]:
        """StrategicHeuristicV17._hunt_plan with near-tied chase targets kept."""
        scored, exposed = self._hunt_scores(ctx, hunter, claimed)
        top = max((score for score, _ in scored), default=0.0)
        if top > HUNT_THRESHOLD:
            targets = tuple(dict.fromkeys(
                tuple(enemy.pos) for score, enemy in scored if score * SCORE_BAND >= top))
            return targets, "hunt"
        if exposed:
            store = max(exposed, key=lambda s: s.battery + 6 * s.specials)
            return (tuple(store.center),), "patrol_defend"
        target = self._defend_patrol_target(ctx["graphic"], hunter.pos)
        return ((tuple(target),), "patrol_defend") if target is not None else ((), "idle")

