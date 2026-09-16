"""Seventeenth-generation heuristic: a whole-team planner built around denial.

A BlackOut match is decided in its first minute.  The ~200 battery points spawn once and never
respawn, so the field is empty within ~40s and scores barely move after the third absorption.
Units that carry cargo die to a Hunter on contact and the cargo is destroyed, while the
V1-V16 heuristics steer empty Collectors straight past Hunters.  V17 therefore spends three of
its five units on Hunters and one on the Carrier:

* One Hunter camps just outside the enemy base (the 4x4 corner square around its spawn, whose
  floor we cannot enter) and strikes whatever comes within reach: units leaving to collect,
  respawned units, and returning cargo headed for the protected home storage.
* The other Hunters hunt cargo and Carriers across the map, weighting value against intercept
  time, and trade with an enemy Hunter only when it blocks our spawn or exposed storage.
* The economy is planned as one team assignment per tick: field batteries valued per second of
  round trip, steals only when we arrive before absorption, speed items valued by how long the
  effect lasts, and deposits that weigh travel time against the risk of being raided before
  the next absorption.

Movement reuses V6's threat-cost A* and V1's stuck recovery.  Measured choices (64s-truncated
side-swapped gauntlets vs V1-V16): three Hunters in ``mixed`` mode won 96-98% of games, one
Hunter hunting 66%, and every "avoid Hunters by blocking their surroundings" navigation variant
lost more than it saved (20-27%), because stalled Collectors forfeit the race.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..env.constants import unit_index
from .risk_path import StrategicHeuristicV6
from .strategic import (
    ABSORPTION_INTERVAL_SECONDS, BATTERY, CARRIER, COLLECTOR, FIRST_SPECIAL, HUNTER,
    SITE_CARRIER, SITE_HUNTER, SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY, WALL,
)

MATCH_SECONDS = 420.0
BASE_SPEED = {COLLECTOR: 4.0, HUNTER: 6.0, CARRIER: 6.0}
BUFF_SPEED, DEBUFF_SPEED, BUFF_SIZE, DEBUFF_SIZE = 9, 10, 11, 12


@dataclass
class _Unit:
    name: str | None
    row: int
    pos: tuple[int, int]
    cls: int
    slot: int  # 0 none, 1 battery, 2.. specials in channel order
    amount: float
    in_base: bool = False

    @property
    def holding(self) -> bool:
        return self.slot != 0


@dataclass
class _Store:
    cells: list[tuple[int, int]]
    center: tuple[int, int]
    protected: bool
    battery: float = 0.0
    specials: int = 0
    free_battery_capacity: int = 0
    free_empty_tiles: int = 0


class StrategicHeuristicV17(StrategicHeuristicV6):
    """Team planner with absorption-aware deposits, raids, denial hunting and item timing."""

    def __init__(
        self,
        *,
        carrier_quota: int = 1,
        hunter_quota: int = 3,
        exposure_cost: float = 0.9,
        steal_weight: float = 1.8,
        special_weights: tuple[float, float, float, float] = (0.45, 0.8, 0.12, 0.12),
        hunter_mode: str = "mixed",
        camp_engage_radius: float = 4.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.carrier_quota = int(carrier_quota)
        self.hunter_quota = int(hunter_quota)
        self.exposure_cost = float(exposure_cost)
        self.steal_weight = float(steal_weight)
        self.special_weights = tuple(float(w) for w in special_weights)
        self._role_of: dict[int, str] = {}
        if hunter_mode not in ("hunt", "camp", "mixed"):
            raise ValueError(f"unknown hunter_mode {hunter_mode!r}")
        self.hunter_mode = hunter_mode
        self.camp_engage_radius = float(camp_engage_radius)
        self._storage_fields: dict[tuple[str, int], np.ndarray] = {}
        # Per-tick derived masks and per-episode component indices; see _per_tick/_components_index.
        self._tick_cache: dict[str, tuple[int, np.ndarray, object]] = {}
        self._cell_index: dict[int, tuple] = {}
        self._store_flags: dict[int, tuple[bool, bool]] = {}  # storage channel -> (buff, debuff)
        self._base_masks: dict[tuple, np.ndarray] = {}
        self.role_assignments: dict[str, str] = {}
        self.plan: dict[str, tuple[tuple[int, int] | None, str]] = {}

    def reset(self) -> None:
        super().reset()
        self._role_of.clear()
        self._storage_fields.clear()
        self._tick_cache.clear()
        self._cell_index.clear()
        self.role_assignments.clear()
        self.plan.clear()

    # ------------------------------------------------------------------ observation model

    def _parse_unit(self, name, row, state, shape, pos=None) -> _Unit:
        items = state[4:9].tolist()  # item slots 1..5; slot 0 (state[3]) means empty-handed
        peak = max(items)
        slot = 1 + items.index(peak) if peak > 1e-5 else 0
        return _Unit(
            name=name, row=row, pos=self._to_pixel(state[:2], shape) if pos is None else pos,
            cls=self._class_id(state), slot=slot, amount=items[0] * 15.0 if slot == 1 else 0.0,
        )

    @staticmethod
    def _state_pixels(states, shape) -> list[tuple[int, int]]:
        """``_to_pixel`` for every row at once (same float32 rounding, one array allocation)."""
        h, w = shape
        pixels = np.array(
            [((1.0 - y) * 0.5 * h - 0.5, (x + 1.0) * 0.5 * w - 0.5)
             for x, y in states[:, :2].tolist()], dtype=np.float32).tolist()
        return [(min(h - 1, max(0, int(round(py)))), min(w - 1, max(0, int(round(px)))))
                for py, px in pixels]

    def _per_tick(self, name, graphic, build):
        """Memoise ``build(graphic)`` for the current decision tick (read-only results)."""
        entry = self._tick_cache.get(name)
        if entry is not None and entry[0] == self._tick and entry[1] is graphic:
            return entry[2]
        value = build(graphic)
        self._tick_cache[name] = (self._tick, graphic, value)
        return value

    def _special_mask(self, graphic) -> np.ndarray:
        return self._per_tick(
            "special", graphic, lambda g: np.any(g[..., FIRST_SPECIAL:] > 0.5, axis=-1))

    def _stores(self, graphic, channel, protected_mask) -> list[_Store]:
        components = self._cached_components(graphic[..., channel] > 0.5)
        if not components:
            self._store_flags[channel] = (False, False)
            return []
        ys, xs, starts, centers = self._components_index(components)
        block = graphic[ys, xs]  # every storage cell of this team, all channels
        # round() and rint() both round half to even; all sums below are small integers.
        amounts = np.rint(block[:, BATTERY].astype(np.float64) * 15.0)
        is_special = (block[:, FIRST_SPECIAL:] > 0.5).any(axis=1)
        plain = ~is_special
        battery = np.add.reduceat(amounts, starts).tolist()
        specials = np.add.reduceat(is_special.astype(np.int64), starts).tolist()
        capacity = np.add.reduceat(np.where(plain, 10.0 - amounts, 0.0), starts).tolist()
        empty = np.add.reduceat((plain & (amounts == 0.0)).astype(np.int64), starts).tolist()
        self._store_flags[channel] = (
            bool((block[:, BUFF_SPEED] > 0.5).any()), bool((block[:, DEBUFF_SPEED] > 0.5).any()))
        return [
            _Store(cells=cells, center=centers[k], protected=bool(protected_mask[cells[0]]),
                   battery=battery[k], specials=specials[k],
                   free_battery_capacity=int(capacity[k]), free_empty_tiles=empty[k])
            for k, cells in enumerate(components)
        ]

    def _components_index(self, components):
        """Concatenated cell indices, component start offsets and centres (episode cache)."""
        entry = self._cell_index.get(id(components))
        if entry is None or entry[0] is not components:
            cells = [c for component in components for c in component]
            ys = np.fromiter((c[0] for c in cells), dtype=np.intp, count=len(cells))
            xs = np.fromiter((c[1] for c in cells), dtype=np.intp, count=len(cells))
            starts = np.cumsum([0] + [len(c) for c in components[:-1]]).astype(np.intp)
            centers = [self._component_center(c) for c in components]
            entry = (components, ys, xs, starts, centers)
            self._cell_index[id(components)] = entry
        return entry[1:]

    def _store_field(self, walkable, team: str, index: int, store: _Store) -> np.ndarray:
        """Path length from every cell to the nearest cell of a (static) storage component."""
        key = (team, index)
        field = self._storage_fields.get(key)
        if field is None:
            field = np.full(walkable.shape, np.inf, dtype=np.float32)
            for cell in store.cells:
                field = np.minimum(field, self._cached_distance_map(walkable, cell))
            self._storage_fields[key] = field
        return field

    # ------------------------------------------------------------------ main loop

    def act(self, obs):
        if not obs:
            return {}
        sample = next(iter(obs.values()))
        graphic = sample["graphic"]
        states = sample["agent_states"]
        team_state = sample["team_state"]
        time_left = float(team_state[2])
        if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
            self.reset()
        self._last_time_left = time_left
        self._tick += 1

        shape = graphic.shape[:2]
        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}
        walkable = graphic[..., WALL] < 0.5

        ally_spawn = self._nearest_pixel(graphic[..., SPAWN_ALLY] > 0.5, (0, 0))
        enemy_spawn = self._nearest_pixel(graphic[..., SPAWN_ENEMY] > 0.5, (0, 0))
        pixels = self._state_pixels(states, shape)
        own = [self._parse_unit(n, row_for[n], states[row_for[n]], shape, pixels[row_for[n]])
               for n in controlled]
        enemies = []
        enemy_base = self._cached_base_mask(enemy_spawn, shape)
        for row, side in enumerate(states[:, 2].tolist()):
            if side < 0:
                unit = self._parse_unit(None, row, states[row], shape, pixels[row])
                unit.in_base = bool(enemy_base[unit.pos])
                enemies.append(unit)

        ctx = {
            "graphic": graphic, "walkable": walkable, "shape": shape, "states": states,
            "own": own, "enemies": enemies, "ally_spawn": ally_spawn, "enemy_spawn": enemy_spawn,
            "ally_stores": self._stores(
                graphic, STORAGE_ALLY, self._cached_protected_ally_storage_mask(graphic)),
            "enemy_stores": self._stores(
                graphic, STORAGE_ENEMY, self._cached_protected_enemy_storage_mask(graphic)),
            "absorb": max(0.0, float(team_state[3]) * ABSORPTION_INTERVAL_SECONDS),
            "remaining": max(0.0, time_left * MATCH_SECONDS),
            "lead": (float(team_state[0]) - float(team_state[1])) * 100.0,
        }
        # Speed effects are the items sitting in each team's storages (set by _stores).
        ctx["own_buff"] = self._store_flags[STORAGE_ALLY][0]
        ctx["own_slowed"] = self._store_flags[STORAGE_ENEMY][1]
        ctx["enemy_buff"] = self._store_flags[STORAGE_ENEMY][0]
        ctx["enemy_slowed"] = self._store_flags[STORAGE_ALLY][1]
        pixels = self._battery_pixels(graphic)
        field: list[bool] = []
        if pixels:
            outside = (graphic[..., STORAGE_ALLY] < 0.5) & (graphic[..., STORAGE_ENEMY] < 0.5)
            field = outside[[p[0] for p in pixels], [p[1] for p in pixels]].tolist()
        # Summed one by one in scan order, as before, so the float total is unchanged.
        ctx["field_value"] = float(sum(p[2] for p, keep in zip(pixels, field) if keep))

        self._update_roles(ctx)
        plan = self._plan(ctx)
        self.plan = plan
        actions = {}
        for unit in own:
            target, kind = plan.get(unit.name, (None, "idle"))
            actions[unit.name] = self._navigate(
                unit.name, states[unit.row], states, target, kind, walkable, shape)
        return actions

    # ------------------------------------------------------------------ helpers

    def _cached_base_mask(self, spawn, shape) -> np.ndarray:
        key = (spawn, tuple(shape))
        mask = self._base_masks.get(key)
        if mask is None:
            mask = self._base_masks[key] = self._base_mask(spawn, shape)
            mask.flags.writeable = False
        return mask

    @staticmethod
    def _base_mask(spawn, shape) -> np.ndarray:
        """Each team base is the 4x4 corner square holding its spawn (Prototype.unity fixed
        regions); its floor blocks enemy movement."""
        mask = np.zeros(shape, dtype=bool)
        if spawn is None:
            return mask
        sy, sx = spawn
        rows = range(sy - 3, sy + 1) if sy > shape[0] // 2 else range(sy, sy + 4)
        cols = range(sx - 3, sx + 1) if sx > shape[1] // 2 else range(sx, sx + 4)
        for y in rows:
            for x in cols:
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    mask[y, x] = True
        return mask

    def _camp_slots(self, ctx) -> list[tuple[int, int]]:
        """Walkable cells just outside the enemy base, best first: the corner facing the map
        centre, then cells along each base edge moving away from it."""
        key = ("camp", ctx["enemy_spawn"])
        cached = self._static_masks.get(key)
        if cached is not None:
            return cached
        shape, walkable = ctx["shape"], ctx["walkable"]
        base = self._cached_base_mask(ctx["enemy_spawn"], shape)
        frontier = []
        for y, x in zip(*np.nonzero(walkable & ~base)):
            y, x = int(y), int(x)
            if any(0 <= y + dy < shape[0] and 0 <= x + dx < shape[1] and base[y + dy, x + dx]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                frontier.append((y, x))
        center = (shape[0] / 2 - 0.5, shape[1] / 2 - 0.5)
        frontier.sort(key=lambda c: math.dist(c, center))
        slots = []
        for cell in frontier:
            if all(math.dist(cell, other) >= 2.0 for other in slots):
                slots.append(cell)
        self._static_masks[key] = slots
        return slots

    def _speed(self, ctx, cls: int, enemy: bool = False) -> float:
        mult = 1.0
        if enemy:
            mult += 0.5 * ctx["enemy_buff"]
            if cls == COLLECTOR:
                mult -= 0.9 * ctx["enemy_slowed"]
        else:
            mult += 0.5 * ctx["own_buff"]
            if cls == COLLECTOR:
                mult -= 0.9 * ctx["own_slowed"]
        return BASE_SPEED[cls] * max(0.1, mult)

    def _dist(self, ctx, start, goal) -> float:
        return float(self._cached_distance_map(ctx["walkable"], start)[goal])

    def _next_absorption_exposure(self, ctx, arrival: float) -> float:
        """Seconds an item delivered after ``arrival`` seconds stays stealable."""
        remaining, absorb = ctx["remaining"], ctx["absorb"]
        if arrival >= remaining:
            return 0.0
        end = absorb
        while end <= arrival:
            end += ABSORPTION_INTERVAL_SECONDS
        return max(0.0, min(end, remaining) - arrival)

    def _enemy_reach_seconds(self, ctx, walkable, team_key, index, store) -> float:
        """Fastest time any empty-handed enemy Collector/Carrier can step into ``store``."""
        field = self._store_field(walkable, team_key, index, store)
        best = math.inf
        for enemy in ctx["enemies"]:
            if enemy.cls == HUNTER or enemy.holding:
                continue
            distance = float(field[enemy.pos])
            if np.isfinite(distance):
                best = min(best, distance / self._speed(ctx, enemy.cls, enemy=True))
        return best

    def _hunter_near(self, ctx, point, radius, *, enemy: bool) -> bool:
        units = ctx["enemies"] if enemy else ctx["own"]
        return any(
            u.cls == HUNTER and not u.in_base and math.dist(u.pos, point) <= radius for u in units
        )

    # ------------------------------------------------------------------ roles

    def _update_roles(self, ctx) -> None:
        own = ctx["own"]
        graphic = ctx["graphic"]
        roles = {}
        carriers = [i for i, u in enumerate(own) if u.cls == CARRIER]
        hunters = [i for i, u in enumerate(own) if u.cls == HUNTER]
        for i in carriers:
            roles[i] = "carrier"
        for i in hunters:
            roles[i] = "hunter"

        def pending(role):
            return [i for i, r in self._role_of.items()
                    if r == role and i < len(own) and own[i].cls == COLLECTOR]

        # Carrier: cheap (sanctuary next to spawn) and 50% faster while field value remains.
        want_carrier = self.carrier_quota > 0 and ctx["field_value"] >= 25.0
        if want_carrier and not carriers:
            keep = [i for i in pending("carrier") if not own[i].holding]
            choice = keep[0] if keep else self._closest_to_site(ctx, SITE_CARRIER, set(roles))
            if choice is not None:
                roles[choice] = "carrier"
        want_hunters = self.hunter_quota
        missing = want_hunters - len(hunters)
        if missing > 0:
            keep = [i for i in pending("hunter") if not own[i].holding and i not in roles]
            for i in keep[:missing]:
                roles[i] = "hunter"
                missing -= 1
            while missing > 0:
                choice = self._closest_to_site(ctx, SITE_HUNTER, set(roles))
                if choice is None:
                    break
                roles[choice] = "hunter"
                missing -= 1
        for i, unit in enumerate(own):
            roles.setdefault(i, "collector")
        self._role_of = roles
        self.role_assignments = {u.name: roles[i] for i, u in enumerate(own)}

    def _closest_to_site(self, ctx, channel, excluded) -> int | None:
        sites = list(zip(*np.nonzero(ctx["graphic"][..., channel] > 0.5)))
        if not sites:
            return None
        best = None
        for i, unit in enumerate(ctx["own"]):
            if i in excluded or unit.cls != COLLECTOR or unit.holding:
                continue
            field = self._cached_distance_map(ctx["walkable"], unit.pos)
            distance = min(float(field[s]) for s in sites)
            if np.isfinite(distance) and (best is None or distance < best[0]):
                best = (distance, i)
        return None if best is None else best[1]

    # ------------------------------------------------------------------ planning

    def _plan(self, ctx) -> dict[str, tuple[tuple[int, int] | None, str]]:
        own = ctx["own"]
        graphic = ctx["graphic"]
        plan: dict[str, tuple[tuple[int, int] | None, str]] = {}
        reserved_tiles: set[tuple[int, int]] = set()
        economic = []
        for i, unit in enumerate(own):
            role = self._role_of.get(i, "collector")
            if unit.cls == HUNTER:
                continue  # planned after the economy so it can react to our exposed storages
            if unit.holding:
                plan[unit.name] = self._deposit_plan(ctx, unit, reserved_tiles)
                continue
            if unit.cls == COLLECTOR and role in ("carrier", "hunter"):
                channel = SITE_CARRIER if role == "carrier" else SITE_HUNTER
                target = self._nearest_site_center(graphic, channel, unit.pos)
                plan[unit.name] = (target, "transform")
                continue
            economic.append(unit)

        plan.update(self._assign_economy(ctx, economic))
        claimed = {target for target, kind in plan.values() if kind == "hunt"}
        for unit in own:
            if unit.cls == HUNTER:
                plan[unit.name] = self._hunter_plan(ctx, unit, claimed)
                if plan[unit.name][1] == "hunt":
                    claimed.add(plan[unit.name][0])
        return plan

    def _nearest_site_center(self, graphic, channel, origin):
        """``_nearest_component_center`` over the episode component cache (sites are static)."""
        components = self._cached_components(graphic[..., channel] > 0.5)
        if not components:
            return None
        component = min(
            components,
            key=lambda c: min((p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2 for p in c),
        )
        return self._component_center(component)

    def _deposit_plan(self, ctx, unit, reserved_tiles):
        walkable = ctx["walkable"]
        speed = self._speed(ctx, unit.cls)
        value = unit.amount if unit.slot == 1 else 8.0
        best = None
        for index, store in enumerate(ctx["ally_stores"]):
            fits = (store.free_battery_capacity >= max(1, round(unit.amount))
                    if unit.slot == 1 else store.free_empty_tiles > 0)
            if not fits:
                continue
            field = self._store_field(walkable, "ally", index, store)
            distance = float(field[unit.pos])
            if not np.isfinite(distance):
                continue
            arrival = distance / speed
            cost = arrival
            if not store.protected:
                exposure = self._next_absorption_exposure(ctx, arrival)
                if exposure > 0:
                    reach = self._enemy_reach_seconds(ctx, walkable, "ally", index, store)
                    # Loss grows with how much of the exposure window the enemy can use.
                    window = max(0.0, arrival + exposure - max(arrival, reach))
                    risk = min(1.0, window / 6.0)
                    if self._hunter_near(ctx, store.center, 3.0, enemy=False):
                        risk *= 0.5
                    cost += self.exposure_cost * value * risk
            if best is None or cost < best[0]:
                best = (cost, index, store)
        if best is None:
            target, can_deposit = self._storage_target(ctx["graphic"], ctx["states"][unit.row], unit.pos)
            return target, "deposit" if can_deposit else "wait_storage"
        _, _, store = best
        free = [c for c in store.cells if c not in reserved_tiles] or store.cells
        target = min(free, key=lambda c: math.dist(c, unit.pos))
        reserved_tiles.add(target)
        return target, "deposit"

    def _assign_economy(self, ctx, units):
        if not units:
            return {}
        graphic = ctx["graphic"]
        walkable = ctx["walkable"]
        ally_stores = ctx["ally_stores"]
        depot = None
        for index, store in enumerate(ally_stores):
            field = self._store_field(walkable, "ally", index, store)
            depot = field if depot is None else np.minimum(depot, field)
        enemy_hunters = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        enemy_collectors = [e for e in ctx["enemies"] if e.cls != HUNTER and not e.holding]

        tasks: list[tuple[tuple[int, int], str, float, float]] = []  # target, kind, value, extra
        protected_enemy = self._cached_protected_enemy_storage_mask(graphic)
        for y, x, amount in self._battery_pixels(graphic):
            if graphic[y, x, STORAGE_ALLY] > 0.5 or protected_enemy[y, x]:
                continue
            kind = "steal" if graphic[y, x, STORAGE_ENEMY] > 0.5 else "battery"
            tasks.append(((y, x), kind, amount, 0.0))
        for channel, y, x in self._special_pixels(graphic):
            if graphic[y, x, STORAGE_ALLY] > 0.5 or protected_enemy[y, x]:
                continue
            tasks.append(((y, x), "special", float(channel), 1.0 if graphic[y, x, STORAGE_ENEMY] > 0.5 else 0.0))

        # Everything that does not depend on the unit is computed once per task.
        task_ys = [target[0] for target, _, _, _ in tasks]
        task_xs = [target[1] for target, _, _, _ in tasks]
        home = depot[task_ys, task_xs].tolist() if depot is not None and tasks else None
        enemy_speeds = [self._speed(ctx, e.cls, enemy=True) for e in enemy_collectors]
        hunter_danger, carrier_danger, contest = [], [], []
        for target, kind, _, _ in tasks:
            # Danger at the pickup point: enemy Hunters kill everything we have; for a
            # Carrier any enemy contact is lethal.
            danger = 0.0
            for hunter in enemy_hunters:
                separation = math.dist(target, hunter.pos)
                if separation < 4.0:
                    danger = max(danger, (4.0 - separation) / 4.0)
            hunter_danger.append(danger)
            for enemy in enemy_collectors:
                separation = math.dist(target, enemy.pos)
                if separation < 2.5:
                    danger = max(danger, (2.5 - separation) / 2.5)
            carrier_danger.append(danger)
            # Contest: an enemy much closer to a field battery will usually take it first.
            fastest = math.inf
            if kind == "battery":
                for enemy, enemy_speed in zip(enemy_collectors, enemy_speeds):
                    fastest = min(fastest, math.dist(target, enemy.pos) / enemy_speed)
            contest.append(fastest)

        pairs = []
        for unit in units:
            speed = self._speed(ctx, unit.cls)
            field = self._cached_distance_map(walkable, unit.pos)
            distances = field[task_ys, task_xs].tolist() if tasks else []
            dangers = carrier_danger if unit.cls == CARRIER else hunter_danger
            old = self._memory.get(unit.name)
            old_target = old.target if old is not None else None
            for task_index, (target, kind, value, extra) in enumerate(tasks):
                distance = distances[task_index]
                if not math.isfinite(distance):
                    continue
                t_pick = distance / speed
                t_home = home[task_index] / speed if home is not None else 10.0
                if not math.isfinite(t_home):
                    t_home = 10.0
                if kind == "battery":
                    gain = value
                elif kind == "steal":
                    if t_pick + 0.3 >= min(ctx["absorb"], ctx["remaining"]):
                        continue  # absorbed (or match over) before we arrive
                    gain = value * self.steal_weight
                else:
                    channel = int(value)
                    weight = self.special_weights[channel - BUFF_SPEED]
                    duration = self._next_absorption_exposure(ctx, t_pick + t_home)
                    gain = weight * duration
                    if extra:  # stealing it also ends the enemy's effect now
                        if t_pick + 0.3 >= min(ctx["absorb"], ctx["remaining"]):
                            continue
                        gain += weight * max(0.0, ctx["absorb"] - t_pick)
                    if unit.cls == CARRIER:
                        gain *= 0.8
                gain *= 1.0 - 0.6 * dangers[task_index]
                if contest[task_index] + 1.0 < t_pick:
                    gain *= 0.75
                score = gain / (t_pick + t_home + 1.0)
                if old_target == target:
                    score *= 1.12
                pairs.append((score, unit.name, task_index))

        pairs.sort(key=lambda p: p[0], reverse=True)
        result = {}
        used = set()
        for score, name, task_index in pairs:
            if name in result or task_index in used:
                continue
            target, kind, _, _ = tasks[task_index]
            result[name] = (target, "special" if kind == "special" else kind)
            used.add(task_index)
        for unit in units:
            if unit.name not in result:
                result[unit.name] = self._idle_plan(ctx, unit)
        return result

    def _idle_plan(self, ctx, unit):
        """No task: shadow the enemy external storage most likely to receive the next deposit."""
        graphic = ctx["graphic"]
        best = None
        for index, store in enumerate(ctx["enemy_stores"]):
            if store.protected:
                continue
            field = self._store_field(ctx["walkable"], "enemy", index, store)
            # Enemy cargo headed home picks the nearest storage: count cargo closest to this one.
            incoming = sum(
                e.amount for e in ctx["enemies"] if e.slot == 1
                and np.isfinite(field[e.pos]) and field[e.pos] <= 8.0
            )
            distance = self._dist(ctx, unit.pos, store.center)
            score = incoming * 2.0 + store.battery - 0.15 * distance
            if best is None or score > best[0]:
                best = (score, store)
        if best is None:
            return self._patrol_component_center(
                graphic[..., STORAGE_ENEMY] > 0.5, unit_index(unit.name), period=260), "patrol_raid"
        store = best[1]
        # Wait just outside so we don't bounce on the region boundary; enter when value appears.
        return store.center, "patrol_raid"

    def _hunter_plan(self, ctx, hunter, claimed):
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        rank = hunters.index(hunter.name)
        if self.hunter_mode == "camp" or (self.hunter_mode == "mixed" and rank == 0):
            return self._camp_plan(ctx, hunter, rank, claimed)
        return self._hunt_plan(ctx, hunter, claimed)

    def _camp_plan(self, ctx, hunter, rank, claimed):
        """Hold a cell outside the enemy base; strike anything that comes within reach."""
        slots = self._camp_slots(ctx)
        if not slots:
            return self._hunt_plan(ctx, hunter, claimed)
        slot = slots[min(rank, len(slots) - 1)]
        field = self._cached_distance_map(ctx["walkable"], hunter.pos)
        best = None
        for enemy in ctx["enemies"]:
            if enemy.in_base or enemy.cls == HUNTER:
                continue
            if math.dist(enemy.pos, slot) > self.camp_engage_radius:
                continue
            distance = float(field[enemy.pos])
            if not np.isfinite(distance):
                continue
            value = 1.0 + (enemy.amount + 4.0 if enemy.slot == 1 else 7.0 if enemy.slot >= 2 else 0.0)
            value += 3.0 if enemy.cls == CARRIER else 0.0
            score = value / (distance + 1.0)
            if enemy.pos in claimed:
                score *= 0.5
            if best is None or score > best[0]:
                best = (score, enemy)
        if best is not None:
            return best[1].pos, "hunt"
        return slot, "patrol_defend"

    def _hunt_plan(self, ctx, hunter, claimed):
        walkable = ctx["walkable"]
        speed = self._speed(ctx, HUNTER)
        field = self._cached_distance_map(walkable, hunter.pos)
        exposed = []
        for index, store in enumerate(ctx["ally_stores"]):
            if store.protected or store.battery + store.specials * 6 <= 0:
                continue
            exposed.append(store)
        best = None
        for enemy in ctx["enemies"]:
            if enemy.in_base:
                continue
            distance = float(field[enemy.pos])
            if not np.isfinite(distance):
                continue
            value = 0.0
            if enemy.slot == 1:
                value += enemy.amount
            elif enemy.slot >= 2:
                value += 6.0
            if enemy.cls == CARRIER:
                value += 3.0
            if enemy.cls == COLLECTOR and not enemy.holding:
                value += 1.0  # sending a worker home costs the enemy a trip
                for store in exposed:
                    gap = min(math.dist(enemy.pos, cell) for cell in store.cells)
                    if gap <= 5.0:
                        value += min(12.0, store.battery + 6.0 * store.specials) * (5.0 - gap) / 5.0
            if enemy.cls == HUNTER:
                # Trading Hunters (both die) only helps when theirs blocks our economy: camping
                # our spawn exit or our exposed value.  Theirs respawns far away as a Collector.
                camping = any(min(math.dist(enemy.pos, c) for c in s.cells) <= 3.0 for s in exposed)
                at_spawn = ctx["ally_spawn"] is not None and math.dist(enemy.pos, ctx["ally_spawn"]) <= 7.0
                value = 12.0 if at_spawn else 4.0 if camping else 0.0
            if value <= 0:
                continue
            enemy_speed = self._speed(ctx, enemy.cls, enemy=True)
            chase = 1.0 if enemy_speed < speed else 2.5  # equal speed chases rarely close
            t = distance / speed
            score = value / (t * chase + 1.0)
            if enemy.pos in claimed:
                score *= 0.5
            if best is None or score > best[0]:
                best = (score, enemy)
        if best is not None and best[0] > 0.35:
            return best[1].pos, "hunt"
        if exposed:
            store = max(exposed, key=lambda s: s.battery + 6 * s.specials)
            return store.center, "patrol_defend"
        # Nothing to protect: stand between enemy economy and their storages (field centre).
        return self._defend_patrol_target(ctx["graphic"], hunter.pos), "patrol_defend"

    def _cached_astar(self, walkable, start, goal):
        # V6's threat-cost A* depends on where enemies are right now, so a (start, goal) path
        # cache would replay routes planned around threats that have since moved.
        return list(self._astar(walkable, start, goal))
