"""Twenty-second-generation heuristic: the any-angle V21 teacher that wins the opening Hunter race.

Against the strongest distilled students (P5a, D6) the any-angle V21 (``strategic_v21_anyangle``)
loses or wins by ~60 points, and the side decides which before 5 s of play: whichever team
turns a Collector into a Hunter at the central sanctuary first (both arrive at ~4.0 s) kills the
other team's walkers on the spot, keeps it from ever getting a Hunter (V21 only sends walkers on
routes clear of enemy Hunters) and camps its base exit for the rest of the match.  In 48 held-out
matches (``reports/teacher_v2/base_students``) the first-Hunter side won 41 times; P5a was first
in 13/24, D6 in 7/24.  Replaying the teacher's own labels showed why it lost the race: its three
walkers stepped on field batteries on the way in most of the lost matches, and a V17-teach
Collector holding cargo turns back to deposit (it is no longer a transform candidate).

V22 keeps V21's goals and changes these (each can be switched off for ablations; measured
contribution, V22 minus V22 without it, in docs/policy_v2_design_ko.md section 11):

* ``race_cargo`` -- while the race is on (no own Hunter yet, or the first ``opening_seconds``) a
  Collector holding a battery is still a Hunter candidate and keeps walking: the battery is lost
  on transforming (Hunters cannot carry), a few points against the match;
* ``item_free_walkers`` -- a walker routes around field items when that route is no longer than
  the direct one (+ ``detour_slack`` tiles), so it keeps its cargo-free hands and its points
  (V18/V19's opening rule, here on V21's Hunter-free map);
* ``region_approach`` (the playing class) -- transformations and deposits fire when the unit's
  centre *enters* any tile of the sanctuary / storage region (``Unit.ChangePos`` ->
  ``MapRegion.OnUnitEnter``), not at a tile centre, so once the region is in sight the unit heads
  for the nearest point of the region (inset by ``REGION_INSET``) instead of the nearest centre:
  up to ~0.7 tiles (0.18 s for a Collector) shorter on the decisive opening walk;
* ``sentry`` -- having won the race, V21 left the sanctuary unguarded and the enemy got 3-4
  Hunters at 10-20 s; with two or more Hunters the one nearest V19's sentry post (enemy side of
  the sanctuary) strikes enemy Collectors near the sanctuary or holds the post;
* ``commit`` -- a walker V21 would turn back (no route clear of enemy Hunters) goes on when it
  enters the sanctuary before any enemy Hunter running straight at it could touch it;
* ``break_ties`` -- a union of tied goals that cancels (opposite directions) is unlabelled in the
  teach labellers, and a unit playing it stood still for the rest of the match; V22 takes the
  nearest of the tied goals instead.

Every rule is a function of the observation (as V21): a fresh instance plays the same moves.

Unit size (``size_aware``): ``Unit.Move`` passes the class bound (``UnitData.CollisionBound``) to
the movement system, so walls and corners never see the size stat; ``UnitInteractionSystem``
uses ``Unit.CollisionBound`` (width x ``UnitSize``) for every unit-unit and unit-item overlap.  A
Hunter therefore kills from ``R_hunter * size_h + R_victim * size_v`` (0.6 tiles for a Collector at
size 1).  Size and speed come from the items lying in storages (``BuffItemEffect``: a stat
modifier per item while it is in a storage; modifiers of one stat add, ``TeamContext.CalculateStat``):
BuffSize in a team's storage +50% for that team, DebuffSize -30% for the other team, BuffSpeed
+50% for the owner, DebuffSpeed -90% for the other team's Collectors.  The observation shows every
storage cell's item, so the heuristic counts them.  With ``size_aware`` the threat repulsion radius
and the Hunter danger zones (V20 camp avoidance, V21 safe transform routes) grow or shrink with the
current contact distance, and travel times use the counted speed modifiers (V17 used one
"any buff" flag per storage side).
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .any_angle import (AnyAngleMixin, SAMPLE_STEP, _point_clearance, _segment_clear, point_clearance,  # noqa: F401
                        segment_clear, pixel_position, world_heading)
from .strategic import BATTERY, CARRIER, COLLECTOR, FIRST_SPECIAL, HUNTER, SITE_CARRIER, SITE_HUNTER, STORAGE_ALLY, STORAGE_ENEMY
from .teach import Goal, TeachLabel
from .v17_planner import BASE_SPEED, BUFF_SIZE, BUFF_SPEED, DEBUFF_SIZE, DEBUFF_SPEED, MATCH_SECONDS
from .v21_safe_transform import StrategicHeuristicV21Teach

# UnitData/*.asset CollisionBound.Width / 2: the radius at size 1 (also any_angle.UNIT_RADIUS)
CONTACT_RADIUS = {COLLECTOR: 0.275, HUNTER: 0.325, CARRIER: 0.225}
BASE_CONTACT = CONTACT_RADIUS[HUNTER] + CONTACT_RADIUS[COLLECTOR]  # what the 0.16 threat radius was tuned at
SIZE_BUFF, SIZE_DEBUFF = 0.5, -0.3  # ItemEffects/SizeBuff.asset, SizeDebuff.asset (Multiply: additive percents)
SPEED_BUFF, SPEED_DEBUFF = 0.5, -0.9  # ItemEffects/SpeedBuff.asset, SpeedDebuff.asset (Collectors only)
HALF_MAP_TILES = 12.0  # agent_states positions are normalised by half the 24-tile map
OPENING_SECONDS = 10.0
DETOUR_SLACK = 1e-6  # tiles: the item-free walk is taken only when it is no longer
REGION_INSET = 0.1  # tiles inside the region boundary a region approach aims for
SENTRY_MIN_HUNTERS = 2  # a sentry is posted once we have this many Hunters
SENTRY_OFFSET = 2.5  # tiles from the sanctuary centre toward the enemy spawn (V19)
SENTRY_RADIUS = 7.0  # the sentry strikes enemy Collectors this close to the sanctuary centre (V19)
COMMIT_MARGIN = 0.0  # seconds a committed walker must reach the sanctuary ahead of any enemy Hunter
_ITEM_CHANNELS = (BUFF_SPEED, DEBUFF_SPEED, BUFF_SIZE, DEBUFF_SIZE)



def _gap(a, b) -> float:
    """``math.hypot(*np.subtract(a, b))`` for two (row, col) float tuples, without the array."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class StrategicHeuristicV22Teach(StrategicHeuristicV21Teach):
    """V21-teach that keeps its walkers walking in the opening race (and sees unit size/speed items)."""

    def __init__(self, *, race_cargo: bool = True, item_free_walkers: bool = True, size_aware: bool = True,
                 sentry: bool = True, sentry_min_hunters: int = SENTRY_MIN_HUNTERS, sentry_offset: float = SENTRY_OFFSET,
                 sentry_radius: float = SENTRY_RADIUS, commit: bool = True, commit_margin: float = COMMIT_MARGIN,
                 break_ties: bool = True,
                 opening_seconds: float = OPENING_SECONDS, detour_slack: float = DETOUR_SLACK, **kwargs):
        super().__init__(**kwargs)
        self.sentry = bool(sentry)
        self.sentry_min_hunters = int(sentry_min_hunters)
        self.sentry_offset = float(sentry_offset)
        self.sentry_radius = float(sentry_radius)
        self.sentries: list[str] = []  # for inspection: this tick's sentry
        self.commit = bool(commit)
        self.break_ties = bool(break_ties)
        self.commit_margin = float(commit_margin)
        self.committed: list[str] = []  # for inspection: walkers committed past enemy Hunters this tick
        self._committed_xy: set[tuple[float, float]] = set()
        self.race_cargo = bool(race_cargo)
        self.item_free_walkers = bool(item_free_walkers)
        self.size_aware = bool(size_aware)
        self.opening_seconds = float(opening_seconds)
        self.detour_slack = float(detour_slack)
        self.sizes = (1.0, 1.0)  # (own, enemy) UnitSize this tick, for inspection

    # ------------------------------------------------------------------ size and speed items

    # threat_radius / danger_tiles read as the configured value (what the parents' __init__ and the
    # mixture's in-place retuning set) shifted by this tick's contact-distance change (``_size_extra``).
    @property
    def threat_radius(self) -> float:
        return max(0.0, self._threat_radius_base + getattr(self, "_size_extra", 0.0) / HALF_MAP_TILES)

    @threat_radius.setter
    def threat_radius(self, value: float) -> None:
        self._threat_radius_base = float(value)

    @property
    def danger_tiles(self) -> float:
        return max(0.0, self._danger_tiles_base + getattr(self, "_size_extra", 0.0))

    @danger_tiles.setter
    def danger_tiles(self, value: float) -> None:
        self._danger_tiles_base = float(value)

    @staticmethod
    def _storage_items(graphic: np.ndarray, storage: int, channel: int) -> int:
        return int(((graphic[..., storage] > 0.5) & (graphic[..., channel] > 0.5)).sum())

    def _context(self, obs) -> dict:
        ctx = super()._context(obs)
        g = ctx["graphic"]
        # = {(s, c): self._storage_items(g, s, c) ...}: one gather of each team's storage cells
        count = {}
        for s in (STORAGE_ALLY, STORAGE_ENEMY):
            hits = (g[g[..., s] > 0.5][:, _ITEM_CHANNELS] > 0.5).sum(axis=0).tolist()
            for c, k in zip(_ITEM_CHANNELS, hits):
                count[s, c] = int(k)
        own_size = max(0.0, 1.0 + SIZE_BUFF * count[STORAGE_ALLY, BUFF_SIZE] + SIZE_DEBUFF * count[STORAGE_ENEMY, DEBUFF_SIZE])
        enemy_size = max(0.0, 1.0 + SIZE_BUFF * count[STORAGE_ENEMY, BUFF_SIZE] + SIZE_DEBUFF * count[STORAGE_ALLY, DEBUFF_SIZE])
        ctx["own_size"], ctx["enemy_size"] = own_size, enemy_size
        ctx["speed_items"] = count
        self.sizes = (own_size, enemy_size)
        # How much farther (or nearer) an enemy Hunter reaches one of our Collectors than at size 1.
        self._size_extra = (CONTACT_RADIUS[HUNTER] * enemy_size + CONTACT_RADIUS[COLLECTOR] * own_size - BASE_CONTACT
                            if self.size_aware else 0.0)
        return ctx

    def _speed(self, ctx, cls: int, enemy: bool = False) -> float:
        if not self.size_aware or "speed_items" not in ctx:
            return super()._speed(ctx, cls, enemy)
        count = ctx["speed_items"]
        mine, theirs = (STORAGE_ENEMY, STORAGE_ALLY) if enemy else (STORAGE_ALLY, STORAGE_ENEMY)
        mult = 1.0 + SPEED_BUFF * count[mine, BUFF_SPEED]
        if cls == COLLECTOR:
            mult += SPEED_DEBUFF * count[theirs, DEBUFF_SPEED]
        return BASE_SPEED[cls] * max(0.1, mult)  # Unity clamps at 0; 0.1 keeps times finite (as V17)

    def contact_distance(self, ctx, cls_a: int, a_is_enemy: bool, cls_b: int, b_is_enemy: bool) -> float:
        """Centre distance at which two units touch (tiles), with the current size modifiers."""
        size = lambda enemy: ctx["enemy_size"] if enemy else ctx["own_size"]
        return CONTACT_RADIUS[cls_a] * size(a_is_enemy) + CONTACT_RADIUS[cls_b] * size(b_is_enemy)

    # ------------------------------------------------------------------ the opening race

    def _racing(self, ctx) -> bool:
        elapsed = MATCH_SECONDS - ctx["remaining"]
        return elapsed < self.opening_seconds or not any(u.cls == HUNTER for u in ctx["own"])

    def teach(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, TeachLabel]:
        """V17-teach's ``teach``, except that a transform goal outranks a deposit (race_cargo)."""
        if not obs:
            return {}
        self._memory.clear()
        ctx = self._context(obs)
        graphic, states = ctx["graphic"], ctx["states"]
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
            if unit.name in transforms:
                goals[unit.name] = transforms[unit.name]
            elif unit.holding:
                goals[unit.name] = self._deposit_goals(ctx, unit)
            else:
                economic.append(unit)
        goals.update(self._economy_goals(ctx, economic))
        goals.update(self._hunter_goals(ctx, [unit for unit in own if unit.cls == HUNTER], move_walkable))
        return self._labels(ctx, goals, move_walkable)

    def _composition_goals(self, controlled, row_for, states, graphic, walkable) -> dict[str, tuple[Goal, ...]]:
        """TeachLabeller._composition_goals on V21's Hunter-free map, where during the race a
        Collector holding a battery (not a special) may still be sent to the Hunter sanctuary."""
        ctx = self._teach_ctx
        plain = walkable
        walkable = self._hunter_free(ctx, walkable)
        self.committed = []
        self._committed_xy = set()
        target_hunters, target_carriers = self._composition_targets(graphic, states, row_for)
        if not target_hunters and not target_carriers:
            return {}
        shape = graphic.shape[:2]
        classes = {name: self._class_id(states[row_for[name]]) for name in controlled}
        racing = self.race_cargo and self._racing(ctx)
        holding = {name: self._is_holding(states[row_for[name]]) for name in controlled}
        battery_only = {name: float(states[row_for[name]][5:9].max()) < 0.5 for name in controlled}
        goals: dict[str, tuple[Goal, ...]] = {}
        taken: set[str] = set()
        for role, channel, target in ((CARRIER, SITE_CARRIER, target_carriers), (HUNTER, SITE_HUNTER, target_hunters)):
            missing = (target or 0) - sum(cls == role for cls in classes.values())
            if missing <= 0:
                continue
            candidates = [
                name for name in controlled
                if name not in taken and classes[name] == COLLECTOR
                and (not holding[name] or (role == HUNTER and racing and battery_only[name]))
            ]
            if not candidates:
                continue
            tiles = self._site_tiles(graphic, channel, walkable)
            ranked = []
            if tiles:
                distance = self._distance_to(walkable, tiles)
                reach = {name: float(distance[self._to_pixel(states[row_for[name]][:2], shape)]) for name in candidates}
                ranked = sorted((name for name in candidates if math.isfinite(reach[name])),
                                key=lambda name: (reach[name], unit_index(name)))
            for name in ranked[:missing]:
                goals[name] = ((tiles, "transform"),)
                taken.add(name)
            if role == HUNTER and self.commit and len(ranked) < missing and walkable is not plain:
                self._commit_walkers(ctx, [n for n in candidates if n not in taken], missing - len(ranked),
                                     row_for, states, graphic, plain, goals, taken)
        return goals

    def _commit_walkers(self, ctx, candidates, missing, row_for, states, graphic, plain, goals, taken) -> None:
        """Walkers with no route clear of enemy Hunters that still reach the sanctuary first.

        V21 drops a walker's transformation as soon as an enemy Hunter covers every route -- also
        one a step from the sanctuary, which then turns round and is caught (a Hunter runs 6 tiles/s
        to its 4).  Here a Collector goes on when it enters the sanctuary region (path distance to a
        tile centre - 0.5) before any enemy Hunter outside its base could touch it running straight
        at it: ``d / v_collector + margin < (gap - contact) / (v_hunter + v_collector)``.  Once a
        Hunter, a contact is a trade instead of a lost unit."""
        shape = graphic.shape[:2]
        tiles = self._site_tiles(graphic, SITE_HUNTER, plain)
        if not tiles or not candidates:
            return
        hunters = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        distance = self._distance_to(plain, tiles)
        v_me = self._speed(ctx, COLLECTOR)
        v_them = self._speed(ctx, HUNTER, enemy=True)
        contact = CONTACT_RADIUS[HUNTER] * ctx["enemy_size"] + CONTACT_RADIUS[COLLECTOR] * ctx["own_size"]
        scored = []
        for name in candidates:
            state = states[row_for[name]]
            reach = float(distance[self._to_pixel(state[:2], shape)])
            if not math.isfinite(reach):
                continue
            arrive = max(0.0, reach - 0.5) / v_me + self.commit_margin
            here = pixel_position(state, shape)
            caught = min(
                (_gap(pixel_position(states[e.row], shape), here) - contact) / (v_them + v_me)
                for e in hunters) if hunters else math.inf
            if arrive < caught:
                scored.append((reach, unit_index(name), name))
        for _, _, name in sorted(scored)[:missing]:
            goals[name] = ((tiles, "transform"),)
            taken.add(name)
            self.committed.append(name)
            state = states[row_for[name]]
            self._committed_xy.add((float(state[0]), float(state[1])))

    def _label(self, state, states, goals, walkable, shape) -> TeachLabel:
        """TeachLabeller._label, but tied goals pulling opposite ways resolve to the nearest one.

        The teach labellers leave such a union unlabelled (teaching "both ways" teaches the
        oscillation); played, an unlabelled unit stands -- and since nothing changes while it
        stands, a Carrier or Collector between two equally good batteries stood there for the
        rest of the match (V22 vs V19: 0.7 of the matches had such a freeze, ~200 unit-seconds
        each).  The nearest goal by path is self-reinforcing: walking toward it only shortens it."""
        label = super()._label(state, states, goals, walkable, shape)
        if label.valid or not self.break_ties or len(goals) < 2:
            return label
        pos = self._to_pixel(state[:2], shape)
        if any(pos in tiles for tiles, _ in goals):
            return label  # interacting on a goal tile
        reach = [(float(self._distance_to(walkable, tiles)[pos]), tiles, i) for i, (tiles, _) in enumerate(goals)]
        reach = [r for r in reach if math.isfinite(r[0])]
        if not reach:
            return label
        _, _, i = min(reach)
        return super()._label(state, states, (goals[i],), walkable, shape)

    def _evasion(self, state, states, delta):
        if self._committed_xy and (float(state[0]), float(state[1])) in self._committed_xy:
            return None  # a committed walker runs for the sanctuary, not away from the Hunter
        return super()._evasion(state, states, delta)

    # ------------------------------------------------------------------ sanctuary sentry

    def _hunter_goals(self, ctx, hunters, move_walkable) -> dict[str, tuple[Goal, ...]]:
        """V17-teach's camper and chasers, plus (with ``sentry_min_hunters`` Hunters) a sentry that
        keeps enemy walkers off the central sanctuary: the Hunter nearest its post (V19's, on the
        enemy side of the sanctuary) strikes the enemy Collector nearest it within ``sentry_radius``
        of the sanctuary, or holds the post."""
        self.sentries = []
        if not self.sentry or len(hunters) < self.sentry_min_hunters:
            return super()._hunter_goals(ctx, hunters, move_walkable)
        graphic = ctx["graphic"]
        tiles = np.argwhere(graphic[..., SITE_HUNTER] > 0.5)
        spawn = ctx["enemy_spawn"]
        if not len(tiles) or spawn is None:
            return super()._hunter_goals(ctx, hunters, move_walkable)
        centre = (float(tiles[:, 0].mean()), float(tiles[:, 1].mean()))
        post = self._sentry_post(move_walkable, centre, spawn)
        to_post = self._distance_to(move_walkable, (post,))
        sentry = min(hunters, key=lambda h: (float(to_post[h.pos]), unit_index(h.name)))
        self.sentries = [sentry.name]
        field = self._cached_distance_map(ctx["walkable"], sentry.pos)
        prey = [
            (float(field[e.pos]), e.pos) for e in ctx["enemies"]
            if e.cls == COLLECTOR and not e.in_base and math.dist(e.pos, centre) <= self.sentry_radius
            and math.isfinite(float(field[e.pos]))
        ]
        goals = super()._hunter_goals(ctx, [h for h in hunters if h is not sentry], move_walkable)
        goals[sentry.name] = (((min(prey)[1],), "hunt"),) if prey else (((post,), "patrol_defend"),)
        return goals

    def _sentry_post(self, walkable, centre, spawn) -> tuple[int, int]:
        key = ("v22_sentry_post", centre, spawn)
        cell = self._static_masks.get(key)
        if cell is None:
            direction = (spawn[0] - centre[0], spawn[1] - centre[1])
            norm = math.hypot(*direction) or 1.0
            post = (centre[0] + direction[0] / norm * self.sentry_offset, centre[1] + direction[1] / norm * self.sentry_offset)
            best = min(zip(*np.nonzero(walkable)), key=lambda c: math.dist(c, post))
            cell = self._static_masks[key] = (int(best[0]), int(best[1]))
        return cell

    def _field_item_mask(self, graphic: np.ndarray) -> np.ndarray:
        """Cells holding an item a Collector of ours would pick up by walking over them."""
        return (((graphic[..., BATTERY] > 1e-5) | np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1))
                & (graphic[..., STORAGE_ALLY] < 0.5))

    def _labels(self, ctx, goals: dict[str, tuple[Goal, ...]], move_walkable: np.ndarray) -> dict[str, TeachLabel]:
        labels = super()._labels(ctx, goals, move_walkable)
        graphic, states, shape = ctx["graphic"], ctx["states"], ctx["shape"]
        rows = {unit.name: unit.row for unit in ctx["own"]}
        for name in self.committed:  # straight in on the plain map (V21 would take a Hunter-free detour)
            labels[name] = self._label(states[rows[name]], states, goals.get(name, ()), move_walkable, shape)
        if not self.item_free_walkers:
            return labels
        items = None
        for unit in ctx["own"]:
            unit_goals = goals.get(unit.name, ())
            if unit.holding or unit.cls != COLLECTOR or unit.name in self.committed:
                continue
            if not any(kind == "transform" and graphic[tiles[0]][SITE_HUNTER] > 0.5 for tiles, kind in unit_goals):
                continue
            if items is None:
                items = self._field_item_mask(graphic)
                if not items.any():
                    return labels
            base = self._hunter_free(ctx, move_walkable)
            tiles = tuple(dict.fromkeys(t for goal_tiles, _ in unit_goals for t in goal_tiles))
            direct = float(self._distance_to(base, tiles)[unit.pos])
            free = base & ~items
            free[unit.pos] = base[unit.pos]
            for tile in tiles:
                free[tile] = base[tile]
            around = float(self._distance_to(free, tiles)[unit.pos])
            if math.isfinite(around) and around <= direct + self.detour_slack:
                route = self._label(states[unit.row], states, unit_goals, free, shape)
                if route.valid:
                    labels[unit.name] = route
        return labels


class StrategicHeuristicV22(AnyAngleMixin, StrategicHeuristicV22Teach):
    """V22 as a player: any-angle moves (as ``strategic_v21_anyangle``) plus region approaches."""

    region_inset = REGION_INSET  # tiles inside the region boundary a region approach aims for (V23 varies it)

    def __init__(self, *, region_approach: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.region_approach = bool(region_approach)

    def _any_angle(self, label, state, states, goals, walkable, shape) -> np.ndarray | None:
        if self.break_ties and label.goals:
            goals = label.goals  # the goal a tie was resolved to (the label's goals otherwise)
        if self.region_approach and label.kind in ("transform", "deposit"):
            heading = self._region_heading(label, state, goals, walkable, shape)
            if heading is not None:
                self._count("region")
                return heading
        return super()._any_angle(label, state, states, goals, walkable, shape)

    def _region_heading(self, label, state, goals, walkable, shape) -> np.ndarray | None:
        """Head for the nearest visible point of the goal region (any of its tiles, inset)."""
        pos = self._to_pixel(state[:2], shape)
        if not (0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]) or not walkable[pos]:
            return None
        tiles = {tile for goal_tiles, kind in goals if kind == label.kind for tile in goal_tiles}
        if not tiles or pos in tiles:
            return None
        position = pixel_position(state, shape)
        radius = CONTACT_RADIUS.get(self._class_id(state), 0.325)
        blocked = self._blocked(walkable)
        required = min(radius + self.clearance_margin,
                       float(_point_clearance(blocked, float(position[0]), float(position[1]))) - 1e-3) + 0.5 * SAMPLE_STEP
        lo = 0.5 - self.region_inset
        points = []
        for ty, tx in tiles:
            py = min(max(position[0], ty - lo), ty + lo)
            px = min(max(position[1], tx - lo), tx + lo)
            points.append((math.hypot(py - position[0], px - position[1]), (py, px)))
        points.sort()
        for gap, point in points[:8]:
            if gap < 1e-3:
                return None
            if _segment_clear(blocked, float(position[0]), float(position[1]), float(point[0]), float(point[1]),
                              float(max(required, 0.0)), float(SAMPLE_STEP)):
                return world_heading(point[0] - position[0], point[1] - position[1])
        return None
