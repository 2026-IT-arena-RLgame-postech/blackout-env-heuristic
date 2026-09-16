"""A coordinated, stateful full-game heuristic for BlackOut.

The policy deliberately consumes only the public competition observation.  It is therefore
usable both as a baseline in ``run_match`` and as a behaviour-cloning teacher without giving
the learner privileged Unity state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import heapq
import math

import numpy as np

from . import _native
from ..env.constants import unit_index
from ..model.base import BaseModel


# Observation channel/layout constants (see MyObsPreprocessor).
WALL, SITE_HUNTER, SITE_CARRIER = 1, 2, 3
SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY, BATTERY = 4, 5, 6, 7, 8
FIRST_SPECIAL = 9
COLLECTOR, HUNTER, CARRIER = 0, 1, 2

# team_state[3] is 1 - AbsorptionTimer.Ratio (MapObsAgent.cs), i.e. normalized by the absorption
# interval.  The observation does not carry the interval itself, so it must match
# GameBalanceConfig.asset's AbsorptionInterval.
ABSORPTION_INTERVAL_SECONDS = 20.0


@dataclass
class _UnitMemory:
    target: tuple[int, int] | None = None
    target_kind: str = ""
    path: list[tuple[int, int]] = field(default_factory=list)
    last_pos: np.ndarray | None = None
    stuck_ticks: int = 0
    replan_in: int = 0
    recent_positions: list[np.ndarray] = field(default_factory=list)
    escape_ticks: int = 0
    escape_action: np.ndarray | None = None
    arrival_key: tuple[tuple[int, int], str] | None = None
    arrival_ticks: int = 0


class StrategicHeuristic(BaseModel):
    """Coordinated gather/deposit/steal policy with Carrier and Hunter roles.

    Team-local role order is stable: the first unit becomes the sole Carrier, the second a
    Hunter, and the remaining three stay Collectors.  This gives BC a consistent role signal
    while still recovering automatically after death (a respawned specialist revisits its
    sanctuary).  Set ``use_specialists=False`` for an all-Collector ablation baseline.
    """

    def __init__(
        self,
        *,
        use_specialists: bool = True,
        replan_interval: int = 10,
        threat_radius: float = 0.16,
    ):
        self.use_specialists = use_specialists
        self.replan_interval = max(1, int(replan_interval))
        self.threat_radius = float(threat_radius)
        self._memory: dict[str, _UnitMemory] = {}
        self._static_masks: dict[str, np.ndarray] = {}
        self._component_cache: dict[tuple[tuple[int, int], bytes], list[list[tuple[int, int]]]] = {}
        self._path_cache: OrderedDict[
            tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], ...]
        ] = OrderedDict()
        self._battery_scan_tick = -1
        self._battery_scan: list[tuple[int, int, float]] = []
        self._special_scan_tick = -1
        self._special_scan: list[tuple[int, int, int]] = []
        self._tick = 0
        self._last_time_left: float | None = None

    def reset(self) -> None:
        """Clear episode-local paths and target reservations."""
        self._memory.clear()
        self._static_masks.clear()
        self._component_cache.clear()
        self._path_cache.clear()
        self._battery_scan_tick = -1
        self._battery_scan.clear()
        self._special_scan_tick = -1
        self._special_scan.clear()
        self._tick = 0
        self._last_time_left = None

    def retune(self, **parameters) -> None:
        """Change constructor parameters mid-match while keeping all episode state.

        Paths, escape timers, role assignments, strategy modes and caches survive; only the
        named knobs change.  A throwaway instance is built so every value goes through the
        constructor's own clamping.  Only for parameters stored under their own name and not
        derived into other attributes (e.g. not V16's ``siege_raid_slots``).
        """
        template = type(self)(**parameters)
        for name in parameters:
            if not hasattr(self, name):
                raise ValueError(f"{type(self).__name__} has no retunable attribute {name!r}")
            setattr(self, name, getattr(template, name))

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not obs:
            return {}
        sample = next(iter(obs.values()))
        graphic = sample["graphic"]
        states = sample["agent_states"]
        team_state = sample["team_state"]

        # Remaining time jumps upwards at reset.  BaseModel has no episode callback, so infer it.
        time_left = float(team_state[2])
        if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
            self.reset()
        self._last_time_left = time_left
        self._tick += 1

        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        # Names and state rows share physical unit indices; retain this fallback for synthetic
        # tests or future wrappers which rename agents.
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}
        local_order = {name: j for j, name in enumerate(controlled)}

        walkable = graphic[..., WALL] < 0.5
        reservations: set[tuple[int, int]] = set()
        actions: dict[str, np.ndarray] = {}
        for name in controlled:
            row = row_for[name]
            state = states[row]
            role = self._role(local_order[name])
            target, kind = self._choose_target(
                role, state, states, graphic, reservations, local_order[name], team_state
            )
            if target is not None and kind in {"battery", "special", "steal"}:
                reservations.add(target)
            actions[name] = self._navigate(
                name, state, states, target, kind, walkable, graphic.shape[:2]
            )
        return actions

    def _role(self, local_index: int) -> str:
        if not self.use_specialists:
            return "collector"
        if local_index == 0:
            return "carrier"
        if local_index == 1:
            return "hunter"
        return "collector"

    @staticmethod
    def _class_id(state: np.ndarray) -> int:
        # np.argmax setup dominates the comparison for a fixed three-element one-hot.
        a, b, c = float(state[-3]), float(state[-2]), float(state[-1])
        return 0 if a >= b and a >= c else 1 if b >= c else 2

    @staticmethod
    def _is_holding(state: np.ndarray) -> bool:
        # item encoding begins at 3: slot 0=none, slot 1=battery, 2+=specials.
        return any(float(state[i]) > 1e-5 for i in range(4, 9))

    def _choose_target(
        self,
        role: str,
        state: np.ndarray,
        states: np.ndarray,
        graphic: np.ndarray,
        reservations: set[tuple[int, int]],
        local_index: int,
        team_state: np.ndarray,
    ) -> tuple[tuple[int, int] | None, str]:
        pos = self._to_pixel(state[:2], graphic.shape[:2])
        cls = self._class_id(state)
        holding = self._is_holding(state)

        if role == "carrier" and cls == COLLECTOR and not holding:
            return self._nearest_component_center(graphic[..., SITE_CARRIER] > 0.5, pos), "transform"
        if role == "hunter" and cls == COLLECTOR and not holding:
            return self._nearest_component_center(graphic[..., SITE_HUNTER] > 0.5, pos), "transform"

        # A dead specialist is a Collector again; if it happened to pick something up on the
        # sanctuary route, never destroy that cargo by transforming—deposit it first.
        if holding:
            target, can_deposit = self._storage_target(graphic, state, pos)
            return target, "deposit" if can_deposit else "wait_storage"

        if cls == HUNTER:
            enemies = [(i, s) for i, s in enumerate(states) if s[2] < 0]
            enemy_spawn = self._nearest_pixel(graphic[..., SPAWN_ENEMY] > 0.5, pos)
            if enemy_spawn is not None:
                # Units idling/respawning inside their protected base are not reachable.  A
                # Hunter targeting them otherwise presses against the permission boundary.
                enemies = [
                    pair for pair in enemies
                    if math.dist(self._to_pixel(pair[1][:2], graphic.shape[:2]), enemy_spawn) > 3.5
                ]
            if enemies:
                def hunter_value(pair: tuple[int, np.ndarray]) -> float:
                    _, enemy = pair
                    ecls = self._class_id(enemy)
                    cargo = 2.0 if self._is_holding(enemy) else 0.0
                    carrier = 1.5 if ecls == CARRIER else 0.0
                    dist = float(np.linalg.norm(enemy[:2] - state[:2]))
                    return cargo + carrier - dist
                enemy = max(enemies, key=hunter_value)[1]
                return self._to_pixel(enemy[:2], graphic.shape[:2]), "hunt"
            return self._defend_patrol_target(graphic, pos), "patrol_defend"

        candidates: list[tuple[float, tuple[int, int], str]] = []
        protected_enemy = self._cached_protected_enemy_storage_mask(graphic)
        absorption_urgency = 1.0 - float(team_state[3])
        for y, x, amount in self._battery_pixels(graphic):
            p = (y, x)
            if p in reservations:
                continue
            # Own stored items cannot be picked up by an ally and are already fulfilling their
            # purpose.  Targeting them made empty collectors camp on their last deposit until
            # absorption removed the pixel.
            if graphic[y, x, STORAGE_ALLY] > 0.5:
                continue
            # The storage adjacent to enemy spawn is inside the protected base.  Its pixels
            # look traversable in the public map, but MapManager rejects our team's movement.
            if protected_enemy[y, x]:
                continue
            is_enemy_storage = graphic[y, x, STORAGE_ENEMY] > 0.5
            kind = "steal" if is_enemy_storage else "battery"
            distance = math.hypot(y - pos[0], x - pos[1])
            # Amount-weighting follows the reward proposal.  Steals become more valuable just
            # before absorption because they deny an imminent score as well as recover cargo.
            score = amount * (1.0 + (1.4 * absorption_urgency if is_enemy_storage else 0.0)) - 0.11 * distance
            candidates.append((score, p, kind))

        # Speed buff/debuff has strategic value, but battery wins ties and dominates near the
        # score threshold.  Later local indices are slightly less eager to avoid team-wide
        # over-commitment to the lone special.
        score_gap_to_win = max(0.0, 1.0 - float(team_state[0]))
        if score_gap_to_win > 0.08:
            for channel, y, x in self._special_pixels(graphic):
                p = (y, x)
                if p in reservations or graphic[y, x, STORAGE_ALLY] > 0.5:
                    continue
                distance = math.hypot(y - pos[0], x - pos[1])
                priority = 5.8 if channel in (9, 10) else 3.8
                candidates.append((priority - 0.12 * distance - 0.3 * local_index, p, "special"))

        if candidates:
            _, target, kind = max(candidates, key=lambda x: x[0])
            return target, kind

        # The field can be empty late in the game.  Raid a storage even if the currently
        # visible tile is empty; rotate through external storages instead of camping one
        # center forever.  Newly deposited/spawned items preempt this fallback immediately.
        raidable_storage = (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected_enemy
        return self._patrol_component_center(
            raidable_storage, local_index, period=260
        ), "patrol_raid"

    def _navigate(
        self,
        name: str,
        state: np.ndarray,
        states: np.ndarray,
        target: tuple[int, int] | None,
        kind: str,
        walkable: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if target is None:
            return np.zeros(2, dtype=np.float32)
        mem = self._memory.setdefault(name, _UnitMemory())
        pos_float = self._to_pixel_float(state[:2], shape)
        pos = (int(np.clip(round(pos_float[0]), 0, shape[0] - 1)),
               int(np.clip(round(pos_float[1]), 0, shape[1] - 1)))
        target_distance = math.hypot(target[0] - pos_float[0], target[1] - pos_float[1])

        # Escape actions must take precedence over the arrival dead-zone; otherwise a unit
        # whose interaction failed would be told to escape once and then immediately frozen
        # again by the early return on the following tick.
        if mem.escape_ticks > 0 and mem.escape_action is not None:
            mem.escape_ticks -= 1
            return mem.escape_action.astype(np.float32)

        # Continuous movement is normalized by Unity, and we normalize the policy output too.
        # Without an arrival dead-zone, even a 0.01-pixel residual becomes a full-speed action,
        # overshoots the center, and produces the visible left/right vibration on the next tick.
        arrival_radius = 0.38 if kind == "wait_storage" else 0.22
        if target_distance <= arrival_radius and kind != "hunt":
            arrival_key = (target, kind)
            if mem.arrival_key == arrival_key:
                mem.arrival_ticks += 1
            else:
                mem.arrival_key = arrival_key
                mem.arrival_ticks = 1
            mem.target, mem.target_kind = target, kind
            mem.path.clear()
            mem.recent_positions.clear()
            mem.stuck_ticks = 0

            # Pickup/deposit/transform should change observation state almost immediately.
            # If it does not, leave the interaction region and approach again.  This closes a
            # true permanent-idle path which sat before all stuck/oscillation detection.
            retry_kinds = {"transform", "deposit", "battery", "special", "steal"}
            if kind in retry_kinds and mem.arrival_ticks >= 24:
                neighbours = []
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    ny, nx = pos[0] + dy, pos[1] + dx
                    if 0 <= ny < shape[0] and 0 <= nx < shape[1] and walkable[ny, nx]:
                        neighbours.append((ny, nx))
                if neighbours:
                    waypoint = neighbours[(unit_index(name) + self._tick // 24) % len(neighbours)]
                    escape = np.array(
                        [waypoint[1] - pos_float[1], pos_float[0] - waypoint[0]],
                        dtype=np.float32,
                    )
                    norm = max(float(np.linalg.norm(escape)), 1e-6)
                    mem.escape_action = escape / norm
                    mem.escape_ticks = 12
                    mem.arrival_ticks = 0
                    mem.replan_in = 0
                    return mem.escape_action.astype(np.float32)
            return np.zeros(2, dtype=np.float32)
        mem.arrival_key = None
        mem.arrival_ticks = 0

        if mem.last_pos is not None and math.hypot(
            float(state[0] - mem.last_pos[0]), float(state[1] - mem.last_pos[1])
        ) < 2e-4:
            mem.stuck_ticks += 1
        else:
            mem.stuck_ticks = 0
        mem.last_pos = state[:2].copy()
        mem.replan_in -= 1
        mem.recent_positions.append(state[:2].copy())
        if len(mem.recent_positions) > 24:
            mem.recent_positions.pop(0)

        # Detect short limit cycles as well as complete immobility.  A unit oscillating over
        # the same boundary does move every tick, so the old last-position test never fired.
        oscillating = False
        if len(mem.recent_positions) >= 16:
            delta_net = mem.recent_positions[-1] - mem.recent_positions[-16]
            net = math.hypot(float(delta_net[0]), float(delta_net[1]))
            travelled = sum(
                math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
                for a, b in zip(mem.recent_positions[-16:-1], mem.recent_positions[-15:])
            )
            oscillating = net < 0.012 and travelled > 0.045

        moving_target = kind == "hunt"
        changed = mem.target != target or mem.target_kind != kind
        path_exhausted = not mem.path
        stuck = mem.stuck_ticks >= 12 or oscillating
        if changed or path_exhausted or stuck or mem.replan_in <= 0 or (moving_target and self._tick % 4 == 0):
            mem.target, mem.target_kind = target, kind
            mem.path = self._cached_astar(walkable, pos, target)
            mem.replan_in = 4 if moving_target else self.replan_interval

        # Follow cell centers closely.  Skipping two or three A* nodes made the continuous
        # steering chord cut inside corners even though the discrete path itself was valid.
        while mem.path and math.hypot(mem.path[0][0] - pos_float[0], mem.path[0][1] - pos_float[1]) < 0.42:
            mem.path.pop(0)
        if mem.path:
            waypoint = mem.path[0]
        else:
            # An empty A* result means the semantic grid has no connected route.  The old
            # fallback steered directly at the goal and could press against a wall forever.
            # Explore a valid neighbour in the current component until the target changes or
            # a later replan succeeds.
            neighbours = []
            for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                ny, nx = pos[0] + dy, pos[1] + dx
                if 0 <= ny < shape[0] and 0 <= nx < shape[1] and walkable[ny, nx]:
                    neighbours.append((ny, nx))
            if not neighbours:
                mem.stuck_ticks += 1
                return np.zeros(2, dtype=np.float32)
            phase = (self._tick // 18 + unit_index(name)) % len(neighbours)
            waypoint = neighbours[phase]
        # Graphic rows are top-down (the Unity sensor flips Texture2D's bottom-up rows), while
        # world/agent y grows upward.  A larger row therefore requires a negative y action.
        delta = np.array([waypoint[1] - pos_float[1], pos_float[0] - waypoint[0]], dtype=np.float32)

        # Cargo carriers and the fragile Carrier class steer away from nearby threats.  This
        # is intentionally local: global A* still makes progress toward the economic target.
        cls = self._class_id(state)
        if self._is_holding(state) or cls == CARRIER:
            for enemy in states[states[:, 2] < 0]:
                enemy_cls = self._class_id(enemy)
                dangerous = enemy_cls == HUNTER or cls == CARRIER
                if not dangerous:
                    continue
                away = state[:2] - enemy[:2]
                distance = float(np.linalg.norm(away))
                if 1e-5 < distance < self.threat_radius:
                    # normalized-coordinate repulsion converted to x/y action orientation.
                    delta += (away / distance) * ((self.threat_radius - distance) / self.threat_radius) * 9.0

        if stuck and mem.escape_ticks <= 0:
            # A multi-tick perpendicular escape breaks symmetric unit/unit contacts and limit
            # cycles.  One-tick nudges were immediately cancelled by the next path action.
            sign = -1.0 if unit_index(name) % 2 else 1.0
            escape = np.array([-delta[1] * sign, delta[0] * sign], dtype=np.float32)
            if float(np.linalg.norm(escape)) < 1e-6:
                escape = np.array([sign, 0.0], dtype=np.float32)
            mem.escape_action = escape / max(float(np.linalg.norm(escape)), 1e-6)
            mem.escape_ticks = 6
            mem.stuck_ticks = 0
            mem.recent_positions.clear()
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (delta / norm).astype(np.float32)

    @staticmethod
    def _to_pixel_float(position: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        return np.array([(1.0 - float(position[1])) * 0.5 * h - 0.5,
                         (float(position[0]) + 1.0) * 0.5 * w - 0.5], dtype=np.float32)

    @classmethod
    def _to_pixel(cls, position: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
        p = cls._to_pixel_float(position, shape)
        return (min(shape[0] - 1, max(0, int(round(float(p[0]))))),
                min(shape[1] - 1, max(0, int(round(float(p[1]))))))

    def _cached_protected_ally_storage_mask(self, graphic: np.ndarray) -> np.ndarray:
        cached = self._static_masks.get("protected_ally_storage")
        if cached is None:
            cached = self._protected_storage_mask(graphic, STORAGE_ALLY, SPAWN_ALLY)
            self._static_masks["protected_ally_storage"] = cached
        return cached

    def _defendable_storage_mask(self, graphic: np.ndarray) -> np.ndarray:
        """Own storage the enemy can actually reach. The component next to our own spawn sits inside
        the protected base (MapManager rejects enemy movement there), so guarding it or reacting to
        cargo near it wastes a unit."""
        return (graphic[..., STORAGE_ALLY] > 0.5) & ~self._cached_protected_ally_storage_mask(graphic)

    def _defend_patrol_target(self, graphic: np.ndarray, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Center of the defendable storage component nearest this unit; all own storage only if the
        map has no defendable storage at all."""
        mask = self._defendable_storage_mask(graphic)
        if not mask.any():
            mask = graphic[..., STORAGE_ALLY] > 0.5
        return self._nearest_component_center(mask, pos)

    def _cached_protected_enemy_storage_mask(self, graphic: np.ndarray) -> np.ndarray:
        """Cache topology-only protected storage inference for one episode."""
        cached = self._static_masks.get("protected_enemy_storage")
        if cached is None:
            cached = self._protected_enemy_storage_mask(graphic)
            self._static_masks["protected_enemy_storage"] = cached
        return cached

    def _cached_components(self, mask: np.ndarray) -> list[list[tuple[int, int]]]:
        """Cache connected components of immutable semantic-region masks."""
        key = (mask.shape, mask.tobytes())
        cached = self._component_cache.get(key)
        if cached is None:
            cached = self._components(mask)
            self._component_cache[key] = cached
        return cached

    def _battery_pixels(self, graphic: np.ndarray) -> list[tuple[int, int, float]]:
        """Scan dynamic battery pixels once per team decision tick."""
        if self._battery_scan_tick != self._tick:
            ys, xs = np.nonzero(graphic[..., BATTERY] > 1e-5)
            self._battery_scan = [
                (int(y), int(x), float(graphic[y, x, BATTERY]) * 15.0)
                for y, x in zip(ys.tolist(), xs.tolist())
            ]
            self._battery_scan_tick = self._tick
        return self._battery_scan

    def _special_pixels(self, graphic: np.ndarray) -> list[tuple[int, int, int]]:
        """Scan dynamic special-item pixels once per team decision tick."""
        if self._special_scan_tick != self._tick:
            found: list[tuple[int, int, int]] = []
            for channel in range(FIRST_SPECIAL, graphic.shape[-1]):
                ys, xs = np.nonzero(graphic[..., channel] > 0.5)
                found.extend(
                    (channel, int(y), int(x))
                    for y, x in zip(ys.tolist(), xs.tolist())
                )
            self._special_scan = found
            self._special_scan_tick = self._tick
        return self._special_scan

    def _cached_astar(
        self,
        walkable: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return a mutable copy of a deterministic path from a bounded episode LRU."""
        key = (start, goal)
        cached = self._path_cache.get(key)
        if cached is None:
            cached = tuple(self._astar(walkable, start, goal))
            self._path_cache[key] = cached
            if len(self._path_cache) > 4096:
                self._path_cache.popitem(last=False)
        else:
            self._path_cache.move_to_end(key)
        return list(cached)

    @staticmethod
    def _nearest_pixel(mask: np.ndarray, origin: tuple[int, int]) -> tuple[int, int] | None:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        idx = int(np.argmin((ys - origin[0]) ** 2 + (xs - origin[1]) ** 2))
        return int(ys[idx]), int(xs[idx])

    @classmethod
    def _nearest_component_center(cls, mask: np.ndarray, origin: tuple[int, int]) -> tuple[int, int] | None:
        """Center of the nearest 4-connected region, snapped to a pixel in that region."""
        components = cls._components(mask)
        if not components:
            return None
        component = min(
            components,
            key=lambda c: min((p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2 for p in c),
        )
        return cls._component_center(component)

    def _patrol_component_center(
        self,
        mask: np.ndarray,
        local_index: int,
        *,
        period: int,
    ) -> tuple[int, int] | None:
        """Deterministically rotate team members through region components."""
        components = self._cached_components(mask)
        if not components:
            return None
        # Stable spatial ordering makes the patrol reproducible and spreads teammates out.
        centers = sorted(self._component_center(c) for c in components)
        phase = self._tick // max(1, period)
        return centers[(local_index + phase) % len(centers)]

    @staticmethod
    def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        remaining = mask.copy()
        components: list[list[tuple[int, int]]] = []
        h, w = mask.shape
        for sy, sx in zip(*np.nonzero(mask)):
            sy, sx = int(sy), int(sx)
            if not remaining[sy, sx]:
                continue
            remaining[sy, sx] = False
            stack = [(sy, sx)]
            component: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
            components.append(component)
        return components

    @staticmethod
    def _component_center(component: list[tuple[int, int]]) -> tuple[int, int]:
        cy = sum(p[0] for p in component) / len(component)
        cx = sum(p[1] for p in component) / len(component)
        return min(component, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)

    @classmethod
    def _storage_target(
        cls,
        graphic: np.ndarray,
        state: np.ndarray,
        origin: tuple[int, int],
    ) -> tuple[tuple[int, int] | None, bool]:
        """Choose a storage that atomically fits the cargo, or wait just outside one.

        Storage.TryDistributeItem rejects the *whole* item if aggregate compatible capacity is
        insufficient.  Reconstructing that check from semantic pixels prevents a carrier from
        repeatedly crossing a full storage until the next absorption clears it.
        """
        storage_mask = graphic[..., STORAGE_ALLY] > 0.5
        components = cls._components(storage_mask)
        if not components:
            return None, False

        held_slot = int(np.argmax(state[3:9]))
        special_presence = np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1)

        def fits(component: list[tuple[int, int]]) -> bool:
            if held_slot == 1:  # battery; state stores amount / 15
                incoming = max(1, int(round(float(state[4]) * 15.0)))
                capacity = 0
                for y, x in component:
                    if special_presence[y, x]:
                        continue
                    amount = int(round(float(graphic[y, x, BATTERY]) * 15.0))
                    capacity += max(0, 10 - amount) if amount > 0 else 10
                return capacity >= incoming
            if held_slot >= 2:  # special items are single, non-stackable in current assets
                return any(
                    graphic[y, x, BATTERY] <= 1e-5 and not special_presence[y, x]
                    for y, x in component
                )
            return True

        distance = lambda c: min((p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2 for p in c)
        available = [c for c in components if fits(c)]
        if available:
            return cls._component_center(min(available, key=distance)), True

        # All storages are atomically full.  Stage on a non-storage, non-wall neighbour of
        # the nearest storage.  Absorption changes the map contents, after which the normal
        # capacity branch sends the unit into the newly available storage.
        component = min(components, key=distance)
        h, w = storage_mask.shape
        staging: set[tuple[int, int]] = set()
        for y, x in component:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and not storage_mask[ny, nx]
                        and graphic[ny, nx, WALL] < 0.5):
                    staging.add((ny, nx))
        if staging:
            return min(staging, key=lambda p: (p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2), False
        return cls._component_center(component), False

    @classmethod
    def _protected_enemy_storage_mask(cls, graphic: np.ndarray) -> np.ndarray:
        """Infer the unraidable home storage as the component nearest enemy spawn."""
        return cls._protected_storage_mask(graphic, STORAGE_ENEMY, SPAWN_ENEMY)

    @classmethod
    def _protected_storage_mask(cls, graphic: np.ndarray, storage_channel: int, spawn_channel: int) -> np.ndarray:
        """The storage component nearest a team's spawn, which sits inside that team's protected base."""
        storage = graphic[..., storage_channel] > 0.5
        result = np.zeros_like(storage)
        components = cls._components(storage)
        spawn_points = list(zip(*np.nonzero(graphic[..., spawn_channel] > 0.5)))
        if not components or not spawn_points:
            return result
        spawn = (int(spawn_points[0][0]), int(spawn_points[0][1]))
        protected = min(
            components,
            key=lambda c: min((y - spawn[0]) ** 2 + (x - spawn[1]) ** 2 for y, x in c),
        )
        for y, x in protected:
            result[y, x] = True
        return result

    @staticmethod
    def _astar(
        walkable: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """8-neighbour A* with diagonal corner-cut prevention."""
        h, w = walkable.shape
        if not (0 <= goal[0] < h and 0 <= goal[1] < w):
            return []
        if start == goal:
            return [goal]
        if _native.NUMBA_AVAILABLE:
            path_y, path_x = _native.astar_numba(walkable, start[0], start[1], goal[0], goal[1])
            return list(zip(path_y.tolist(), path_x.tolist()))
        neighbours = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                      (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))
        queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start: 0.0}
        reached: tuple[int, int] | None = None
        while queue:
            _, g, current = heapq.heappop(queue)
            if g != cost.get(current):
                continue
            if current == goal:
                reached = current
                break
            cy, cx = current
            for dy, dx, step in neighbours:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < h and 0 <= nx < w) or not walkable[ny, nx]:
                    continue
                if dy and dx and (not walkable[cy, nx] or not walkable[ny, cx]):
                    continue
                ng = g + step
                nxt = (ny, nx)
                if ng >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = ng
                came[nxt] = current
                heuristic = math.hypot(goal[0] - ny, goal[1] - nx)
                heapq.heappush(queue, (ng + heuristic, ng, nxt))
        if reached is None:
            return []
        path = [reached]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        return path
