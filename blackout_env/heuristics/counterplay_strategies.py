"""Intentionally polarised heuristic teachers for counterplay-rich datasets.

These policies trade broad Elo strength for clear, repeatable strategic commitments.
They are useful for behaviour cloning and offline RL because a learner sees both the
strategy *and* the situations in which that strategy is punishable:

* :class:`StrategicHeuristicV13` keeps a three-unit strike on external enemy storage.
* :class:`StrategicHeuristicV14` spends an early unit on protecting home storage.
* :class:`StrategicHeuristicV15` forgoes a Hunter and commits to Carrier throughput.

Each policy inherits the proven navigation/recovery behaviour of the previous
heuristics.  The difference is a strategic objective, not a less reliable pathfinder.
"""

from __future__ import annotations

import math

import numpy as np

from .dynamic_roles import StrategicHeuristicV7
from .phase_strategies import StrategicHeuristicV10, StrategicHeuristicV11
from .strategic import (
    ABSORPTION_INTERVAL_SECONDS, BATTERY, CARRIER, HUNTER, STORAGE_ALLY, STORAGE_ENEMY,
)


class StrategicHeuristicV13(StrategicHeuristicV11):
    """Storage-siege teacher: persistently steals from every exposed enemy depot.

    V11 raids only during an absorption window.  V13 deliberately maintains a wider,
    earlier strike team whenever visible external storage holds value.  This is often
    effective against passive economy or fortress policies, but is intentionally exposed
    to a home guardian and to teams that win the field-battery race.
    """

    def __init__(
        self, *, raid_slots: int = 3, siege_min_value: float = 1.0, **kwargs,
    ):
        super().__init__(raid_slots=raid_slots, raid_absorption=0.95, **kwargs)
        self.siege_min_value = max(0.0, float(siege_min_value))
        self.style_label = "storage_siege"

    def _update_mode(self, sample) -> None:
        graphic = sample["graphic"]
        protected = self._cached_protected_enemy_storage_mask(graphic)
        exposed_value = float(np.sum(graphic[..., BATTERY][
            (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected
        ]) * 15.0)
        # No absorption-clock gate: losing a few race-to-absorption opportunities is the
        # intentional cost of this policy's constant strategic pressure.
        self._set_mode_candidate(
            "raid_window" if exposed_value >= self.siege_min_value else "harvest"
        )

class StrategicHeuristicV14(StrategicHeuristicV7):
    """Counter-raid sentinel: make one early Hunter defend, rather than score.

    The Hunter is created immediately and only chases cargo that has entered the home
    perimeter.  Without a threat it patrols friendly storage instead of crossing the map.
    That commitment should counter prolonged raiding policies, while losing efficiency to
    a fast field-economy or carrier-rush opponent.
    """

    def __init__(self, *, guard_radius: float = 7.0, **kwargs):
        super().__init__(
            carrier_quota=0,
            hunter_quota=1,
            hunter_activation_time=1.0,
            hunter_activation_field_battery=0.0,
            **kwargs,
        )
        self.guard_radius = max(1.0, float(guard_radius))
        self.current_mode = "home_guard"

    def reset(self) -> None:
        super().reset()
        self.current_mode = "home_guard"

    def _update_roles(self, obs, sample) -> None:
        super()._update_roles(obs, sample)
        states = sample["agent_states"]
        graphic = sample["graphic"]
        home_points = list(zip(*np.nonzero(self._defendable_storage_mask(graphic))))
        cargo_near_home = any(
            enemy[2] < 0 and self._is_holding(enemy)
            and home_points
            and min(math.dist(self._to_pixel(enemy[:2], graphic.shape[:2]), point)
                    for point in home_points) <= self.guard_radius
            for enemy in states
        )
        self.current_mode = "counter_intercept" if cargo_near_home else "home_guard"
        self.role_assignments = {
            name: f"{self.current_mode}:{role}" for name, role in self.role_assignments.items()
        }

    def _choose_target(
        self, role, state, states, graphic, reservations, local_index, team_state,
    ):
        if self._class_id(state) == HUNTER:
            home_points = list(zip(*np.nonzero(self._defendable_storage_mask(graphic))))
            threats = []
            for enemy in states:
                if enemy[2] >= 0 or not self._is_holding(enemy) or not home_points:
                    continue
                pixel = self._to_pixel(enemy[:2], graphic.shape[:2])
                if min(math.dist(pixel, point) for point in home_points) <= self.guard_radius:
                    threats.append(enemy)
            if threats:
                threat = min(threats, key=lambda enemy: np.linalg.norm(enemy[:2] - state[:2]))
                return self._to_pixel(threat[:2], graphic.shape[:2]), "hunt"
            return self._defend_patrol_target(graphic, self._to_pixel(state[:2], graphic.shape[:2])), "patrol_defend"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )


class StrategicHeuristicV15(StrategicHeuristicV7):
    """Convoy-rush teacher: one early Carrier, no Hunter, high-value field battery focus."""

    def __init__(self, *, cargo_amount_weight: float = 20.0, **kwargs):
        super().__init__(carrier_quota=1, hunter_quota=0, **kwargs)
        self.cargo_amount_weight = max(0.0, float(cargo_amount_weight))
        self.current_mode = "convoy_rush"

    def reset(self) -> None:
        super().reset()
        self.current_mode = "convoy_rush"

    def _update_roles(self, obs, sample) -> None:
        super()._update_roles(obs, sample)
        self.role_assignments = {
            name: f"convoy_rush:{role}" for name, role in self.role_assignments.items()
        }

    def _assign_economic_tasks(
        self, names, row_for, states, graphic, team_state, walkable,
    ):
        result = super()._assign_economic_tasks(
            names, row_for, states, graphic, team_state, walkable
        )
        # Once transformed, reserve the Carrier for large *field* batteries.  It never raids
        # enemy storage: that creates a clear throughput-vs-denial trade-off with V13.
        carrier_names = [
            name for name in names if self._class_id(states[row_for[name]]) == CARRIER
        ]
        if not carrier_names:
            return result
        carrier = carrier_names[0]
        occupied = {target for name, (target, _) in result.items() if name != carrier}
        state = states[row_for[carrier]]
        start = self._to_pixel(state[:2], graphic.shape[:2])
        distances = self._cached_distance_map(walkable, start)
        candidates = []
        for y, x, amount in self._battery_pixels(graphic):
            target = (y, x)
            if target in occupied or graphic[y, x, STORAGE_ALLY] > 0.5 or graphic[y, x, STORAGE_ENEMY] > 0.5:
                continue
            distance = float(distances[target])
            if np.isfinite(distance):
                candidates.append((amount * self.cargo_amount_weight - distance * 0.20, target))
        if candidates:
            result[carrier] = (max(candidates, key=lambda item: item[0])[1], "battery")
        return result


class StrategicHeuristicV16(StrategicHeuristicV10):
    """V10's phase director extended with siege, sentinel, and convoy modes.

    This is intentionally a new immutable policy ID rather than a mutation of V10.  It
    retains V10's opening/pressure/closeout modes and selects the three counterplay plans
    from public state: exposed enemy storage activates ``storage_siege``, threatening cargo
    activates ``home_guard``, and an early rich field activates ``convoy_rush``.
    """

    def __init__(
        self,
        *,
        siege_value: float = 7.0,
        guard_radius: float = 7.0,
        siege_raid_slots: int = 3,
        convoy_field_battery: float = 55.0,
        convoy_min_time_left: float = 0.76,
        convoy_cargo_amount_weight: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.siege_value = max(0.0, float(siege_value))
        self.guard_radius = max(1.0, float(guard_radius))
        # V11's raid assignment is deliberately reused in storage_siege mode.
        self.raid_slots = max(1, min(3, int(siege_raid_slots)))
        self.convoy_field_battery = max(0.0, float(convoy_field_battery))
        self.convoy_min_time_left = float(np.clip(convoy_min_time_left, 0.0, 1.0))
        self.convoy_cargo_amount_weight = max(0.0, float(convoy_cargo_amount_weight))
        self._configured_carrier_quota = self.carrier_quota

    def reset(self) -> None:
        super().reset()
        self.carrier_quota = self._configured_carrier_quota

    def _storage_values(self, graphic: np.ndarray) -> tuple[float, float]:
        protected = self._cached_protected_enemy_storage_mask(graphic)
        external = float(np.sum(graphic[..., BATTERY][
            (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected
        ]) * 15.0)
        field = float(np.sum(graphic[..., BATTERY][
            (graphic[..., STORAGE_ALLY] < 0.5) & (graphic[..., STORAGE_ENEMY] < 0.5)
        ]) * 15.0)
        return external, field

    def _enemy_cargo_near_home(self, states: np.ndarray, graphic: np.ndarray) -> bool:
        home_points = list(zip(*np.nonzero(self._defendable_storage_mask(graphic))))
        return any(
            enemy[2] < 0 and self._is_holding(enemy) and home_points
            and min(math.dist(self._to_pixel(enemy[:2], graphic.shape[:2]), point)
                    for point in home_points) <= self.guard_radius
            for enemy in states
        )

    def _desired_mode(self, sample) -> str:
        graphic = sample["graphic"]
        states = sample["agent_states"]
        _, _, time_left, _ = map(float, sample["team_state"][:4])
        external_value, field_value = self._storage_values(graphic)
        # Protecting an actually threatened deposit always wins over a greedier plan.
        if self._enemy_cargo_near_home(states, graphic):
            return "home_guard"
        # The siege threshold is deliberately high: V16 is not permanently a V13 clone.
        if external_value >= self.siege_value and time_left >= 0.20:
            return "storage_siege"
        if field_value >= self.convoy_field_battery and time_left >= self.convoy_min_time_left:
            return "convoy_rush"
        return super()._desired_mode(sample)

    def _update_roles(self, obs, sample) -> None:
        self._set_mode_candidate(self._desired_mode(sample))
        states = sample["agent_states"]
        own_hunter_exists = any(
            state[2] > 0 and self._class_id(state) == HUNTER for state in states
        )
        own_carrier_exists = any(
            state[2] > 0 and self._class_id(state) == CARRIER for state in states
        )
        # Role transformations are irreversible.  A mode can decide only future commitment;
        # an already transformed specialist remains a valid unit with a new target.
        self.carrier_quota = (
            self._configured_carrier_quota
            if own_carrier_exists or self.current_mode != "home_guard" else 0
        )
        self.hunter_quota = (
            self._configured_hunter_quota
            if own_hunter_exists or self.current_mode in {"home_guard", "pressure_raid", "closeout_defend"}
            else 0
        )
        # Bypass V10's narrower three-mode quota controller but retain V7's validated,
        # quota-aware assignment implementation.
        StrategicHeuristicV7._update_roles(self, obs, sample)
        self.role_assignments = {
            name: f"{self.current_mode}:{role}"
            for name, role in self.role_assignments.items()
        }

    def _assign_economic_tasks(
        self, names, row_for, states, graphic, team_state, walkable,
    ):
        if self.current_mode == "storage_siege":
            # This mirrors V11's bounded raid matcher, but lives here instead of calling its
            # method directly: V16 is a V10 subclass, so V11's zero-argument ``super()``
            # cannot safely be borrowed across the unrelated inheritance branch.
            protected = self._cached_protected_enemy_storage_mask(graphic)
            steals = [
                ((y, x), amount)
                for y, x, amount in self._battery_pixels(graphic)
                if graphic[y, x, STORAGE_ENEMY] > 0.5 and not protected[y, x]
            ]
            if not steals:
                return super()._assign_economic_tasks(
                    names, row_for, states, graphic, team_state, walkable
                )
            absorption_seconds = max(0.0, float(team_state[3]) * ABSORPTION_INTERVAL_SECONDS)
            pairs: list[tuple[float, str, int]] = []
            for name in names:
                state = states[row_for[name]]
                start = self._to_pixel(state[:2], graphic.shape[:2])
                distances = self._cached_distance_map(walkable, start)
                speed = 6.0 if self._class_id(state) == CARRIER else 4.0
                for task_index, (target, amount) in enumerate(steals):
                    distance = float(distances[target])
                    if not np.isfinite(distance) or distance / speed + 0.35 >= absorption_seconds:
                        continue
                    pairs.append((amount * 8.0 - distance * 0.22, name, task_index))
            pairs.sort(reverse=True, key=lambda pair: pair[0])
            assigned, used, result = set(), set(), {}
            for _, name, task_index in pairs:
                if name in assigned or task_index in used:
                    continue
                result[name] = (steals[task_index][0], "steal")
                assigned.add(name)
                used.add(task_index)
                if len(result) >= self.raid_slots:
                    break
            return result

        result = super()._assign_economic_tasks(
            names, row_for, states, graphic, team_state, walkable
        )
        if self.current_mode != "convoy_rush":
            return result
        carrier_names = [
            name for name in names if self._class_id(states[row_for[name]]) == CARRIER
        ]
        if not carrier_names:
            return result
        carrier = carrier_names[0]
        occupied = {target for name, (target, _) in result.items() if name != carrier}
        state = states[row_for[carrier]]
        start = self._to_pixel(state[:2], graphic.shape[:2])
        distances = self._cached_distance_map(walkable, start)
        candidates = []
        for y, x, amount in self._battery_pixels(graphic):
            target = (y, x)
            if target in occupied or graphic[y, x, STORAGE_ALLY] > 0.5 or graphic[y, x, STORAGE_ENEMY] > 0.5:
                continue
            distance = float(distances[target])
            if np.isfinite(distance):
                candidates.append((amount * self.convoy_cargo_amount_weight - distance * 0.20, target))
        if candidates:
            result[carrier] = (max(candidates, key=lambda item: item[0])[1], "battery")
        return result

    def _choose_target(
        self, role, state, states, graphic, reservations, local_index, team_state,
    ):
        if self.current_mode == "home_guard" and self._class_id(state) == HUNTER:
            home_points = list(zip(*np.nonzero(self._defendable_storage_mask(graphic))))
            threats = [
                enemy for enemy in states
                if enemy[2] < 0 and self._is_holding(enemy) and home_points
                and min(math.dist(self._to_pixel(enemy[:2], graphic.shape[:2]), point)
                        for point in home_points) <= self.guard_radius
            ]
            if threats:
                threat = min(threats, key=lambda enemy: np.linalg.norm(enemy[:2] - state[:2]))
                return self._to_pixel(threat[:2], graphic.shape[:2]), "hunt"
            return self._defend_patrol_target(graphic, self._to_pixel(state[:2], graphic.shape[:2])), "patrol_defend"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )
