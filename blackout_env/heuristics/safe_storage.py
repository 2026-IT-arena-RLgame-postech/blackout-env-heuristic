"""Third-generation heuristic: V2 coordination plus risk-aware deposits."""

from __future__ import annotations

import math

import numpy as np

from .advanced import StrategicHeuristicV2
from .strategic import (
    ABSORPTION_INTERVAL_SECONDS, BATTERY, CARRIER, COLLECTOR, FIRST_SPECIAL, HUNTER, SPAWN_ALLY,
    STORAGE_ALLY, WALL,
)


class StrategicHeuristicV3(StrategicHeuristicV2):
    """Preserve cargo by preferring reachable protected storage and avoiding ambushes."""

    def __init__(self, *, protected_storage_bonus: float = 7.0, **kwargs):
        super().__init__(**kwargs)
        self.protected_storage_bonus = max(0.0, float(protected_storage_bonus))

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
        if self._is_holding(state):
            origin = self._to_pixel(state[:2], graphic.shape[:2])
            target, can_deposit = self._risk_aware_storage_target(
                graphic, state, states, origin, team_state
            )
            return target, "deposit" if can_deposit else "wait_storage"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )

    def _risk_aware_storage_target(
        self,
        graphic: np.ndarray,
        state: np.ndarray,
        states: np.ndarray,
        origin: tuple[int, int],
        team_state: np.ndarray,
    ) -> tuple[tuple[int, int] | None, bool]:
        components = self._cached_components(graphic[..., STORAGE_ALLY] > 0.5)
        if not components:
            return None, False
        available = self._fitting_storage_components(graphic, state, components)
        if not available:
            return self._storage_target(graphic, state, origin)
        costs = self._storage_component_costs(graphic, state, states, origin, team_state, components, available)
        chosen = min(costs, key=lambda pair: pair[0])[1]
        return self._component_center(chosen), True

    def _fitting_storage_components(
        self, graphic: np.ndarray, state: np.ndarray, components: list[list[tuple[int, int]]]
    ) -> list[list[tuple[int, int]]]:
        """Storages whose free capacity takes the held item whole (Storage.TryDistributeItem)."""
        held_slot = int(np.argmax(state[3:9]))
        special_presence = np.any(graphic[..., FIRST_SPECIAL:] > 0.5, axis=-1)

        def fits(component):
            if held_slot == 1:
                incoming = max(1, int(round(float(state[4]) * 15.0)))
                capacity = 0
                for y, x in component:
                    if special_presence[y, x]:
                        continue
                    amount = int(round(float(graphic[y, x, BATTERY]) * 15.0))
                    capacity += max(0, 10 - amount) if amount > 0 else 10
                return capacity >= incoming
            if held_slot >= 2:
                return any(
                    graphic[y, x, BATTERY] <= 1e-5 and not special_presence[y, x]
                    for y, x in component
                )
            return True

        return [component for component in components if fits(component)]

    def _storage_component_costs(
        self,
        graphic: np.ndarray,
        state: np.ndarray,
        states: np.ndarray,
        origin: tuple[int, int],
        team_state: np.ndarray,
        components: list[list[tuple[int, int]]],
        available: list[list[tuple[int, int]]],
    ) -> list[tuple[float, list[tuple[int, int]]]]:
        """(cost, component) for every storage that fits the cargo; the minimum is chosen."""
        spawn = self._nearest_pixel(graphic[..., SPAWN_ALLY] > 0.5, origin)
        protected = None if spawn is None else min(
            components,
            key=lambda component: min(math.dist(point, spawn) for point in component),
        )
        unit_class = self._class_id(state)
        speed = 6.0 if unit_class == CARRIER else 4.0
        seconds_to_absorption = max(0.0, float(team_state[3]) * ABSORPTION_INTERVAL_SECONDS)
        enemy_states = states[states[:, 2] < 0]

        def cost(component):
            center = self._component_center(component)
            distance = self._storage_distance(graphic, origin, component)
            is_protected = component is protected
            # A home deposit is unraidable and its carrier cannot be intercepted inside base.
            # Close to absorption, use any substantially nearer store so cargo still lands in
            # time; otherwise the safety premium is roughly three map tiles of travel.
            safety = -self.protected_storage_bonus if is_protected else 0.0
            if seconds_to_absorption < distance / speed + 0.45:
                safety *= 0.2
            threat = 0.0
            if not is_protected:
                for enemy in enemy_states:
                    enemy_class = self._class_id(enemy)
                    dangerous = enemy_class == HUNTER or unit_class == CARRIER
                    if not dangerous:
                        continue
                    enemy_pixel = self._to_pixel(enemy[:2], graphic.shape[:2])
                    separation = math.dist(center, enemy_pixel)
                    threat += max(0.0, 6.0 - separation) * 1.6
            return distance + safety + threat

        return [(cost(component), component) for component in available]

    def _storage_distance(
        self, graphic: np.ndarray, origin: tuple[int, int], component: list[tuple[int, int]]
    ) -> float:
        return math.dist(origin, self._component_center(component))
