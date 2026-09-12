"""Fifth-generation heuristic: public-state predictive Hunter interception."""

from __future__ import annotations

import math

import numpy as np

from .spread_deposit import StrategicHeuristicV4
from .strategic import CARRIER, HUNTER, SPAWN_ENEMY, STORAGE_ENEMY, WALL


class StrategicHeuristicV5(StrategicHeuristicV4):
    """V4 plus route-ensemble interception of enemies carrying cargo."""

    def __init__(
        self,
        *,
        intercept_margin_seconds: float = 0.24,
        intent_temperature: float = 4.0,
        velocity_ema: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intercept_margin_seconds = float(intercept_margin_seconds)
        self.intent_temperature = max(0.1, float(intent_temperature))
        self.velocity_ema = float(np.clip(velocity_ema, 0.0, 0.95))
        self._previous_positions: dict[int, np.ndarray] = {}
        self._enemy_velocities: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        super().reset()
        self._previous_positions.clear()
        self._enemy_velocities.clear()

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if obs:
            sample = next(iter(obs.values()))
            time_left = float(sample["team_state"][2])
            if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
                self.reset()
            self._update_enemy_motion(sample["agent_states"])
        return super().act(obs)

    def _update_enemy_motion(self, states: np.ndarray) -> None:
        for row, enemy in enumerate(states):
            if enemy[2] >= 0:
                continue
            current = enemy[:2].copy()
            previous = self._previous_positions.get(row)
            if previous is None or float(np.linalg.norm(current - previous)) > 0.2:
                velocity = np.zeros(2, dtype=np.float32)  # new episode or respawn teleport
            else:
                measured = current - previous
                old = self._enemy_velocities.get(row, np.zeros(2, dtype=np.float32))
                velocity = self.velocity_ema * old + (1.0 - self.velocity_ema) * measured
            self._previous_positions[row] = current
            self._enemy_velocities[row] = velocity.astype(np.float32)

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
        if self._class_id(state) == HUNTER:
            intercept = self._best_cargo_intercept(state, states, graphic)
            if intercept is not None:
                return intercept, "hunt"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )

    def _best_cargo_intercept(
        self,
        hunter: np.ndarray,
        states: np.ndarray,
        graphic: np.ndarray,
    ) -> tuple[int, int] | None:
        walkable = graphic[..., WALL] < 0.5
        hunter_pixel = self._to_pixel(hunter[:2], graphic.shape[:2])
        hunter_distances = self._distance_map(walkable, hunter_pixel)
        storage_centers = [
            self._component_center(component)
            for component in self._components(graphic[..., STORAGE_ENEMY] > 0.5)
        ]
        if not storage_centers:
            return None
        protected = self._protected_enemy_storage_mask(graphic)
        enemy_spawn = self._nearest_pixel(graphic[..., SPAWN_ENEMY] > 0.5, hunter_pixel)
        hunter_cells_per_tick = 6.0 / 50.0
        margin_ticks = self.intercept_margin_seconds * 50.0
        best: tuple[float, tuple[int, int]] | None = None

        for row, enemy in enumerate(states):
            if enemy[2] >= 0 or not self._is_holding(enemy):
                continue
            enemy_pixel = self._to_pixel(enemy[:2], graphic.shape[:2])
            enemy_class = self._class_id(enemy)
            enemy_cells_per_tick = (6.0 if enemy_class == CARRIER else 4.0) / 50.0
            velocity = self._enemy_velocities.get(row, np.zeros(2, dtype=np.float32))
            # Convert normalized world velocity to graphic col/row velocity.  It is used only
            # as a soft intent cue; route feasibility remains entirely grid/path based.
            velocity_pixel = np.array(
                [-velocity[1] * graphic.shape[0] * 0.5,
                 velocity[0] * graphic.shape[1] * 0.5], dtype=np.float32,
            )
            routes = []
            for storage in storage_centers:
                path = self._astar(walkable, enemy_pixel, storage)
                if path:
                    heading = np.asarray(path[min(2, len(path) - 1)], dtype=np.float32) - enemy_pixel
                    alignment = 0.0
                    if np.linalg.norm(velocity_pixel) > 1e-5 and np.linalg.norm(heading) > 1e-5:
                        alignment = float(np.dot(
                            velocity_pixel / np.linalg.norm(velocity_pixel),
                            heading / np.linalg.norm(heading),
                        ))
                    routes.append((len(path) - 2.0 * alignment, path))
            routes.sort(key=lambda item: item[0])
            routes = routes[:3]
            if not routes:
                continue
            route_costs = np.asarray([cost for cost, _ in routes], dtype=np.float64)
            weights = np.exp(-(route_costs - route_costs.min()) / self.intent_temperature)
            weights /= weights.sum()
            cargo_slot = int(np.argmax(enemy[3:9]))
            cargo_value = float(enemy[4] * 15.0) if cargo_slot == 1 else 6.0

            for weight, (_, path) in zip(weights, routes):
                travelled = 0.0
                previous = path[0]
                for node in path[1:]:
                    travelled += math.dist(previous, node)
                    previous = node
                    if protected[node]:
                        continue
                    if enemy_spawn is not None and math.dist(node, enemy_spawn) <= 3.5:
                        continue
                    hunter_distance = float(hunter_distances[node])
                    if not np.isfinite(hunter_distance):
                        continue
                    enemy_ticks = travelled / enemy_cells_per_tick
                    hunter_ticks = hunter_distance / hunter_cells_per_tick
                    if hunter_ticks + margin_ticks >= enemy_ticks:
                        continue
                    score = (
                        float(weight) * cargo_value * math.exp(-0.018 * hunter_ticks)
                        + 0.035 * (enemy_ticks - hunter_ticks - margin_ticks)
                        - 0.018 * hunter_distance
                    )
                    if best is None or score > best[0]:
                        best = (score, node)
        return None if best is None else best[1]

