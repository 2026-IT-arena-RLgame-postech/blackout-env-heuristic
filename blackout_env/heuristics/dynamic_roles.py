"""V7 (``strategic_v7``, parent V4): quota-aware, phase-sensitive role assignment.

V1-V6 fix roles by unit order (unit 0 Carrier, unit 1 Hunter), so a far-away unit may walk
across the map to transform and the Hunter is made at kick-off whether or not it has prey.
V7 assigns roles dynamically every tick: the Carrier role goes to the empty Collector with
the shortest path to the Carrier sanctuary (at most one Carrier -- a game limit), and the
Hunter is only committed once there is something to hunt (enemy cargo or a Carrier), the
match is past ``hunter_activation_time``, or the field is running out of batteries.  The
economy is V4's.  V8-V10, V12 and V14-V16 build on this role layer.
"""

from __future__ import annotations

import numpy as np

from ..env.constants import unit_index
from .spread_deposit import StrategicHeuristicV4
from .strategic import (
    BATTERY, CARRIER, COLLECTOR, HUNTER, SITE_CARRIER, SITE_HUNTER,
    STORAGE_ALLY, STORAGE_ENEMY, WALL,
)


class StrategicHeuristicV7(StrategicHeuristicV4):
    """V4 economics with explicit specialist quotas and delayed Hunter commitment."""

    def __init__(
        self,
        *,
        carrier_quota: int = 1,
        hunter_quota: int = 1,
        hunter_activation_time: float = 0.82,
        hunter_activation_field_battery: float = 90.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.carrier_quota = max(0, min(1, int(carrier_quota)))  # game hard-limit is one
        self.hunter_quota = max(0, int(hunter_quota))
        self.hunter_activation_time = float(hunter_activation_time)
        self.hunter_activation_field_battery = float(hunter_activation_field_battery)
        self._carrier_assignee: int | None = None
        self._hunter_assignee: int | None = None
        self._hunter_committed = False
        self._active_roles: dict[int, str] = {}
        self.role_assignments: dict[str, str] = {}

    def reset(self) -> None:
        super().reset()
        self._carrier_assignee = None
        self._hunter_assignee = None
        self._hunter_committed = False
        self._active_roles.clear()
        self.role_assignments.clear()

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if obs:
            sample = next(iter(obs.values()))
            time_left = float(sample["team_state"][2])
            if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
                self.reset()
            self._update_roles(obs, sample)
        return super().act(obs)

    def _role(self, local_index: int) -> str:
        """Role chosen by ``_update_roles`` this tick (``use_specialists`` is not consulted)."""
        return self._active_roles.get(local_index, "collector")

    def _update_roles(self, obs, sample) -> None:
        """Recompute ``_active_roles`` for this tick.

        An existing Carrier/Hunter keeps its role; otherwise the chosen assignee is kept for
        the match, so a respawned specialist walks back to its sanctuary.  Hunter commitment is sticky: once any trigger fires (enemy
        transport, time_left <= hunter_activation_time, field battery <= threshold) it stays
        on for the match.  ``role_assignments`` (name -> role) is exposed for provenance.
        """
        graphic = sample["graphic"]
        states = sample["agent_states"]
        team_state = sample["team_state"]
        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}
        local_for_row = {row_for[name]: local for local, name in enumerate(controlled)}
        self._active_roles = {local: "collector" for local in range(len(controlled))}

        carrier_rows = [row for row in own_rows if self._class_id(states[row]) == CARRIER]
        if self.carrier_quota and carrier_rows:
            self._carrier_assignee = local_for_row.get(carrier_rows[0], self._carrier_assignee)
        elif self.carrier_quota and self._carrier_assignee is None:
            self._carrier_assignee = self._best_transform_candidate(
                controlled, row_for, states, graphic, SITE_CARRIER, excluded=set()
            )
        if self._carrier_assignee is not None and self.carrier_quota:
            self._active_roles[self._carrier_assignee] = "carrier"

        enemies = states[states[:, 2] < 0]
        enemy_transport_active = any(
            self._is_holding(enemy) or self._class_id(enemy) == CARRIER for enemy in enemies
        )
        field_mask = (
            (graphic[..., STORAGE_ALLY] < 0.5)
            & (graphic[..., STORAGE_ENEMY] < 0.5)
        )
        field_battery = float(np.sum(graphic[..., BATTERY][field_mask]) * 15.0)
        if (
            self.hunter_quota > 0
            and (enemy_transport_active
                 or float(team_state[2]) <= self.hunter_activation_time
                 or field_battery <= self.hunter_activation_field_battery)
        ):
            self._hunter_committed = True

        hunter_rows = [row for row in own_rows if self._class_id(states[row]) == HUNTER]
        if hunter_rows:
            self._hunter_committed = True
            self._hunter_assignee = local_for_row.get(hunter_rows[0], self._hunter_assignee)
        elif self._hunter_committed and self._hunter_assignee is None:
            excluded = {self._carrier_assignee} if self._carrier_assignee is not None else set()
            self._hunter_assignee = self._best_transform_candidate(
                controlled, row_for, states, graphic, SITE_HUNTER, excluded=excluded
            )
        if self._hunter_assignee is not None and self.hunter_quota > 0:
            self._active_roles[self._hunter_assignee] = "hunter"

        self.role_assignments = {
            name: self._active_roles[local] for local, name in enumerate(controlled)
        }

    def _best_transform_candidate(
        self, controlled, row_for, states, graphic, site_channel, *, excluded
    ) -> int | None:
        """Local index of the empty Collector with the shortest path to ``site_channel``."""
        walkable = graphic[..., WALL] < 0.5
        site_pixels = list(zip(*np.nonzero(graphic[..., site_channel] > 0.5)))
        if not site_pixels:
            return None
        candidates = []
        for local, name in enumerate(controlled):
            if local in excluded:
                continue
            state = states[row_for[name]]
            if self._class_id(state) != COLLECTOR or self._is_holding(state):
                continue
            start = self._to_pixel(state[:2], graphic.shape[:2])
            distances = self._distance_map(walkable, start)
            distance = min((float(distances[p]) for p in site_pixels), default=float("inf"))
            if np.isfinite(distance):
                candidates.append((distance, local))
        return min(candidates)[1] if candidates else None
