"""V8 (``strategic_v8``, parent V7): deliberate death as a guarded role-respec mechanism.

A specialist can only become a Collector again by dying.  When a Hunter has nothing worth
hunting while the team trails and batteries remain, V8 sends it into the nearest enemy Hunter
(Hunters kill each other on contact); the respawned Collector then stays a Collector for a
cooldown so it does not immediately re-transform.  The gates are strict, so in practice the
respec rarely fires and V8 plays almost like V7 (it shares one mixture budget with V7/V9/V12).
V9 is the same machine with looser gates.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.constants import unit_index
from .dynamic_roles import StrategicHeuristicV7
from .strategic import (
    BATTERY, CARRIER, COLLECTOR, HUNTER, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY,
)


class StrategicHeuristicV8(StrategicHeuristicV7):
    """V7 plus conservative Hunter retirement through a mutual-Hunter collision.

    Death is the only legal way to turn a specialist back into a Collector. A Hunter is
    retired only when it has had no transport target for a sustained period, the team is
    materially behind, useful field batteries remain, and enough time is left to recover.
    """

    def __init__(
        self,
        *,
        respec_inactivity_ticks: int = 150,
        respec_score_gap: float = 0.05,
        respec_min_field_battery: float = 20.0,
        respec_min_time_left: float = 0.20,
        respec_cooldown_ticks: int = 600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.respec_inactivity_ticks = max(1, int(respec_inactivity_ticks))
        self.respec_score_gap = max(0.0, float(respec_score_gap))
        self.respec_min_field_battery = max(0.0, float(respec_min_field_battery))
        self.respec_min_time_left = float(respec_min_time_left)
        self.respec_cooldown_ticks = max(1, int(respec_cooldown_ticks))
        self._transport_inactive_ticks = 0
        self._respec_local: int | None = None
        self._hunter_cooldown = 0
        self.respec_attempts = 0
        self.respec_completions = 0
        self.respec_diagnostics: dict[str, int] = self._new_diagnostics()

    def reset(self) -> None:
        super().reset()
        self._transport_inactive_ticks = 0
        self._respec_local = None
        self._hunter_cooldown = 0
        self.respec_attempts = 0
        self.respec_completions = 0
        self.respec_diagnostics = self._new_diagnostics()

    @staticmethod
    def _new_diagnostics() -> dict[str, int]:
        return {
            "max_inactive_ticks": 0,
            "trailing_ticks": 0,
            "both_hunters_ticks": 0,
            "all_gates_ticks": 0,
        }

    def _update_roles(self, obs, sample) -> None:
        """V7 roles, then advance the respec state machine.

        States: normal -> respec (``_respec_local`` set, its Hunter seeks an enemy Hunter)
        -> completed when that unit is observed as a Collector -> cooldown, during which V7's
        Hunter commitment is masked.  A respec starts only when every gate holds: no
        high-value enemy transport for ``respec_inactivity_ticks``, trailing by at least
        ``respec_score_gap``, time_left >= ``respec_min_time_left``, field battery >=
        ``respec_min_field_battery``, and both an own Hunter and an enemy Hunter outside its
        base exist.  Gate counters go to ``respec_diagnostics``.
        """
        super()._update_roles(obs, sample)
        states = sample["agent_states"]
        graphic = sample["graphic"]
        team_state = sample["team_state"]
        controlled = sorted(obs, key=unit_index)
        own_rows = [i for i in range(len(states)) if states[i, 2] > 0]
        row_for = {name: unit_index(name) for name in controlled}
        if any(i >= len(states) or states[i, 2] <= 0 for i in row_for.values()):
            row_for = {name: own_rows[j] for j, name in enumerate(controlled)}

        enemies = states[states[:, 2] < 0]
        # Do not keep a non-collecting Hunter forever merely because a Worker is moving a
        # one-point battery. Carrier cargo, specials, and batteries worth at least five are
        # high-value transport missions; lesser cargo can be ceded while trailing economically.
        transport_active = any(
            self._is_holding(enemy)
            and (
                self._class_id(enemy) == CARRIER
                or float(enemy[4]) >= 5.0 / 15.0
                or float(np.max(enemy[5:9])) > 1e-5
            )
            for enemy in enemies
        )
        self._transport_inactive_ticks = (
            0 if transport_active else self._transport_inactive_ticks + 1
        )
        if self._hunter_cooldown > 0:
            self._hunter_cooldown -= 1

        # Hunter -> Collector is the observable confirmation that death/respawn occurred.
        if self._respec_local is not None:
            name = controlled[self._respec_local]
            if self._class_id(states[row_for[name]]) == COLLECTOR:
                self.respec_completions += 1
                self._respec_local = None
                self._hunter_assignee = None
                self._hunter_committed = False
                self._hunter_cooldown = self.respec_cooldown_ticks

        if self._hunter_cooldown > 0 and self._respec_local is None:
            # Mask V7's phase trigger until the economic payback window has elapsed.
            self._hunter_assignee = None
            self._hunter_committed = False
            for local, name in enumerate(controlled):
                if self._class_id(states[row_for[name]]) == COLLECTOR:
                    self._active_roles[local] = "collector"

        field_mask = (
            (graphic[..., STORAGE_ALLY] < 0.5)
            & (graphic[..., STORAGE_ENEMY] < 0.5)
        )
        field_battery = float(np.sum(graphic[..., BATTERY][field_mask]) * 15.0)
        trailing = float(team_state[1]) - float(team_state[0]) >= self.respec_score_gap
        enemy_spawn = self._nearest_pixel(
            graphic[..., SPAWN_ENEMY] > 0.5, (graphic.shape[0] // 2, graphic.shape[1] // 2)
        )
        enemy_hunter_exists = any(
            self._class_id(enemy) == HUNTER
            and (enemy_spawn is None or math.dist(
                self._to_pixel(enemy[:2], graphic.shape[:2]), enemy_spawn
            ) > 3.5)
            for enemy in enemies
        )
        own_hunter_exists = any(
            self._class_id(states[row_for[name]]) == HUNTER for name in controlled
        )
        self.respec_diagnostics["max_inactive_ticks"] = max(
            self.respec_diagnostics["max_inactive_ticks"], self._transport_inactive_ticks
        )
        self.respec_diagnostics["trailing_ticks"] += int(trailing)
        self.respec_diagnostics["both_hunters_ticks"] += int(
            own_hunter_exists and enemy_hunter_exists
        )
        gates_ready = (
            self._transport_inactive_ticks >= self.respec_inactivity_ticks
            and trailing
            and float(team_state[2]) >= self.respec_min_time_left
            and field_battery >= self.respec_min_field_battery
            and enemy_hunter_exists
            and own_hunter_exists
        )
        self.respec_diagnostics["all_gates_ticks"] += int(gates_ready)
        if (
            self._respec_local is None
            and self._hunter_cooldown == 0
            and gates_ready
        ):
            for local, name in enumerate(controlled):
                if self._class_id(states[row_for[name]]) == HUNTER:
                    self._respec_local = local
                    self.respec_attempts += 1
                    break

        self.role_assignments = {
            name: ("respec" if local == self._respec_local else self._active_roles[local])
            for local, name in enumerate(controlled)
        }

    def _choose_target(
        self, role, state, states, graphic, reservations, local_index, team_state
    ):
        """The respec Hunter targets the nearest enemy Hunter outside its base; with none, abort."""
        if local_index == self._respec_local and self._class_id(state) == HUNTER:
            pos = self._to_pixel(state[:2], graphic.shape[:2])
            enemy_spawn = self._nearest_pixel(graphic[..., SPAWN_ENEMY] > 0.5, pos)
            enemy_hunters = [
                enemy for enemy in states
                if enemy[2] < 0 and self._class_id(enemy) == HUNTER
                and (enemy_spawn is None or math.dist(
                    self._to_pixel(enemy[:2], graphic.shape[:2]), enemy_spawn
                ) > 3.5)
            ]
            if enemy_hunters:
                enemy = min(
                    enemy_hunters,
                    key=lambda other: np.linalg.norm(other[:2] - state[:2]),
                )
                return self._to_pixel(enemy[:2], graphic.shape[:2]), "hunt"
            # Never substitute an economic enemy merely to force our own reset.
            self._respec_local = None
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )
