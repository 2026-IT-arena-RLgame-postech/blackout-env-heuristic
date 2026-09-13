"""Deliberately distinct, state-switching heuristic teachers.

These policies are *not* intended to replace :class:`StrategicHeuristicV4` as the
single strongest baseline.  They expose coherent alternate game plans for behaviour
cloning and offline RL: a learner can observe a strategy change caused by public game
state, rather than only infinitesimal parameter noise around one policy.
"""

from __future__ import annotations

import math

import numpy as np

from .dynamic_roles import StrategicHeuristicV7
from .spread_deposit import StrategicHeuristicV4
from .strategic import (
    BATTERY,
    CARRIER,
    HUNTER,
    SPAWN_ENEMY,
    STORAGE_ALLY,
    STORAGE_ENEMY,
    WALL,
)


class _ModeSwitchMixin:
    """Small hysteretic state machine shared by strategy-switching teachers.

    Public observations fluctuate around score/absorption thresholds.  Requiring a
    condition to persist avoids a policy that alternates its target every decision tick.
    ``strategy_transitions`` is intentionally public dataset provenance.
    """

    def _init_mode_switch(self, initial_mode: str, confirm_ticks: int) -> None:
        self.current_mode = initial_mode
        self.mode_confirm_ticks = max(1, int(confirm_ticks))
        self._pending_mode: str | None = None
        self._pending_ticks = 0
        self.strategy_transitions: list[tuple[int, str]] = []

    def _reset_mode_switch(self, initial_mode: str) -> None:
        self.current_mode = initial_mode
        self._pending_mode = None
        self._pending_ticks = 0
        self.strategy_transitions.clear()

    def _set_mode_candidate(self, candidate: str) -> None:
        if candidate == self.current_mode:
            self._pending_mode = None
            self._pending_ticks = 0
            return
        if candidate != self._pending_mode:
            self._pending_mode = candidate
            self._pending_ticks = 1
            return
        self._pending_ticks += 1
        if self._pending_ticks >= self.mode_confirm_ticks:
            self.current_mode = candidate
            self.strategy_transitions.append((self._tick, candidate))
            self._pending_mode = None
            self._pending_ticks = 0


class StrategicHeuristicV10(_ModeSwitchMixin, StrategicHeuristicV7):
    """Phase director: opening economy -> pressure raid -> protected closeout.

    The policy delays Hunter creation during a resource-rich opening, turns its Hunter
    into an active raider during a vulnerable enemy absorption window, then asks the
    Hunter to protect the lead rather than chase empty enemy Collectors.  Specialist
    classes cannot be reverted without death, so the phase only controls *new* role
    commitment; existing Hunters receive a mode-specific objective.
    """

    def __init__(
        self,
        *,
        mode_confirm_ticks: int = 18,
        pressure_absorption: float = 0.40,
        closeout_lead: float = 0.09,
        closeout_time: float = 0.52,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pressure_absorption = float(np.clip(pressure_absorption, 0.05, 0.95))
        self.closeout_lead = max(0.0, float(closeout_lead))
        self.closeout_time = float(np.clip(closeout_time, 0.05, 0.95))
        self._configured_hunter_quota = self.hunter_quota
        self._init_mode_switch("opening_economy", mode_confirm_ticks)

    def reset(self) -> None:
        super().reset()
        self.hunter_quota = self._configured_hunter_quota
        self._reset_mode_switch("opening_economy")

    def _desired_mode(self, sample) -> str:
        graphic = sample["graphic"]
        score, enemy_score, time_left, absorption = map(float, sample["team_state"][:4])
        lead = score - enemy_score
        protected = self._cached_protected_enemy_storage_mask(graphic)
        external_enemy_value = float(np.sum(graphic[..., BATTERY][
            (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected
        ]) * 15.0)
        if lead >= self.closeout_lead and time_left <= self.closeout_time:
            return "closeout_defend"
        # Stealing immediately before absorption has both a collection and denial value.
        if (absorption <= self.pressure_absorption and external_enemy_value >= 3.0) or (
            enemy_score - score >= 0.10 and external_enemy_value >= 1.0
        ):
            return "pressure_raid"
        return "opening_economy"

    def _update_roles(self, obs, sample) -> None:
        self._set_mode_candidate(self._desired_mode(sample))
        states = sample["agent_states"]
        own_hunter_exists = any(
            state[2] > 0 and self._class_id(state) == HUNTER for state in states
        )
        # V7's commitment is irreversible until death.  The delayed opening is therefore
        # meaningful only while no Hunter has been transformed yet.
        self.hunter_quota = (
            self._configured_hunter_quota
            if own_hunter_exists or self.current_mode != "opening_economy"
            else 0
        )
        super()._update_roles(obs, sample)
        self.role_assignments = {
            name: f"{self.current_mode}:{role}"
            for name, role in self.role_assignments.items()
        }

    def _choose_target(self, role, state, states, graphic, reservations, local_index, team_state):
        if self._class_id(state) == HUNTER:
            pos = self._to_pixel(state[:2], graphic.shape[:2])
            enemies = [enemy for enemy in states if enemy[2] < 0]
            enemy_spawn = self._nearest_pixel(graphic[..., SPAWN_ENEMY] > 0.5, pos)
            enemies = [
                enemy for enemy in enemies
                if enemy_spawn is None
                or math.dist(self._to_pixel(enemy[:2], graphic.shape[:2]), enemy_spawn) > 3.5
            ]
            if self.current_mode == "pressure_raid":
                # A visible Carrier/cargo target is worth pursuing even when an empty enemy is
                # physically closer.  In its absence scout external enemy storage, which makes
                # this policy's Hunter trajectories visibly different from V4's patrol loop.
                valuable = [
                    enemy for enemy in enemies
                    if self._is_holding(enemy) or self._class_id(enemy) == CARRIER
                ]
                if valuable:
                    enemy = min(valuable, key=lambda other: np.linalg.norm(other[:2] - state[:2]))
                    return self._to_pixel(enemy[:2], graphic.shape[:2]), "hunt"
                protected = self._cached_protected_enemy_storage_mask(graphic)
                return self._patrol_component_center(
                    (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected, local_index, period=90
                ), "patrol_raid"
            if self.current_mode == "closeout_defend":
                # Do not donate a leading Hunter to a pointless cross-map chase.  Intercept only
                # cargo likely to matter; otherwise hold a rotating home-storage patrol.
                cargo = [enemy for enemy in enemies if self._is_holding(enemy)]
                if cargo:
                    enemy = min(cargo, key=lambda other: np.linalg.norm(other[:2] - state[:2]))
                    return self._to_pixel(enemy[:2], graphic.shape[:2]), "hunt"
                return self._patrol_component_center(
                    graphic[..., STORAGE_ALLY] > 0.5, local_index, period=100
                ), "patrol_defend"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )


class StrategicHeuristicV11(_ModeSwitchMixin, StrategicHeuristicV4):
    """Raid-window specialist with a bounded two-unit external-storage strike team."""

    def __init__(
        self,
        *,
        mode_confirm_ticks: int = 12,
        raid_slots: int = 2,
        raid_absorption: float = 0.55,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raid_slots = max(1, min(3, int(raid_slots)))
        self.raid_absorption = float(np.clip(raid_absorption, 0.05, 0.95))
        self._init_mode_switch("harvest", mode_confirm_ticks)

    def reset(self) -> None:
        super().reset()
        self._reset_mode_switch("harvest")

    def _update_mode(self, sample) -> None:
        graphic = sample["graphic"]
        score, enemy_score, _, absorption = map(float, sample["team_state"][:4])
        protected = self._cached_protected_enemy_storage_mask(graphic)
        enemy_battery = float(np.sum(graphic[..., BATTERY][
            (graphic[..., STORAGE_ENEMY] > 0.5) & ~protected
        ]) * 15.0)
        candidate = "raid_window" if enemy_battery >= 2.0 and (
            absorption <= self.raid_absorption or enemy_score - score >= 0.08
        ) else "harvest"
        self._set_mode_candidate(candidate)

    def act(self, obs):
        if obs:
            sample = next(iter(obs.values()))
            # Keep the mode tick aligned with V2/V4's tick counter.  The first decision is
            # allowed to stay in harvest for confirmation; that is deliberate hysteresis.
            self._update_mode(sample)
        return super().act(obs)

    def _assign_economic_tasks(self, names, row_for, states, graphic, team_state, walkable):
        if self.current_mode != "raid_window":
            return super()._assign_economic_tasks(
                names, row_for, states, graphic, team_state, walkable
            )
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
        absorption_seconds = max(0.0, float(team_state[3]) * 20.0)
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
                # The deliberately steep amount term makes this a raid policy rather than a
                # subtly reweighted V4 worker assignment.
                value = amount * 8.0 - distance * 0.22
                pairs.append((value, name, task_index))
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
        # Non-strike-team Collectors retain the V4 fallback decision, so a one-dimensional
        # steal opportunity cannot freeze the rest of the economy.
        return result


class StrategicHeuristicV12(_ModeSwitchMixin, StrategicHeuristicV7):
    """Fortress/convoy policy that changes Hunter behaviour when protecting a lead.

    It produces late-game trajectories where one unit guards stored value and the team avoids
    converting a fresh Hunter unless catch-up pressure warrants it.  This is useful negative
    evidence for RL: chasing every visible enemy is not always the right action.
    """

    def __init__(
        self, *, mode_confirm_ticks: int = 20, fortress_lead: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.fortress_lead = max(0.0, float(fortress_lead))
        self._configured_hunter_quota = self.hunter_quota
        self._init_mode_switch("catch_up", mode_confirm_ticks)

    def reset(self) -> None:
        super().reset()
        self.hunter_quota = self._configured_hunter_quota
        self._reset_mode_switch("catch_up")

    def _update_roles(self, obs, sample) -> None:
        score, enemy_score, time_left, _ = map(float, sample["team_state"][:4])
        candidate = "fortress" if score - enemy_score >= self.fortress_lead and time_left <= 0.70 else "catch_up"
        self._set_mode_candidate(candidate)
        states = sample["agent_states"]
        hunter_exists = any(state[2] > 0 and self._class_id(state) == HUNTER for state in states)
        self.hunter_quota = (
            self._configured_hunter_quota
            if hunter_exists or self.current_mode == "catch_up"
            else 0
        )
        super()._update_roles(obs, sample)
        self.role_assignments = {
            name: f"{self.current_mode}:{role}"
            for name, role in self.role_assignments.items()
        }

    def _choose_target(self, role, state, states, graphic, reservations, local_index, team_state):
        if self.current_mode == "fortress" and self._class_id(state) == HUNTER:
            pos = self._to_pixel(state[:2], graphic.shape[:2])
            home_components = self._cached_components(graphic[..., STORAGE_ALLY] > 0.5)
            home_points = [point for component in home_components for point in component]
            threatening = []
            for enemy in states:
                if enemy[2] >= 0 or not self._is_holding(enemy):
                    continue
                enemy_pixel = self._to_pixel(enemy[:2], graphic.shape[:2])
                if home_points and min(math.dist(enemy_pixel, point) for point in home_points) <= 7.0:
                    threatening.append(enemy)
            if threatening:
                enemy = min(threatening, key=lambda other: np.linalg.norm(other[:2] - state[:2]))
                return self._to_pixel(enemy[:2], graphic.shape[:2]), "hunt"
            return self._patrol_component_center(
                graphic[..., STORAGE_ALLY] > 0.5, local_index, period=70
            ), "patrol_defend"
        return super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )
