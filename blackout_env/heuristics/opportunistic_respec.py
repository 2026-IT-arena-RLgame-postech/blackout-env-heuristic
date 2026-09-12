"""Ninth-generation heuristic: an active strategic-death variation of V8."""

from __future__ import annotations

from .lifecycle_roles import StrategicHeuristicV8


class StrategicHeuristicV9(StrategicHeuristicV8):
    """V8 with a shorter payback horizon for collecting deliberate-respec trajectories."""

    def __init__(
        self,
        *,
        respec_inactivity_ticks: int = 30,
        respec_score_gap: float = 0.03,
        respec_min_field_battery: float = 5.0,
        respec_min_time_left: float = 0.12,
        respec_cooldown_ticks: int = 400,
        **kwargs,
    ):
        super().__init__(
            respec_inactivity_ticks=respec_inactivity_ticks,
            respec_score_gap=respec_score_gap,
            respec_min_field_battery=respec_min_field_battery,
            respec_min_time_left=respec_min_time_left,
            respec_cooldown_ticks=respec_cooldown_ticks,
            **kwargs,
        )
