"""Twenty-third-generation heuristic: V22 that walks the opening race on the exact any-angle field.

The match is decided by the race to the central Hunter sanctuary, and the race by single decision
ticks (0.04 s): both sides' walkers arrive at ~tick 99-100, a tie lets both transform, and one tick
late loses the walker and usually the match (P10 analysis, docs/experiments_index_ko.md).  V22
walks it on V21's controller: the heading is a one-step Theta* over the 8-neighbour (octile)
field, with ``CLEARANCE_MARGIN`` from walls and a region approach inset ``REGION_INSET`` -- in
7/24 simulator openings it arrived 1-3 ticks after the ideal.

V23 changes nothing but how the race is walked: while ``_racing`` (no own Hunter yet, or the
first ``opening_seconds``) the goal clusters, their ranking of walkers and the controller use the
any-angle field (``AnyAngleFieldMixin``, as ``strategic_v21_anyangle_ed`` but only for goal
clusters, ``field_ranking=False``), with ``opening_margin`` wall clearance and ``opening_inset``
region inset.  After the race it is V22 exactly.
"""

from __future__ import annotations

from .any_angle import CLEARANCE_MARGIN, AnyAngleFieldMixin
from .v22_teacher import REGION_INSET, StrategicHeuristicV22

OPENING_MARGIN = 0.0  # wall clearance on the race walk (tiles, beyond the unit radius)
OPENING_INSET = 0.01  # region inset on the race walk (tiles)


class StrategicHeuristicV23(AnyAngleFieldMixin, StrategicHeuristicV22):
    """V22 whose opening race is walked on the any-angle field with a tight clearance."""

    def __init__(self, *, opening_field: bool = True, opening_margin: float = OPENING_MARGIN,
                 opening_inset: float = OPENING_INSET, clearance_margin: float = CLEARANCE_MARGIN, **kwargs):
        self._race_now = False
        self.opening_field = bool(opening_field)
        self.opening_margin = float(opening_margin)
        self.opening_inset = float(opening_inset)
        super().__init__(field_ranking=False, clearance_margin=clearance_margin, **kwargs)

    def _opening(self) -> bool:
        return self.opening_field and self._race_now

    # AnyAngleFieldMixin reads field_goals; AnyAngleMixin and V22 read clearance_margin / region_inset.
    @property
    def field_goals(self) -> bool:
        return self._opening()

    @field_goals.setter
    def field_goals(self, value: bool) -> None:  # the mixin's __init__ sets it; the race decides
        pass

    @property
    def clearance_margin(self) -> float:
        return self.opening_margin if self._opening() else self._clearance_margin

    @clearance_margin.setter
    def clearance_margin(self, value: float) -> None:
        self._clearance_margin = float(value)

    @property
    def region_inset(self) -> float:
        return self.opening_inset if self._opening() else REGION_INSET

    def _context(self, obs) -> dict:
        ctx = super()._context(obs)
        self._race_now = self._racing(ctx)
        return ctx
