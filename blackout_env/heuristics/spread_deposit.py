"""V4 (``strategic_v4``, parent V3): spread simultaneous deposits across storage tiles.

V4 is the reference heuristic: exported as ``RecommendedStrategicHeuristic``, it is the
standard evaluation opponent (periodic_eval, offline_pretrain's eval, the default on-policy
opponent) and, with its near variants (``v4_family.V4PolicyFamily``), 19% of the default
data mixture.  Its only change over V3: carriers heading for the same storage component are
sent to different, nearest free tiles instead of all to the component centre, which removes
the queueing/bumping at the entrance.  V5, V6, V7 and V11 branch from here.
"""

from __future__ import annotations

import math

import numpy as np

from .safe_storage import StrategicHeuristicV3
from .strategic import STORAGE_ALLY


class StrategicHeuristicV4(StrategicHeuristicV3):
    """V3 with per-tick storage-slot reservations to reduce team congestion."""

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
        """V3's choice, but a "deposit" target becomes the nearest unreserved tile of that store.

        The chosen tile is added to the tick's shared ``reservations`` so the next carrier
        heading to the same component takes a different tile.
        """
        target, kind = super()._choose_target(
            role, state, states, graphic, reservations, local_index, team_state
        )
        if kind != "deposit" or target is None:
            return target, kind

        component = next(
            (c for c in self._cached_components(graphic[..., STORAGE_ALLY] > 0.5) if target in c),
            None,
        )
        if component:
            origin = self._to_pixel(state[:2], graphic.shape[:2])
            free = [point for point in component if point not in reservations]
            choices = free or component
            # Enter through the closest free storage tile.  This both shortens the final path
            # and stops several carriers from issuing movement into one occupied centre cell.
            target = min(choices, key=lambda point: math.dist(point, origin))
        reservations.add(target)
        return target, kind
