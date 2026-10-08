"""V30 whose Carrier evades less (Q12, docs/policy_v2_design_ko.md §46).

Q12 ran the §40 blind search (o4fmean, rollout v28d256) on top of V30 (``examples/q11_why.py play --base v30``)
and valued its deviations from V30 against the true opponent (``cf --base v30``).  The one unit class whose
deviations are worth something against every opponent is the Carrier (per deviation +6.8 vs V29, +7.9 vs V25; Q11
on top of P17 +3.8), and its direction is the same against all four opponents: relative to V30's move it runs less
away from the nearest enemy Hunter (``d_flee`` -0.2 tiles/decision) and less away from the nearest enemy
Collector (``d_prey`` +0.2) -- V30's Carrier spends time fleeing.

V4's threat repulsion (``TeachLabeller._evasion`` / ``AnyAngleMixin._evasion_vector``) steers a Carrier away from
*every* enemy within ``threat_radius`` (a Collector beats a Carrier, two Carriers both die, a Hunter beats all),
while a cargo-holding Collector only flees Hunters.  The Carrier is the fastest unit (6 tiles/s, Collector 4), so an
enemy Collector can only catch it head-on.  Two knobs (default 1 = V30 exactly) shrink the repulsion radius the
Carrier uses -- enemies farther than ``scale * threat_radius`` no longer repel it (those inside repel as before):

* ``carrier_col_scale``: for enemy Collectors and Carriers (0 = never flee them);
* ``carrier_hunter_scale``: for enemy Hunters.

Other units, and the evasion switches of V22 (committed walkers) and V24 (base exit / dash), are unchanged.
Observation-only, as the repulsion is.
"""

from __future__ import annotations

import numpy as np

from .strategic import CARRIER, HUNTER
from .v30_solo_sentry import StrategicHeuristicV30


class StrategicHeuristicV31(StrategicHeuristicV30):
    """V30 + a Carrier that flees only nearer enemies (``carrier_col_scale`` / ``carrier_hunter_scale``)."""

    def __init__(self, *, carrier_col_scale: float = 1.0, carrier_hunter_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.carrier_col_scale = float(carrier_col_scale)
        self.carrier_hunter_scale = float(carrier_hunter_scale)

    def _carrier_threats(self, state, states):
        """``states`` without the enemies too far to repel this Carrier (unchanged for other units / defaults)."""
        if (self.carrier_col_scale >= 1.0 and self.carrier_hunter_scale >= 1.0) or self._class_id(state) != CARRIER:
            return states
        keep = np.ones(len(states), dtype=bool)
        radius = self.threat_radius
        for i, row in enumerate(states):
            if row[2] >= 0:
                continue
            scale = self.carrier_hunter_scale if self._class_id(row) == HUNTER else self.carrier_col_scale
            if scale >= 1.0:
                continue
            away = state[:2] - row[:2]
            if float(np.sqrt(away.dot(away))) >= scale * radius:
                keep[i] = False
        return states if keep.all() else states[keep]

    def _evasion(self, state, states, delta):
        return super()._evasion(state, self._carrier_threats(state, states), delta)

    def _evasion_vector(self, state, states, delta):
        return super()._evasion_vector(state, self._carrier_threats(state, states), delta)
