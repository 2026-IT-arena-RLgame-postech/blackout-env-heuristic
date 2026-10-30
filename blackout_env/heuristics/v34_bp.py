"""V34: V33 for the balance patch (max 2 living Hunters per team, spawn-side walls; docs: reports/bp).

Post-patch matches are decided in the first ~20 s by the Hunter war.  Both sides' two race walkers reach the
sanctuary on the same tick (~4 s) and trade Hunter for Hunter; the side whose *next* Collector enters first then owns
the sanctuary and kills ~10 Collectors.  So the count of Collectors committed near the sanctuary decides, and the
Carrier made at 0.4 s is one walker fewer (it can never become a Hunter).  New knobs:

* ``carrier_delay`` (s, 0 = off): no Carrier before this match time, unless ``carrier_on_monopoly`` and we hold a
  Hunter monopoly (a Hunter of ours out, none of theirs out of base) -- the would-be Carrier joins the Hunter war.
* ``reserve_count`` may be 3 (V27's staged reserve walkers).

Defaults = V33 exactly.
"""

from __future__ import annotations

from .strategic import HUNTER
from .v17_planner import MATCH_SECONDS
from .v33_specials import StrategicHeuristicV33


class StrategicHeuristicV34(StrategicHeuristicV33):
    def __init__(self, *, carrier_delay: float = 0.0, carrier_on_monopoly: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.carrier_delay = float(carrier_delay)
        self.carrier_on_monopoly = bool(carrier_on_monopoly)

    def _composition_targets(self, graphic, states, row_for):
        hunters, carriers = super()._composition_targets(graphic, states, row_for)
        if carriers and self.carrier_delay > 0:
            ctx = self._teach_ctx
            if MATCH_SECONDS - ctx["remaining"] < self.carrier_delay:
                ours = sum(u.cls == HUNTER for u in ctx["own"])
                theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
                if not (self.carrier_on_monopoly and ours > 0 and theirs == 0):
                    carriers = 0
        return hunters, carriers
