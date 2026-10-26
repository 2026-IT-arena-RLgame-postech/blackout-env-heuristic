"""V33: V32.1 retuned after the E4 analysis (reports/heur_v33), plus knobs for the special items.

E4 (PPO beta 0.01, the strongest net) beats V32.1 by +49 almost entirely in the Hunter war, not in the battery
economy: V32.1's trailing race walker flees a fresh enemy Hunter 0.2--0.3 tiles before the sanctuary
(``race_commit`` only commits a walker that arrives before contact) and the enemy Hunter then camps our exit.  A
negative ``commit_margin`` (V32.1 knob) fixes most of it.  The special items are valued by V17's planner with fixed
weights per second of effect (``special_weights`` = BuffSpeed 0.45, DebuffSpeed 0.8, BuffSize 0.12, DebuffSize
0.12); a DebuffSpeed (enemy Workers -90 %) also decides Hunter races, so the weights are knobs here:

* ``w_bspd`` / ``w_dspd`` / ``w_bsize`` / ``w_dsize``: the four weights (defaults = V17's).

Defaults = V32 exactly (the class adds no behaviour of its own with default knobs).
"""

from __future__ import annotations

from .v32_four_hunters import StrategicHeuristicV32

_W = (0.45, 0.8, 0.12, 0.12)


class StrategicHeuristicV33(StrategicHeuristicV32):
    """V32 + scalar knobs for the special-item weights."""

    def __init__(self, *, w_bspd: float = _W[0], w_dspd: float = _W[1], w_bsize: float = _W[2],
                 w_dsize: float = _W[3], **kwargs):
        kwargs.setdefault("special_weights", (float(w_bspd), float(w_dspd), float(w_bsize), float(w_dsize)))
        super().__init__(**kwargs)
