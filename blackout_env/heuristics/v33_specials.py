"""V33: V32.1 retuned after the E4 analysis (reports/heur_v33), plus knobs for the special items.

E4 (PPO beta 0.01, the strongest net) beats V32.1 by +49 almost entirely in the Hunter war, not in the battery
economy: V32.1's trailing race walker flees a fresh enemy Hunter 0.2--0.3 tiles before the sanctuary
(``race_commit`` only commits a walker that arrives before contact) and the enemy Hunter then camps our exit.  A
negative ``commit_margin`` (V32.1 knob) fixes most of it.  The special items are valued by V17's planner with fixed
weights per second of effect (``special_weights`` = BuffSpeed 0.45, DebuffSpeed 0.8, BuffSize 0.12, DebuffSize
0.12); a DebuffSpeed (enemy Workers -90 %) also decides Hunter races, so the weights are knobs here:

* ``w_bspd`` / ``w_dspd`` / ``w_bsize`` / ``w_dsize``: the four weights (defaults = V17's).

Evasion when the enemy dominates the Hunter war: a team that lost it feeds ~40 Collectors in a minute (V4's threat
repulsion starts at ``threat_radius`` = 1.3 tiles, too late against a 6 tiles/s Hunter).  When the enemy has at least
``dom_margin`` more Hunters out of base than we have (read every tick from the observation):

* ``dom_threat_radius`` (-1 = off): the repulsion radius used instead of ``threat_radius``;
* ``dom_danger_hunter_radius`` (-1 = off): the planner's pickup-danger radius instead of ``danger_hunter_radius``.

Defaults = V32 exactly (the class adds no behaviour of its own with default knobs).
"""

from __future__ import annotations

from .strategic import HUNTER
from .v32_four_hunters import StrategicHeuristicV32

_W = (0.45, 0.8, 0.12, 0.12)


class StrategicHeuristicV33(StrategicHeuristicV32):
    """V32 + scalar knobs for the special-item weights."""

    def __init__(self, *, w_bspd: float = _W[0], w_dspd: float = _W[1], w_bsize: float = _W[2],
                 w_dsize: float = _W[3], dom_threat_radius: float = -1.0, dom_danger_hunter_radius: float = -1.0,
                 dom_margin: int = 1, **kwargs):
        kwargs.setdefault("special_weights", (float(w_bspd), float(w_dspd), float(w_bsize), float(w_dsize)))
        super().__init__(**kwargs)
        self.dom_threat_radius = float(dom_threat_radius)
        self.dom_danger_hunter_radius = float(dom_danger_hunter_radius)
        self.dom_margin = int(dom_margin)
        self._base_threat_radius = self.threat_radius
        self._base_danger_hunter_radius = self.danger_hunter_radius
        self.dominated = False  # for inspection: the enemy dominated the Hunter war this tick

    def _context(self, obs):
        ctx = super()._context(obs)
        if self.dom_threat_radius > 0 or self.dom_danger_hunter_radius > 0:
            theirs = sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"])
            ours = sum(u.cls == HUNTER for u in ctx["own"])
            self.dominated = theirs - ours >= self.dom_margin
            self.threat_radius = (self.dom_threat_radius if self.dominated and self.dom_threat_radius > 0
                                  else self._base_threat_radius)
            self.danger_hunter_radius = (self.dom_danger_hunter_radius
                                         if self.dominated and self.dom_danger_hunter_radius > 0
                                         else self._base_danger_hunter_radius)
        return ctx
