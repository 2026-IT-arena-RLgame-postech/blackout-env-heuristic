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

Hunter-for-Hunter trades decide who keeps the last Hunter (a Hunter monopoly is worth ~50 points: E4 turns 2:1 into
1:0 by trading its spare Hunter into our lone one).  Counting Hunters out of base each tick:

* ``h_evade_radius`` (tiles, -1 = off): while we have fewer Hunters than the enemy, our Hunters are repelled by enemy
  Hunters within this radius (V4's repulsion, applied to Hunters) instead of trading down;
* ``h_trade_up`` (off): while we have more Hunters than the enemy (and it has one out), the pair (our Hunter, enemy
  Hunter) closest together becomes a strike: our Hunter hunts that enemy Hunter.

Defaults = V32 exactly (the class adds no behaviour of its own with default knobs).
"""

from __future__ import annotations

import math

import numpy as np

from .strategic import CARRIER, COLLECTOR, HUNTER
from .teach import DIRECTIONS
from .v32_four_hunters import StrategicHeuristicV32

_W = (0.45, 0.8, 0.12, 0.12)


class StrategicHeuristicV33(StrategicHeuristicV32):
    """V32 + scalar knobs for the special-item weights."""

    def __init__(self, *, w_bspd: float = _W[0], w_dspd: float = _W[1], w_bsize: float = _W[2],
                 w_dsize: float = _W[3], dom_threat_radius: float = -1.0, dom_danger_hunter_radius: float = -1.0,
                 dom_margin: int = 1, h_evade_radius: float = -1.0, h_trade_up: bool = False, **kwargs):
        kwargs.setdefault("special_weights", (float(w_bspd), float(w_dspd), float(w_bsize), float(w_dsize)))
        super().__init__(**kwargs)
        self.dom_threat_radius = float(dom_threat_radius)
        self.dom_danger_hunter_radius = float(dom_danger_hunter_radius)
        self.dom_margin = int(dom_margin)
        self._base_threat_radius = self.threat_radius
        self._base_danger_hunter_radius = self.danger_hunter_radius
        self.dominated = False  # for inspection: the enemy dominated the Hunter war this tick
        self.h_evade_radius = float(h_evade_radius)
        self.h_trade_up = bool(h_trade_up)
        self._h_counts = (0, 0)  # (ours, theirs out of base) this tick
        self.trade_ups: list[str] = []  # for inspection

    def _context(self, obs):
        ctx = super()._context(obs)
        self._h_counts = (sum(u.cls == HUNTER for u in ctx["own"]),
                          sum(e.cls == HUNTER and not e.in_base for e in ctx["enemies"]))
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

    # ------------------------------------------------------------------ Hunter vs Hunter

    def _hunter_repulsion(self, state, states, delta):
        """Summed repulsion of an outnumbered Hunter from enemy Hunters within ``h_evade_radius`` (None if none)."""
        if self.h_evade_radius <= 0 or self._class_id(state) != HUNTER:
            return None
        ours, theirs = self._h_counts
        if ours >= theirs:
            return None
        radius = self.h_evade_radius / 12.0  # normalised units (threat_radius x 12 = tiles)
        total = np.asarray(delta, np.float32).copy()
        repelled = False
        for enemy in states[states[:, 2] < 0]:
            if self._class_id(enemy) != HUNTER:
                continue
            away = state[:2] - enemy[:2]
            dist = float(np.sqrt(away.dot(away)))
            if 1e-5 < dist < radius:
                total += (away / dist) * ((radius - dist) / radius) * 9.0
                repelled = True
        if not repelled or float(np.linalg.norm(total)) < 1e-6:
            return None
        return total

    def _evasion(self, state, states, delta):
        total = self._hunter_repulsion(state, states, delta)
        if total is not None:
            return int(np.argmax(DIRECTIONS @ total))
        # Every layer below returns None or defers; TeachLabeller._evasion answers None unless the unit
        # holds something, is a Carrier or (evade_empty_collectors) a Collector: skip the chain then.
        cls = self._class_id(state)
        if not (self._is_holding(state) or cls == CARRIER or (self.evade_empty_collectors and cls == COLLECTOR)):
            return None
        return super()._evasion(state, states, delta)

    def _evasion_vector(self, state, states, delta):
        total = self._hunter_repulsion(state, states, delta)
        if total is not None:
            return (total / float(np.linalg.norm(total))).astype(np.float32)
        return super()._evasion_vector(state, states, delta)

    def _hunter_goals(self, ctx, hunters, move_walkable):
        goals = super()._hunter_goals(ctx, hunters, move_walkable)
        self.trade_ups = []
        if not self.h_trade_up or not hunters:
            return goals
        ours, theirs = self._h_counts
        prey = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not prey or ours <= theirs:
            return goals
        gap, h, e = min((math.dist(h.pos, e.pos), h.name, e.pos) for h in hunters for e in prey)
        goals[h] = (((tuple(e),), "hunt"),)
        self.trade_ups.append(h)
        return goals
