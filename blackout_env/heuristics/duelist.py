"""Duelist: Hunters that hunt Hunters, trading one-for-one to keep the enemy's Hunter count at zero.

In the pool a Hunter-for-Hunter trade (both die) is a side effect: V17-V21 trade only with a
Hunter camping their spawn or exposed storage, V18/V19's exit guard only at its own exit, and
V4-V16 Hunters chase economy units.  So the learner has barely seen an opponent that *seeks* its
Hunters out.  The duelist's Hunters (``hunter_quota`` 2) do nothing else:

* every enemy Hunter outside its base is a target, nearest by path first, anywhere on the map.
  A trade sends their Hunter back to its spawn as a Collector -- it must walk to the sanctuary
  again -- and ours back to our spawn, where V17's role update re-transforms it;
* with no enemy Hunter out, each duelist escorts our most valuable load (a teammate with cargo,
  or the Carrier), standing ``escort_offset`` tiles off it on the enemy side of it, and kills any
  enemy that comes within ``escort_radius`` of it.  With nothing to escort it waits at the Hunter
  sanctuary, where enemy Hunters are born.

The economy, Carrier and deposits are V17's.
"""

from __future__ import annotations

import math

from .strategic import CARRIER, HUNTER, SITE_HUNTER
from .v17_planner import StrategicHeuristicV17


class StrategicDuelist(StrategicHeuristicV17):
    """V17 planner whose Hunters seek trades with enemy Hunters and otherwise escort our cargo."""

    def __init__(self, *, hunter_quota: int = 2, escort_radius: float = 4.0, escort_offset: float = 1.5, **kwargs):
        super().__init__(hunter_quota=hunter_quota, **kwargs)
        self.escort_radius = float(escort_radius)
        self.escort_offset = float(escort_offset)

    def _hunter_plan(self, ctx, hunter, claimed):
        rivals = []
        for enemy in ctx["enemies"]:
            if enemy.cls == HUNTER and not enemy.in_base:
                distance = self._dist(ctx, hunter.pos, enemy.pos)
                if math.isfinite(distance):
                    rivals.append((distance * (2.0 if enemy.pos in claimed else 1.0), enemy.pos))
        if rivals:
            return min(rivals)[1], "hunt"
        return self._escort_plan(ctx, hunter, claimed)

    def _escort_plan(self, ctx, hunter, claimed):
        loads = [u for u in ctx["own"] if u.holding or u.cls == CARRIER]
        if not loads:
            sanctuary = self._nearest_site_center(ctx["graphic"], SITE_HUNTER, hunter.pos)
            return sanctuary, "patrol_defend"
        ward = max(loads, key=lambda u: (u.amount if u.slot == 1 else 6.0 if u.slot >= 2 else 0.0)
                   + (3.0 if u.cls == CARRIER else 0.0) - 0.05 * math.dist(u.pos, hunter.pos))
        threats = [(self._dist(ctx, hunter.pos, e.pos), e.pos) for e in ctx["enemies"]
                   if not e.in_base and math.dist(e.pos, ward.pos) <= self.escort_radius]
        threats = [t for t in threats if math.isfinite(t[0])]
        if threats:
            return min(threats)[1], "hunt"
        spawn = ctx["enemy_spawn"]
        if spawn is None:
            return ward.pos, "patrol_defend"
        dy, dx = spawn[0] - ward.pos[0], spawn[1] - ward.pos[1]
        norm = math.hypot(dy, dx) or 1.0
        post = (round(ward.pos[0] + dy / norm * self.escort_offset), round(ward.pos[1] + dx / norm * self.escort_offset))
        post = (min(max(post[0], 0), ctx["shape"][0] - 1), min(max(post[1], 0), ctx["shape"][1] - 1))
        return (post if ctx["walkable"][post] else ward.pos), "patrol_defend"
