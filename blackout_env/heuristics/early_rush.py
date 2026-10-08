"""Early rush: four Hunters sweep the enemy economy at the start, then the team turns economic.

Every policy in the pool keeps one composition all match: V4-V16 one or two Hunters, V17 three,
V18/V19 four, V21 as many as it can reach safely.  A unit can only leave the Hunter class by
dying, so none of them ever gives Hunters back.  The rush plays two compositions in sequence:

* **rush** (the first ``rush_seconds``, 35) -- ``rush_hunters`` (4) transform at once, no Carrier.
  They sweep: every enemy Collector or Carrier outside its base is a target, nearest by path first
  (value 1, plus the cargo, plus 3 for a Carrier, over the time to reach it), each Hunter
  preferring a target no teammate claimed.  With nothing in reach they stand on the enemy base
  exits, one Hunter per exit cell, to catch the units respawning there;
* **economy** -- from then on the team wants ``economy_hunters`` (1) Hunters and one Carrier.  No
  Collector transforms while the surplus Hunters are alive; each surplus Hunter hunts enemy Hunters
  to trade with them -- the only way back to Collector is to die -- and hunts cargo like V17 while
  no enemy Hunter is out.  The kept Hunter plays V17's hunt; the economy is V17's.

Against a pool whose field is empty within ~40 s this is a bet that killing the opponent's early
collectors outweighs collecting less oneself; the switch shows the learner an opponent whose
composition changes mid-match.
"""

from __future__ import annotations

import math

from .strategic import CARRIER, HUNTER
from .v17_planner import MATCH_SECONDS, StrategicHeuristicV17


class StrategicEarlyRush(StrategicHeuristicV17):
    """V17 planner with a four-Hunter opening sweep, then one Hunter and a Carrier."""

    def __init__(
        self,
        *,
        rush_seconds: float = 35.0,
        rush_hunters: int = 4,
        economy_hunters: int = 1,
        economy_carriers: int = 1,
        **kwargs,
    ):
        kwargs.setdefault("hunter_quota", rush_hunters)
        kwargs.setdefault("carrier_quota", 0)
        super().__init__(**kwargs)
        self.rush_seconds = float(rush_seconds)
        self.rush_hunters = int(rush_hunters)
        self.economy_hunters = int(economy_hunters)
        self.economy_carriers = int(economy_carriers)
        self.phase = "rush"

    def reset(self) -> None:
        super().reset()
        self.phase = "rush"

    def _update_roles(self, ctx) -> None:
        elapsed = MATCH_SECONDS - ctx["remaining"]
        self.phase = "rush" if elapsed < self.rush_seconds else "economy"
        if self.phase == "rush":
            self.hunter_quota, self.carrier_quota = self.rush_hunters, 0
        else:
            self.hunter_quota, self.carrier_quota = self.economy_hunters, self.economy_carriers
        super()._update_roles(ctx)

    def _hunter_plan(self, ctx, hunter, claimed):
        if self.phase == "rush":
            return self._sweep_plan(ctx, hunter, claimed)
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        if hunters.index(hunter.name) >= self.economy_hunters:
            rivals = [e for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
            reachable = [(self._dist(ctx, hunter.pos, e.pos), e.pos) for e in rivals]
            reachable = [r for r in reachable if math.isfinite(r[0])]
            if reachable:
                return min(reachable)[1], "hunt"  # trade, respawn as a Collector
        return self._hunt_plan(ctx, hunter, claimed)

    def _sweep_plan(self, ctx, hunter, claimed):
        best = None
        speed = self._speed(ctx, HUNTER)
        for enemy in ctx["enemies"]:
            if enemy.in_base or enemy.cls == HUNTER:
                continue
            distance = self._dist(ctx, hunter.pos, enemy.pos)
            if not math.isfinite(distance):
                continue
            value = 1.0 + (enemy.amount if enemy.slot == 1 else 6.0 if enemy.slot >= 2 else 0.0)
            value += 3.0 if enemy.cls == CARRIER else 0.0
            score = value / (distance / speed + 1.0) * (0.5 if enemy.pos in claimed else 1.0)
            if best is None or score > best[0]:
                best = (score, enemy.pos)
        if best is not None:
            return best[1], "hunt"
        slots = self._camp_slots(ctx)
        if not slots:
            return self._hunt_plan(ctx, hunter, claimed)
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        return slots[hunters.index(hunter.name) % len(slots)], "patrol_defend"
