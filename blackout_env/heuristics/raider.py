"""Raider: a steal-first economy on the V17 planner, with a Hunter escorting the thieves.

Every heuristic from V4 to V21 earns its points from the field and steals only on the side: V17's
planner values a stolen battery at ``steal_weight`` 1.8 times a field one and only when it can
arrive before absorption, and V13/V11 raid from V4's weaker economy.  None of them lets the
opponent do the collecting.  The raider does:

* a **raid squad** of ``raid_squad`` Collectors never takes field batteries.  They wait beside
  the enemy's exposed storages (V17's idle plan: the storage most likely to receive the next
  deposit) and take whatever lands there before it is absorbed.  Members are kept until they die,
  transform or pick something up, so the squad does not reshuffle every tick;
* the rest of the team still collects, with stolen cargo worth ``steal_weight`` (3.5) and field
  batteries ``field_weight`` (0.6) of V17's value -- a free unit goes stealing whenever there is
  anything to steal;
* everything goes home to the protected storage: ``exposure_cost`` 2.0 (V17: 0.9), so the loot is
  not stolen straight back;
* ``escorts`` Hunters (1) stand at the enemy storage the squad works and kill anything that comes
  within ``escort_radius`` of it or of a squad member -- defenders, enemy Hunters guarding it
  (a trade), and the opponent's own cargo arriving to deposit.  The other Hunter hunts like V17.

An opponent that deposits only in its protected base storage starves the squad; that is the
matchup this policy exists to show (and the one V17-family planners with their exposure cost
already play).
"""

from __future__ import annotations

import math

from .strategic import CARRIER, COLLECTOR, HUNTER
from .v17_planner import StrategicHeuristicV17


class StrategicRaider(StrategicHeuristicV17):
    """V17 planner whose economy steals first; a Hunter escorts the raid squad."""

    def __init__(
        self,
        *,
        hunter_quota: int = 2,
        steal_weight: float = 3.5,
        exposure_cost: float = 2.0,
        field_weight: float = 0.6,
        raid_squad: int = 2,
        escorts: int = 1,
        escort_radius: float = 5.0,
        **kwargs,
    ):
        super().__init__(hunter_quota=hunter_quota, steal_weight=steal_weight, exposure_cost=exposure_cost, **kwargs)
        self.field_weight = float(field_weight)
        self.raid_squad = int(raid_squad)
        self.escorts = int(escorts)
        self.escort_radius = float(escort_radius)
        self.squad: list[str] = []  # for inspection: the current raid squad

    def reset(self) -> None:
        super().reset()
        self.squad = []

    # ------------------------------------------------------------------ economy

    def _economy_pairs(self, ctx, units):
        tasks, pairs, pickup = super()._economy_pairs(ctx, units)
        squad = set(self._update_squad(ctx, units))
        kept = []
        for score, name, task_index in pairs:
            if tasks[task_index][1] == "battery":
                if name in squad:
                    continue  # the squad leaves the field to the opponent
                score *= self.field_weight
            kept.append((score, name, task_index))
        return tasks, kept, pickup

    def _update_squad(self, ctx, units) -> list[str]:
        """Keep empty-handed economic Collectors in the squad; refill with the ones nearest an
        exposed enemy storage."""
        eligible = {u.name: u for u in units if u.cls == COLLECTOR and not u.holding}
        squad = [name for name in self.squad if name in eligible][: self.raid_squad]
        targets = [s for s in ctx["enemy_stores"] if not s.protected]
        if len(squad) < self.raid_squad and targets:
            spare = sorted(
                (name for name in eligible if name not in squad),
                key=lambda name: min(self._dist(ctx, eligible[name].pos, s.center) for s in targets))
            squad += spare[: self.raid_squad - len(squad)]
        self.squad = squad
        return squad

    # ------------------------------------------------------------------ Hunters

    def _hunter_plan(self, ctx, hunter, claimed):
        hunters = sorted(u.name for u in ctx["own"] if u.cls == HUNTER)
        if hunters.index(hunter.name) < self.escorts:
            plan = self._escort_plan(ctx, hunter, claimed)
            if plan is not None:
                return plan
        return self._hunt_plan(ctx, hunter, claimed)

    def _escort_plan(self, ctx, hunter, claimed):
        """Stand at the enemy storage the squad works; strike anything near it or near a thief."""
        exposed = [s for s in ctx["enemy_stores"] if not s.protected]
        if not exposed:
            return None
        thieves = [u for u in ctx["own"] if u.name in self.squad]
        if any(s.battery + s.specials for s in exposed):
            post = max(exposed, key=lambda s: s.battery + 6.0 * s.specials).center
        elif thieves:
            post = min(exposed, key=lambda s: min(math.dist(s.center, t.pos) for t in thieves)).center
        else:
            post = min(exposed, key=lambda s: self._dist(ctx, hunter.pos, s.center)).center
        guarded = [post] + [t.pos for t in thieves]
        best = None
        for enemy in ctx["enemies"]:
            if enemy.in_base or min(math.dist(enemy.pos, p) for p in guarded) > self.escort_radius:
                continue
            distance = self._dist(ctx, hunter.pos, enemy.pos)
            if not math.isfinite(distance):
                continue
            value = 1.0 + (enemy.amount if enemy.slot == 1 else 6.0 if enemy.slot >= 2 else 0.0)
            value += 3.0 if enemy.cls in (CARRIER, HUNTER) else 0.0  # Hunters threaten the thieves
            score = value / (distance + 1.0) * (0.5 if enemy.pos in claimed else 1.0)
            if best is None or score > best[0]:
                best = (score, enemy.pos)
        if best is not None:
            return best[1], "hunt"
        return post, "patrol_defend"
