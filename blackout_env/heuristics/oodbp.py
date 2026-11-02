"""OOD-BP personalities that need more than V34 knobs (blackout_env/train/distill/ood_bp.py).

``strategic_oodbp`` = V34 (so B6 + its descriptor knobs) plus a *Hunter plan*: the number of Hunters the team wants
as a step function of match time, ``hunter_plan="0:2,25:0,200:2"`` = two Hunters (the opening race as B6 plays it),
none wanted from 25 s (no Collector is sent to the sanctuary, the quota-refill rules are off), two again from 200 s.
While the target is 0 the Hunters that are still alive ``hide`` (``hide=home``: walk into our own base, which enemies
cannot enter; ``hide=stay``: stand still) instead of raiding -- an economy-only mid-game that saves its Hunters for
a late burst (or never bursts: ``hunter_plan="0:2,25:0"``).
"""

from __future__ import annotations

import numpy as np

from .any_angle import UNIT_RADIUS, AnyAngleLabel, pixel_position
from .strategic import HUNTER
from .v17_planner import MATCH_SECONDS
from .v34_bp import StrategicHeuristicV34


def parse_plan(text: str) -> list[tuple[float, int]]:
    plan = []
    for item in filter(None, str(text).replace("/", ",").split(",")):
        t, _, n = item.partition(":")
        plan.append((float(t), int(n)))
    return sorted(plan)


class StrategicOODBP(StrategicHeuristicV34):
    def __init__(self, *, hunter_plan: str = "", hide: str = "", **kwargs):
        super().__init__(**kwargs)
        self.plan = parse_plan(hunter_plan)
        self.hide = str(hide)

    def _plan_target(self, ctx):
        """Hunters wanted now, or None when no plan entry applies yet."""
        now = MATCH_SECONDS - ctx["remaining"]
        want = None
        for t, n in self.plan:
            if now >= t:
                want = n
        return want

    def _composition_targets(self, graphic, states, row_for):
        hunters, carriers = super()._composition_targets(graphic, states, row_for)
        want = self._plan_target(self._teach_ctx) if self.plan else None
        if want is not None:
            hunters = min(hunters, want)
        return hunters, carriers

    def teach(self, obs):
        labels = super().teach(obs)
        if labels and self.plan and self.hide:
            ctx = self._teach_ctx
            if self._plan_target(ctx) == 0:
                self._hide_hunters(labels, ctx)
        return labels

    def _hide_hunters(self, labels, ctx):
        shape, states, walk = ctx["shape"], ctx["states"], self._teach_walkable
        base = self._cached_base_mask(ctx["ally_spawn"], shape)
        tiles = tuple(sorted((int(y), int(x)) for y, x in np.argwhere(base & walk)))
        for u in ctx["own"]:
            if u.cls != HUNTER or u.name not in labels:
                continue
            if self.hide == "stay" or not tiles or base[u.pos]:
                heading = np.zeros(2, dtype=np.float32)
            else:
                here = pixel_position(states[u.row], shape)
                heading = self._path_heading(walk, tiles, here, UNIT_RADIUS[HUNTER])
                if heading is None:
                    continue
            lab = labels[u.name]
            labels[u.name] = AnyAngleLabel(lab.mask, True, lab.kind, lab.target, lab.goals, False, heading)
            self._count("hide")
