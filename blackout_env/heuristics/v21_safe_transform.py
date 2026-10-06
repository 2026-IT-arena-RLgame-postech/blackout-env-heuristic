"""Twenty-first-generation heuristic: V20 that only sends Collectors to become Hunters safely.

V17-teach's composition labels fill the team up to 3 Hunters whatever stands between a Collector
and the central sanctuary.  V19 keeps four Hunters out -- one on our exit, one on the sanctuary,
two sweeping the ground around it -- so against V19 the V20 teacher labelled 2,491 trips to the
sanctuary in 24 matches, 2% of them ended as a Hunter and 17% dead on the way; 71% of its 35
deaths per match came on those trips, and it averaged half a Hunter alive.  Every such trip is a
Collector taken off the economy for nothing.

V21 changes one thing: a Collector is a Hunter candidate only if a route to the sanctuary exists
that stays ``DANGER_TILES`` away from every enemy Hunter outside its base (our own base never
counts as danger), and it walks that route.  Without one it stays a Collector and works the
economy (V20's camp counterplay and threat evasion still apply).  The Carrier sanctuary sits
inside our base and is unchanged.
"""

from __future__ import annotations

import numpy as np

from .strategic import HUNTER, SITE_HUNTER
from .teach import Goal, TeachLabel
from .v20_anticamp import StrategicHeuristicV20Teach


class StrategicHeuristicV21Teach(StrategicHeuristicV20Teach):
    """V20-teach whose Hunter transformations take only routes clear of every enemy Hunter."""

    def _hunter_free(self, ctx, walkable: np.ndarray) -> np.ndarray:
        """``walkable`` minus the cells near any enemy Hunter outside its base (our base stays).

        Memoised per tick (ctx and walkable held by reference): the layers ask it several times."""
        memo = self._tick_memo()
        key = ("hunter_free", id(ctx), id(walkable))
        hit = memo.get(key)
        if hit is not None and hit[0] is ctx and hit[1] is walkable:
            return hit[2]
        hunters = [e.pos for e in ctx["enemies"] if e.cls == HUNTER and not e.in_base]
        if not hunters:
            free = walkable
        else:
            base = self._cached_base_mask(ctx["ally_spawn"], ctx["shape"])
            free = walkable & ~(self._near(ctx["shape"], hunters, self.danger_tiles) & ~base)
        memo[key] = (ctx, walkable, free)
        return free

    def _composition_goals(self, controlled, row_for, states, graphic, walkable) -> dict[str, tuple[Goal, ...]]:
        # Candidates are ranked by path length on the Hunter-free map: no clear route, no candidate.
        return super()._composition_goals(controlled, row_for, states, graphic, self._hunter_free(self._teach_ctx, walkable))

    def _labels(self, ctx, goals: dict[str, tuple[Goal, ...]], move_walkable: np.ndarray) -> dict[str, TeachLabel]:
        labels = super()._labels(ctx, goals, move_walkable)
        free = self._hunter_free(ctx, move_walkable)
        if free is move_walkable:
            return labels
        graphic, states, shape = ctx["graphic"], ctx["states"], ctx["shape"]
        for unit in ctx["own"]:
            unit_goals = goals.get(unit.name, ())
            if any(kind == "transform" and graphic[tiles[0]][SITE_HUNTER] > 0.5 for tiles, kind in unit_goals):
                route = self._label(states[unit.row], states, unit_goals, free, shape)
                if route.valid:  # the clear route it was chosen for
                    labels[unit.name] = route
        return labels


class StrategicHeuristicV21(StrategicHeuristicV21Teach):
    """V21 as a player: follows its own labels (TeachLabeller.act), standing still where it has none."""
