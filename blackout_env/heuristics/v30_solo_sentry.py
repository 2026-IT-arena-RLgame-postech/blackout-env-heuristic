"""V29 whose lone Hunter guards the Hunter sanctuary (Q11, docs/policy_v2_design_ko.md §44).

Q11 recorded the §40 search champion (P17 acting + o4fmean search from 5 s, rollout v28d256) against V29, V28, V25
and P17 next to plain P17 on the same seeds.  The search's whole gain (+11 to +26 per opponent) accrues between 10
and 20 s: it deposits more and, against the opponents that stage a reserve (V29, P17), it cuts the enemy's Hunter
transforms in 10-15 s by a quarter to a half -- it wins the second race after the opening trades more often.  Its
deviations from P17, valued against the true opponent (counterfactual, ``examples/q11_why.py cf``), add up to the
measured gain and are spread over every class; per deviation the largest are Carriers and Hunters at 10-15 s
(+10.9 against V29), without one dominant direction.

In V22-V29 the sanctuary sentry (``StrategicHeuristicV22Teach._hunter_goals``: the Hunter nearest a post on the
enemy side of the sanctuary strikes enemy Collectors within ``sentry_radius`` of it or holds the post) is posted only
with ``sentry_min_hunters`` = 2 Hunters.  After the opening trades V29 is usually left with a single Hunter, which then
plays V17's camper (the enemy base exit) and leaves the sanctuary to the enemy's reserve -- the second race the search
contests.  ``solo_sentry`` (default off = V29 exactly) posts the sentry from the first Hunter on, i.e.
``sentry_min_hunters=1``; with two or more Hunters nothing changes.  Observation-only, as the sentry is.
"""

from __future__ import annotations

from .v29_reserve_join import StrategicHeuristicV29


class StrategicHeuristicV30(StrategicHeuristicV29):
    """V29 + a single Hunter already keeps the sanctuary (``solo_sentry``)."""

    def __init__(self, *, solo_sentry: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.solo_sentry = bool(solo_sentry)
        if self.solo_sentry:
            self.sentry_min_hunters = min(self.sentry_min_hunters, 1)
