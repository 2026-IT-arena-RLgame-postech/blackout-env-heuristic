"""Hand-written policies useful as baselines and behaviour-cloning teachers.

Version -> module (class ``StrategicHeuristicV<N>``, policy_id ``strategic_v<N>``):

  V1       strategic.py               fixed roles, per-unit targets (StrategicHeuristic, alias ...V1)
  V2       advanced.py                team-wide greedy task assignment
  V3       safe_storage.py            threat/absorption-aware deposit choice
  V4       spread_deposit.py          deposit-tile reservation; = RecommendedStrategicHeuristic,
                                      the eval and on-policy reference opponent
  V4-near  v4_family.py               V4PolicyFamily (strategic_v4_near): V4 parameter variants
  V5       intercept.py               predictive Hunter intercepts
  V6       risk_path.py               enemy-threat cost in carrier A*
  V7       dynamic_roles.py           dynamic role assignment
  V8       lifecycle_roles.py         conservative death-respec
  V9       opportunistic_respec.py    eager death-respec
  V10-V12  phase_strategies.py        phase switching / absorption-window raids / fortress on a lead
  V13-V16  counterplay_strategies.py  storage siege / home sentinel / carrier convoy / extended V10
  V17      v17_planner.py             team planner that blocks the enemy base exit
  V18      v18_counter.py             counter to V17
  V19      v19_counter.py             counter to V18 (highest Elo)

HeuristicPolicyMixture (mixture.py) samples among all of them for data collection.
docs/heuristic_policy_catalog_ko.md has the lineage, per-version behaviour and Elo ratings.
"""

from .strategic import StrategicHeuristic
from .advanced import StrategicHeuristicV2
from .safe_storage import StrategicHeuristicV3
from .spread_deposit import StrategicHeuristicV4
from .intercept import StrategicHeuristicV5
from .risk_path import StrategicHeuristicV6
from .v4_family import V4FamilySample, V4PolicyFamily
from .dynamic_roles import StrategicHeuristicV7
from .lifecycle_roles import StrategicHeuristicV8
from .opportunistic_respec import StrategicHeuristicV9
from .phase_strategies import StrategicHeuristicV10, StrategicHeuristicV11, StrategicHeuristicV12
from .counterplay_strategies import (
    StrategicHeuristicV13, StrategicHeuristicV14, StrategicHeuristicV15, StrategicHeuristicV16,
)
from .v17_planner import StrategicHeuristicV17
from .v18_counter import StrategicHeuristicV18
from .v19_counter import StrategicHeuristicV19
from .mixture import HeuristicPolicyMixture, POLICY_REGISTRY, PolicySample, make_heuristic

StrategicHeuristicV1 = StrategicHeuristic
RecommendedStrategicHeuristic = StrategicHeuristicV4

__all__ = [
    "StrategicHeuristic", "StrategicHeuristicV1", "StrategicHeuristicV2",
    "StrategicHeuristicV3",
    "StrategicHeuristicV4",
    "StrategicHeuristicV5",
    "StrategicHeuristicV6",
    "StrategicHeuristicV7",
    "StrategicHeuristicV8",
    "StrategicHeuristicV9",
    "StrategicHeuristicV10", "StrategicHeuristicV11", "StrategicHeuristicV12",
    "StrategicHeuristicV13", "StrategicHeuristicV14", "StrategicHeuristicV15", "StrategicHeuristicV16",
    "StrategicHeuristicV17", "StrategicHeuristicV18", "StrategicHeuristicV19",
    "RecommendedStrategicHeuristic",
    "V4FamilySample", "V4PolicyFamily",
    "HeuristicPolicyMixture", "POLICY_REGISTRY", "PolicySample", "make_heuristic",
]
