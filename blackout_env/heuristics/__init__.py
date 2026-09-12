"""Hand-written policies useful as baselines and behaviour-cloning teachers."""

from .strategic import StrategicHeuristic
from .advanced import StrategicHeuristicV2
from .safe_storage import StrategicHeuristicV3
from .spread_deposit import StrategicHeuristicV4
from .intercept import StrategicHeuristicV5
from .risk_path import StrategicHeuristicV6
from .v4_family import V4FamilySample, V4PolicyFamily
from .dynamic_roles import StrategicHeuristicV7
from .lifecycle_roles import StrategicHeuristicV8
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
    "RecommendedStrategicHeuristic",
    "V4FamilySample", "V4PolicyFamily",
    "HeuristicPolicyMixture", "POLICY_REGISTRY", "PolicySample", "make_heuristic",
]
