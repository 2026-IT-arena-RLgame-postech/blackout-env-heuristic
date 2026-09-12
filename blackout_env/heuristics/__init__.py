"""Hand-written policies useful as baselines and behaviour-cloning teachers."""

from .strategic import StrategicHeuristic
from .advanced import StrategicHeuristicV2
from .safe_storage import StrategicHeuristicV3
from .spread_deposit import StrategicHeuristicV4
from .mixture import HeuristicPolicyMixture, POLICY_REGISTRY, PolicySample, make_heuristic

StrategicHeuristicV1 = StrategicHeuristic
RecommendedStrategicHeuristic = StrategicHeuristicV4

__all__ = [
    "StrategicHeuristic", "StrategicHeuristicV1", "StrategicHeuristicV2",
    "StrategicHeuristicV3",
    "StrategicHeuristicV4",
    "RecommendedStrategicHeuristic",
    "HeuristicPolicyMixture", "POLICY_REGISTRY", "PolicySample", "make_heuristic",
]
