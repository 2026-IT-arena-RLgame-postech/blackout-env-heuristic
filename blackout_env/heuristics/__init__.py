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
from .opportunistic_respec import StrategicHeuristicV9
from .phase_strategies import StrategicHeuristicV10, StrategicHeuristicV11, StrategicHeuristicV12
from .counterplay_strategies import (
    StrategicHeuristicV13, StrategicHeuristicV14, StrategicHeuristicV15, StrategicHeuristicV16,
)
from .v17_planner import StrategicHeuristicV17
from .v18_counter import StrategicHeuristicV18
from .v19_counter import StrategicHeuristicV19
from .teach import StrategicHeuristicV4Teach, TeachLabel, TeachLabeller
from .v17_teach import StrategicHeuristicV17Teach
from .v20_anticamp import StrategicHeuristicV20, StrategicHeuristicV20Teach
from .v21_safe_transform import StrategicHeuristicV21, StrategicHeuristicV21Teach
from .any_angle import AnyAngleFieldMixin, AnyAngleMixin, StrategicHeuristicV21AnyAngle, StrategicHeuristicV21AnyAngleED
from .v22_teacher import StrategicHeuristicV22, StrategicHeuristicV22Teach
from .raider import StrategicRaider
from .cargo_hunter import StrategicCargoHunter
from .early_rush import StrategicEarlyRush
from .fortress import StrategicFortress
from .duelist import StrategicDuelist
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
    "StrategicHeuristicV17", "StrategicHeuristicV18", "StrategicHeuristicV19", "StrategicHeuristicV20", "StrategicHeuristicV21", "StrategicHeuristicV21AnyAngle", "StrategicHeuristicV21AnyAngleED", "AnyAngleMixin", "AnyAngleFieldMixin", "StrategicHeuristicV22", "StrategicHeuristicV22Teach",
    "StrategicRaider", "StrategicCargoHunter", "StrategicEarlyRush", "StrategicFortress", "StrategicDuelist",
    "RecommendedStrategicHeuristic",
    "StrategicHeuristicV4Teach", "StrategicHeuristicV17Teach", "StrategicHeuristicV20Teach", "StrategicHeuristicV21Teach", "TeachLabel", "TeachLabeller",
    "V4FamilySample", "V4PolicyFamily",
    "HeuristicPolicyMixture", "POLICY_REGISTRY", "PolicySample", "make_heuristic",
]
