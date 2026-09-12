"""Reproducible heuristic policy families for BC and offline-RL data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..model.base import BaseModel
from .strategic import StrategicHeuristic
from .advanced import StrategicHeuristicV2
from .safe_storage import StrategicHeuristicV3
from .spread_deposit import StrategicHeuristicV4
from .intercept import StrategicHeuristicV5
from .risk_path import StrategicHeuristicV6
from .v4_family import V4PolicyFamily
from .dynamic_roles import StrategicHeuristicV7
from .lifecycle_roles import StrategicHeuristicV8
from .opportunistic_respec import StrategicHeuristicV9


@dataclass(frozen=True)
class PolicySample:
    """Metadata that should be stored alongside an episode or transition dataset."""

    policy_id: str
    policy_seed: int
    parameters: dict[str, float | bool | str]


POLICY_REGISTRY: dict[str, Callable[..., BaseModel]] = {
    "strategic_v1": StrategicHeuristic,
    "strategic_v2": StrategicHeuristicV2,
    "strategic_v3": StrategicHeuristicV3,
    "strategic_v4": StrategicHeuristicV4,
    "strategic_v5": StrategicHeuristicV5,
    "strategic_v6": StrategicHeuristicV6,
    "strategic_v4_near": V4PolicyFamily,
    "strategic_v7": StrategicHeuristicV7,
    "strategic_v8": StrategicHeuristicV8,
    "strategic_v9": StrategicHeuristicV9,
}


def make_heuristic(policy_id: str, **parameters) -> BaseModel:
    """Construct one immutable named heuristic version from the public registry."""
    try:
        constructor = POLICY_REGISTRY[policy_id]
    except KeyError as exc:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy_id {policy_id!r}; available: {available}") from exc
    return constructor(**parameters)


class HeuristicPolicyMixture(BaseModel):
    """Sample a policy version and bounded parameter perturbation once per episode.

    Sampling at episode boundaries keeps each trajectory behaviourally coherent while the
    complete dataset covers several strategies and navigation styles.  ``current_sample``
    is deliberately public so collectors can persist provenance with every trajectory.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        weights: dict[str, float] | None = None,
        perturb: bool = True,
    ):
        self._rng = np.random.default_rng(seed)
        default_weights = {
            "strategic_v1": 0.11,
            "strategic_v2": 0.10,
            "strategic_v3": 0.13,
            "strategic_v4": 0.25,
            "strategic_v4_near": 0.20,
            "strategic_v5": 0.03,  # intentionally aggressive, but weak in direct evaluation
            "strategic_v6": 0.08,
            "strategic_v7": 0.06,
            "strategic_v8": 0.01,  # conservative lifecycle control
            "strategic_v9": 0.03,  # active respec trajectories remain deliberately sparse
        }
        self.weights = dict(default_weights if weights is None else weights)
        unknown = set(self.weights) - set(POLICY_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown mixture policies: {sorted(unknown)}")
        if not self.weights or any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Mixture weights must be non-negative and non-empty")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one mixture weight must be positive")
        self.perturb = bool(perturb)
        self.current_sample: PolicySample | None = None
        self._policy: BaseModel | None = None
        self._last_time_left: float | None = None
        self.reset()

    def reset(self) -> PolicySample:
        names = tuple(self.weights)
        probabilities = np.asarray([self.weights[n] for n in names], dtype=np.float64)
        probabilities /= probabilities.sum()
        policy_id = str(self._rng.choice(names, p=probabilities))
        policy_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        episode_rng = np.random.default_rng(policy_seed)

        if policy_id == "strategic_v4_near":
            family = V4PolicyFamily(
                seed=policy_seed,
                profile_weights=None if self.perturb else {"exact": 1.0},
            )
            assert family.current_sample is not None
            parameters: dict[str, float | bool | str] = {
                "profile": family.current_sample.profile,
                **family.current_sample.parameters,
            }
            self._policy = family
        elif self.perturb:
            parameters = {
                "use_specialists": bool(episode_rng.random() >= 0.08),
                "replan_interval": int(episode_rng.integers(8, 14)),
                "threat_radius": float(episode_rng.uniform(0.135, 0.185)),
            }
            if policy_id == "strategic_v5":
                parameters.update({
                    "intercept_margin_seconds": float(episode_rng.uniform(0.20, 0.40)),
                    "intent_temperature": float(episode_rng.uniform(2.5, 6.0)),
                    "velocity_ema": float(episode_rng.uniform(0.35, 0.75)),
                })
            elif policy_id == "strategic_v6":
                parameters.update({
                    "risk_radius_tiles": float(episode_rng.uniform(2.0, 3.5)),
                    "risk_weight": float(episode_rng.uniform(2.0, 4.2)),
                })
        else:
            parameters = {
                "use_specialists": True,
                "replan_interval": 10,
                "threat_radius": 0.16,
            }
        if policy_id != "strategic_v4_near":
            self._policy = make_heuristic(policy_id, **parameters)
        self.current_sample = PolicySample(policy_id, policy_seed, parameters)
        self._last_time_left = None
        return self.current_sample

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        assert self._policy is not None
        if obs:
            time_left = float(next(iter(obs.values()))["team_state"][2])
            if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
                self.reset()
            self._last_time_left = time_left
        # Explicit reset() is still preferred for collectors: metadata can be written before
        # the first transition, rather than discovered on its first action.
        return self._policy.act(obs)
