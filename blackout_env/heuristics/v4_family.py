"""A tightly bounded family around the promoted V4 teacher policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.base import BaseModel
from .spread_deposit import StrategicHeuristicV4


@dataclass(frozen=True)
class V4FamilySample:
    """Episode-level provenance for a near-V4 policy sample."""

    policy_id: str
    profile: str
    policy_seed: int
    parameters: dict[str, float | bool]


class V4PolicyFamily(BaseModel):
    """Sample coherent, small V4 parameter variations once per episode.

    No per-step action noise or strategy replacement is used.  Consequently each member
    stays close to the validated V4 policy while providing useful local support around its
    decision boundaries for BC and offline RL.
    """

    DEFAULT_PROFILE_WEIGHTS = {
        "exact": 0.25,
        "balanced": 0.35,
        "responsive": 0.20,
        "cautious": 0.20,
    }

    def __init__(
        self,
        *,
        seed: int = 0,
        profile_weights: dict[str, float] | None = None,
        resample_each_absorption: bool = False,
    ):
        self._rng = np.random.default_rng(seed)
        self.profile_weights = dict(
            self.DEFAULT_PROFILE_WEIGHTS if profile_weights is None else profile_weights
        )
        unknown = set(self.profile_weights) - set(self.DEFAULT_PROFILE_WEIGHTS)
        if unknown:
            raise ValueError(f"Unknown V4 family profiles: {sorted(unknown)}")
        if not self.profile_weights or any(v < 0 for v in self.profile_weights.values()):
            raise ValueError("Profile weights must be non-negative and non-empty")
        if sum(self.profile_weights.values()) <= 0:
            raise ValueError("At least one profile weight must be positive")
        # False reproduces the original once-per-match cadence (tournament/benchmark scripts
        # rely on one stable profile per match for their per-match reporting). True resamples
        # a fresh nearby profile at every absorption instead, for extra BC/offline-RL sample
        # diversity within a single 420s match -- see HeuristicPolicyMixture, which turns this
        # on for its training-data role.
        self.resample_each_absorption = bool(resample_each_absorption)
        self.current_sample: V4FamilySample | None = None
        self._policy: StrategicHeuristicV4 | None = None
        self._last_time_left: float | None = None
        self._last_absorption: float | None = None
        self.reset()

    def reset(self) -> V4FamilySample:
        """Start a new match: fresh profile and a fresh V4 with no episode state."""
        sample = self._draw_sample()
        self._policy = StrategicHeuristicV4(**sample.parameters)
        self.current_sample = sample
        self._last_time_left = None
        self._last_absorption = None
        return sample

    def resample_parameters(self) -> V4FamilySample:
        """Draw a new profile but keep the running V4's paths, roles and caches."""
        return self.adopt_sample(self._draw_sample())

    def adopt_sample(self, sample: V4FamilySample) -> V4FamilySample:
        """Apply an externally drawn sample (see HeuristicPolicyMixture) in place."""
        assert self._policy is not None
        self._policy.retune(**sample.parameters)
        self.current_sample = sample
        return sample

    def _draw_sample(self) -> V4FamilySample:
        profiles = tuple(self.profile_weights)
        probabilities = np.asarray([self.profile_weights[p] for p in profiles], np.float64)
        probabilities /= probabilities.sum()
        profile = str(self._rng.choice(profiles, p=probabilities))
        policy_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(policy_seed)

        if profile == "exact":
            parameters: dict[str, float | bool] = {
                "use_specialists": True,
                "replan_interval": 10,
                "threat_radius": 0.16,
                "protected_storage_bonus": 7.0,
            }
        elif profile == "responsive":
            parameters = {
                "use_specialists": True,
                "replan_interval": int(rng.integers(8, 10)),
                "threat_radius": float(rng.uniform(0.145, 0.165)),
                "protected_storage_bonus": float(rng.uniform(5.8, 6.8)),
            }
        elif profile == "cautious":
            parameters = {
                "use_specialists": True,
                "replan_interval": int(rng.integers(10, 13)),
                "threat_radius": float(rng.uniform(0.165, 0.185)),
                "protected_storage_bonus": float(rng.uniform(7.3, 8.4)),
            }
        else:  # balanced: a narrow continuous cloud centered on exact V4
            parameters = {
                "use_specialists": True,
                "replan_interval": int(rng.integers(9, 12)),
                "threat_radius": float(rng.uniform(0.15, 0.17)),
                "protected_storage_bonus": float(rng.uniform(6.5, 7.5)),
            }

        return V4FamilySample(
            policy_id="strategic_v4_near", profile=profile,
            policy_seed=policy_seed, parameters=parameters,
        )

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        assert self._policy is not None
        if obs:
            team_state = next(iter(obs.values()))["team_state"]
            # episode_time_left only jumps upward at a match reset.
            time_left = float(team_state[2])
            if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
                self.reset()
            elif self.resample_each_absorption:
                # absorption_time_left counts down each tick and snaps back up the tick it
                # fires (see qmix_trainer.collect_step) -- a tiny epsilon catches exactly that.
                absorption_left = float(team_state[3])
                if self._last_absorption is not None and absorption_left > self._last_absorption + 1e-6:
                    self.resample_parameters()
                self._last_absorption = absorption_left
            self._last_time_left = time_left
        return self._policy.act(obs)
