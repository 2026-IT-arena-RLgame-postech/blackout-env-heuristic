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
from .phase_strategies import StrategicHeuristicV10, StrategicHeuristicV11, StrategicHeuristicV12
from .counterplay_strategies import (
    StrategicHeuristicV13, StrategicHeuristicV14, StrategicHeuristicV15, StrategicHeuristicV16,
)
from .v17_planner import StrategicHeuristicV17
from .v18_counter import StrategicHeuristicV18
from .v19_counter import StrategicHeuristicV19


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
    "strategic_v10": StrategicHeuristicV10,
    "strategic_v11": StrategicHeuristicV11,
    "strategic_v12": StrategicHeuristicV12,
    "strategic_v13": StrategicHeuristicV13,
    "strategic_v14": StrategicHeuristicV14,
    "strategic_v15": StrategicHeuristicV15,
    "strategic_v16": StrategicHeuristicV16,
    "strategic_v17": StrategicHeuristicV17,
    "strategic_v18": StrategicHeuristicV18,
    "strategic_v19": StrategicHeuristicV19,
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
    """Sample a policy version once per match, plus a bounded parameter cloud around it.

    Which strategy (``policy_id``) generated a trajectory only changes at a real match
    reset, keeping each match behaviourally coherent -- across matches, the complete
    dataset still covers several strategies and navigation styles.  Within a match, by
    default (``resample_each_absorption``) the small numeric perturbation around that same
    strategy is redrawn at every absorption boundary (this game's natural episode boundary,
    see qmix_trainer's module docstring) instead of staying fixed for the whole 420s match,
    the same way V4PolicyFamily stays a "near V4" policy while sampling a fresh nearby point
    each time -- more local coverage per match for BC/offline-RL, without ever swapping to a
    behaviourally different heuristic mid-match.  The redraw retunes the running policy in
    place, so its paths, role assignments, strategy mode and respec state carry across the
    boundary; ``use_specialists`` is structural (it switches the role scheme) and stays fixed
    for the match.  ``current_sample`` is deliberately public so collectors can persist
    provenance with every trajectory.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        weights: dict[str, float] | None = None,
        perturb: bool = True,
        resample_each_absorption: bool = True,
    ):
        self._rng = np.random.default_rng(seed)
        # Rebalanced 2026-09-15 (second pass) against the leave-one-pair-out Bradley-Terry Elo fit
        # in reports/heuristic_draw_census_20260915_130356/elo_diagnostics_corrected/: a 16-policy
        # round robin (no v4-near), 10 games per pair, re-run after fixing BlackOutEnv's reset leak
        # that recorded early decisive wins as 0-0 draws (commit 0f48f95). The earlier fit
        # (reports/heuristic_tournament_all17_targeted_replication_round2_20260915) had more games
        # but ~27% of them were those phantom draws, which hid real losses as half-wins. Ranking,
        # Elo minus the 1500 pool mean (standard errors ~28-39):
        # v13(+103) > v11(+84) > v12(+79) > v6(+72) > v10(+64) > v3(+62) > v4(+60) = v8(+60)
        # = v9(+60) > v7(+50) > v15(-13) > v16(-37) > v2(-68) > v5(-166) > v14(-171) > v1(-237).
        # Not yet reflected: the later protected-spawn-storage guard fix, which mainly changed v14
        # (also v16, v12, v1, v2); a re-run was skipped for time.
        # Same principles as the first pass, applied to the corrected ranking:
        #  - v4/v4-near stay the dominant pair regardless of rank: v4 is the production/eval
        #    reference and v4-near its bounded neighbourhood.
        #  - v13/v11 moved from neutral to the top two and v12/v6 are close behind -- raised.
        #  - v14 and v5 collapsed to near the bottom once losses stopped counting as draws -- cut to
        #    the contrast-case minimum alongside v1/v2.
        #  - v8 stays low despite its rank: its respec rarely fires, so most games duplicate v7
        #    (see lifecycle_roles.py).
        #  - v16 still rates well below its own base v10 -- kept low.
        default_weights = {
            "strategic_v1": 0.02,   # Elo -237, weakest; trivial-play contrast only
            "strategic_v2": 0.02,   # Elo -68
            "strategic_v3": 0.07,   # Elo +62, "safe deposit" style
            "strategic_v4": 0.16,   # production/eval reference (dominant regardless of Elo)
            "strategic_v4_near": 0.13,  # near-V4 neighbourhood, same rationale as v4
            "strategic_v5": 0.02,   # Elo -166; aggressive interceptor, weak in direct play
            "strategic_v6": 0.07,   # Elo +72, risk-aware routing
            "strategic_v7": 0.06,   # Elo +50, role-assignment diversity
            "strategic_v8": 0.03,   # Elo +60 but mostly duplicates V7 (see above)
            "strategic_v9": 0.06,   # Elo +60; respec trajectories actually fire
            "strategic_v10": 0.07,  # Elo +64; phase switching economy / pressure / closeout
            "strategic_v11": 0.07,  # Elo +84, 2nd; bounded absorption-window raids
            "strategic_v12": 0.07,  # Elo +79, 3rd; lead-preserving fortress trajectories
            "strategic_v13": 0.07,  # Elo +103, strongest; persistent storage siege
            "strategic_v14": 0.02,  # Elo -171 (before the guard fix); home sentinel style
            "strategic_v15": 0.03,  # Elo -13; Carrier throughput race with no Hunter
            "strategic_v16": 0.03,  # Elo -37, below its own base V10 (see above)
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
        self.resample_each_absorption = bool(resample_each_absorption)
        self.current_sample: PolicySample | None = None
        self._policy: BaseModel | None = None
        self._policy_id: str | None = None
        self._last_time_left: float | None = None
        self._last_absorption: float | None = None
        self._match_use_specialists: bool | None = None
        self.reset()

    def reset(self) -> PolicySample:
        """Pick this match's baseline strategy, then sample its first parameter cloud."""
        names = tuple(self.weights)
        probabilities = np.asarray([self.weights[n] for n in names], dtype=np.float64)
        probabilities /= probabilities.sum()
        self._policy_id = str(self._rng.choice(names, p=probabilities))
        self._last_time_left = None
        self._last_absorption = None
        return self._resample_variation(new_match=True)

    def _resample_variation(self, *, new_match: bool = False) -> PolicySample:
        """Redraw the bounded parameter cloud around the match's already-chosen policy_id.

        Called once from reset() (``new_match``: builds a fresh policy) and, while
        resample_each_absorption is set, again at every absorption boundary -- there
        policy_id never changes and the running policy is retuned in place, keeping its
        episode state.
        """
        policy_id = self._policy_id
        assert policy_id is not None
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
            if new_match:
                self._policy = family
            else:
                # The fresh family only served as the sampler for this seed's profile.
                assert isinstance(self._policy, V4PolicyFamily)
                self._policy.adopt_sample(family.current_sample)
        elif self.perturb:
            # Always consume the draw so the rest of the stream is identical either way.
            use_specialists = bool(episode_rng.random() >= 0.08)
            if new_match:
                self._match_use_specialists = use_specialists
            parameters = {
                "use_specialists": self._match_use_specialists,
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
            elif policy_id == "strategic_v10":
                parameters.update({
                    "mode_confirm_ticks": int(episode_rng.integers(14, 23)),
                    "pressure_absorption": float(episode_rng.uniform(0.32, 0.48)),
                    "closeout_lead": float(episode_rng.uniform(0.07, 0.12)),
                })
            elif policy_id == "strategic_v11":
                parameters.update({
                    "mode_confirm_ticks": int(episode_rng.integers(8, 17)),
                    "raid_slots": int(episode_rng.integers(1, 3)),
                    "raid_absorption": float(episode_rng.uniform(0.42, 0.66)),
                })
            elif policy_id == "strategic_v12":
                parameters.update({
                    "mode_confirm_ticks": int(episode_rng.integers(15, 25)),
                    "fortress_lead": float(episode_rng.uniform(0.05, 0.11)),
                })
            elif policy_id == "strategic_v13":
                parameters.update({
                    "mode_confirm_ticks": int(episode_rng.integers(6, 15)),
                    "raid_slots": int(episode_rng.integers(2, 4)),
                    "siege_min_value": float(episode_rng.uniform(0.5, 2.5)),
                })
            elif policy_id == "strategic_v14":
                parameters.update({
                    "guard_radius": float(episode_rng.uniform(5.5, 8.5)),
                })
            elif policy_id == "strategic_v15":
                parameters.update({
                    "cargo_amount_weight": float(episode_rng.uniform(15.0, 26.0)),
                })
            elif policy_id == "strategic_v16":
                parameters.update({
                    "mode_confirm_ticks": int(episode_rng.integers(8, 18)),
                    "siege_value": float(episode_rng.uniform(5.0, 11.0)),
                    "guard_radius": float(episode_rng.uniform(5.5, 8.5)),
                    "convoy_field_battery": float(episode_rng.uniform(40.0, 70.0)),
                })
        else:
            parameters = {
                "use_specialists": True,
                "replan_interval": 10,
                "threat_radius": 0.16,
            }
        if policy_id != "strategic_v4_near":
            if new_match:
                self._policy = make_heuristic(policy_id, **parameters)
            else:
                self._policy.retune(**parameters)
        self.current_sample = PolicySample(policy_id, policy_seed, parameters)
        return self.current_sample

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        assert self._policy is not None
        if obs:
            team_state = next(iter(obs.values()))["team_state"]
            time_left = float(team_state[2])
            if self._last_time_left is not None and time_left > self._last_time_left + 0.25:
                self.reset()
            elif self.resample_each_absorption and self.perturb:
                # absorption_time_left counts down each tick and snaps back up the tick it
                # fires (see qmix_trainer.collect_step) -- a tiny epsilon catches exactly that.
                absorption_left = float(team_state[3])
                if self._last_absorption is not None and absorption_left > self._last_absorption + 1e-6:
                    self._resample_variation()
                self._last_absorption = absorption_left
            self._last_time_left = time_left
        # Explicit reset() is still preferred for collectors: metadata can be written before
        # the first transition, rather than discovered on its first action.
        return self._policy.act(obs)
