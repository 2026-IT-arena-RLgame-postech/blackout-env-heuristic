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
from .v20_anticamp import StrategicHeuristicV20
from .v21_safe_transform import StrategicHeuristicV21
from .any_angle import StrategicHeuristicV21AnyAngle
from .raider import StrategicRaider
from .cargo_hunter import StrategicCargoHunter
from .early_rush import StrategicEarlyRush
from .fortress import StrategicFortress
from .duelist import StrategicDuelist


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
    "strategic_v20": StrategicHeuristicV20,  # not in the mixture's default weights
    "strategic_v21": StrategicHeuristicV21,  # not in the mixture's default weights
    "strategic_v21_anyangle": StrategicHeuristicV21AnyAngle,  # any-angle moves; not in the default weights
    # Named strategies on the V17 planner, each on a strategic axis the V-series lacks (held-out
    # opponents for distillation; not in the mixture's default weights):
    "strategic_raider": StrategicRaider,  # steal-first economy, Hunter escorts the thieves
    "strategic_cargo_hunter": StrategicCargoHunter,  # Hunters intercept only cargo and Carriers
    "strategic_early_rush": StrategicEarlyRush,  # 4-Hunter opening sweep, then 1 Hunter + Carrier
    "strategic_fortress": StrategicFortress,  # zone defence, protected deposits, Hunter-free tasks
    "strategic_duelist": StrategicDuelist,  # Hunters hunt Hunters, escort our cargo otherwise
}


TOP_TIER = ("strategic_v17", "strategic_v18", "strategic_v19")


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
        perturb_top_tier: bool = False,
    ):
        """perturb_top_tier: also vary V17-V19's own strategy knobs (see _top_tier_parameters),
        which ``perturb`` alone leaves at their defaults; off by default so existing collections
        replay unchanged."""
        self._rng = np.random.default_rng(seed)
        self.perturb_top_tier = bool(perturb_top_tier)
        self._match_structure: dict[str, int | str] = {}
        # Rebalanced 2026-09-16 against the 64s-truncated adaptive Bradley-Terry fits
        # (examples/elo_active.py): reports/elo_active_20260916/ for all 20 policies (SE 42-65) and
        # reports/elo_active_mid_20260916/ for the middle tier alone (78/78 pairs, 1,324 games,
        # SE 23; a difference needs ~65 Elo to be significant). Current code, including the
        # protected-spawn-storage guard fix. Full-pool Elo:
        # v19 2059 > v17 1832 = v18 1831 >> v12 1542 .. v11 1443 (middle tier) > v6 1390
        # > v5 1317 = v2 1314 > v14 1220 > v1 1192.
        # Middle tier alone: {v12 1559, v4 1538, v9 1534, v7 1527} > {v8, v13, v3, v4-near}
        # > {v11 1478, v6 1472, v10 1471, v16 1461, v15 1456}; order inside a group is noise.
        # Principles:
        #  - v17-v19 get 32% combined. Their blocking play (a Hunter camping the enemy base exit,
        #    sanctuary sweeps) never occurs in v1-v16 games, which never avoid Hunters either;
        #    without them the learner never sees those states. v17 leads: it is the strongest
        #    against v1-v16 (151-9); v18/v19 are counters to v17/v18.
        #  - Middle tier is near-flat: its Elo gaps are mostly within noise. v4/v4-near stay the
        #    largest single entries as the production/eval reference.
        #  - v7/v8/v9/v12 act almost identically (v8/v9 respec rarely fires), so they share one
        #    budget instead of each taking a full share.
        #  - v5/v2/v14/v1 are clearly weakest and stay at the contrast-case minimum.
        # Heavier weights on v17-v19 cost some collection speed (~300-400us/tick vs ~200 per team).
        default_weights = {
            "strategic_v1": 0.01,   # 1192, weakest; trivial-play contrast only
            "strategic_v2": 0.02,   # 1314
            "strategic_v3": 0.06,   # middle tier, "safe deposit" style
            "strategic_v4": 0.11,   # production/eval reference, upper middle
            "strategic_v4_near": 0.08,  # near-V4 neighbourhood
            "strategic_v5": 0.02,   # 1317; aggressive interceptor
            "strategic_v6": 0.03,   # 1390 in the full pool, bottom of the middle tier
            "strategic_v7": 0.03,   # upper middle; shares one budget with v8/v9/v12
            "strategic_v8": 0.03,   # near-duplicate of v7 (respec rarely fires)
            "strategic_v9": 0.03,   # near-duplicate of v7, occasional respec
            "strategic_v10": 0.04,  # lower middle; phase switching
            "strategic_v11": 0.04,  # lower middle; absorption-window raids
            "strategic_v12": 0.03,  # upper middle; fortress mode on a lead
            "strategic_v13": 0.06,  # middle; persistent storage siege
            "strategic_v14": 0.01,  # 1220; home sentinel
            "strategic_v15": 0.04,  # lower middle; Carrier throughput race
            "strategic_v16": 0.04,  # lower middle; conditional counterplay director
            "strategic_v17": 0.14,  # 1832; denial planner, beats v1-v16 151-9
            "strategic_v18": 0.09,  # 1831; counter to v17 (item-avoiding race, exit guard)
            "strategic_v19": 0.09,  # 2059; counter to v18 (sanctuary sweep and sentry)
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
            elif policy_id in TOP_TIER and self.perturb_top_tier:
                parameters.update(self._top_tier_parameters(policy_id, episode_rng, new_match))
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

    def _top_tier_parameters(self, policy_id: str, rng: np.random.Generator, new_match: bool) -> dict:
        """Knobs of the V17 planner family.  Role counts and the Hunter mode change who does what,
        so like ``use_specialists`` they are drawn once per match; the weights are redrawn with
        the rest at every absorption."""
        if new_match:
            default_hunters = 3 if policy_id == "strategic_v17" else 4
            self._match_structure = {
                "hunter_quota": int(rng.choice([default_hunters - 1, default_hunters], p=[0.3, 0.7])),
                "hunter_mode": str(rng.choice(["mixed", "hunt"], p=[0.8, 0.2])),
            }
        scale = rng.uniform(0.7, 1.3, size=4)
        parameters = {
            **self._match_structure,
            "exposure_cost": float(rng.uniform(0.6, 1.2)),
            "steal_weight": float(rng.uniform(1.3, 2.3)),
            "camp_engage_radius": float(rng.uniform(3.5, 5.5)),
            "special_weights": tuple(float(w * k) for w, k in zip((0.45, 0.8, 0.12, 0.12), scale)),
        }
        if policy_id in ("strategic_v18", "strategic_v19"):
            parameters["guard_radius"] = float(rng.uniform(5.5, 8.5))
        if policy_id == "strategic_v19":
            parameters.update(sweep_radius=float(rng.uniform(7.0, 11.0)), sentry_radius=float(rng.uniform(5.5, 8.5)))
        return parameters

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
