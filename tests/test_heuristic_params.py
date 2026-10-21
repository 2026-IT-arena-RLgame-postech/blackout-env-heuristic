"""Knob registry (heuristics/params.py): defaults are bit-exact with the pre-knob code, descriptors/aliases parse,
sampler is seeded and stays in range, and a changed knob changes behaviour."""
import json
from pathlib import Path

import numpy as np
import pytest

from blackout_env.heuristics.params import (ALIASES, KNOBS, base_values, resolve_descriptor, sample_overrides,
                                            sample_params, split_descriptor, validate, with_params)
from blackout_env.heuristics.replay_hash import BASELINES, match_hash
from blackout_env.train.distill.opponents import make_opponent

GOLDEN = json.loads((Path(__file__).parent / "data" / "heuristic_hashes.json").read_text())


@pytest.mark.parametrize("name", list(BASELINES))
def test_default_actions_are_bit_exact(name):
    """Full matches (to the end) of each champion vs V28 on two maps (one per side) hash like the pre-knob code."""
    for seed_i in (0, 3):   # golden keys: name|4000000+i|i%2
        key = f"{name}|{4_000_000 + seed_i}|{seed_i % 2}"
        got = match_hash(BASELINES[name], 4_000_000 + seed_i, seed_i % 2, GOLDEN["steps"])
        assert got == GOLDEN["hashes"][key], key


def test_aliases_equal_baselines():
    for name, desc in BASELINES.items():
        assert resolve_descriptor(name) == desc or split_descriptor(name) == split_descriptor(desc)


def test_alias_with_overrides_and_old_descriptors():
    assert make_opponent("V32.1", 0).adapt_k == 3
    p = make_opponent("V32.1:hunter_quota=3,sweep_radius=7.5", 0)
    assert (p.hunter_quota, p.sweep_radius, p.adapt_k, p.sweep) == (3, 7.5, 3, True)
    assert make_opponent("V32.1,hunter_quota=2", 0).hunter_quota == 2
    assert make_opponent("strategic_v19", 0).hunter_quota == 4          # plain descriptors unchanged
    assert make_opponent(BASELINES["V32.1"] + ",hunter_quota=3", 0).hunter_quota == 3


def test_new_knobs_default_to_the_old_literals():
    p = make_opponent("V32.1", 0)
    assert (p.hunt_threshold, p.carrier_field_value, p.danger_hunter_radius, p.danger_collector_radius,
            p.danger_weight, p.contest_factor, p.score_band, p.exclusion_margin, p.deposit_band, p.idle_band,
            p.hunt_spawn_radius, p.hunt_spawn_value, p.hunt_camp_radius, p.hunt_camp_value, p.chase_equal,
            p.hunter_speed) == (0.35, 25.0, 4.0, 2.5, 0.6, 0.75, 1.12, 1.0, 0.25, 0.3, 7.0, 12.0, 3.0, 4.0, 2.5, 6.0)


def test_every_knob_is_a_constructor_keyword_of_some_base():
    covered = set()
    for base in ALIASES:
        covered |= set(base_values(base)[1])
    assert covered == set(KNOBS)


def test_sampler_is_seeded_and_in_range():
    a = [sample_params("V32.1", np.random.default_rng(7), 1.5) for _ in range(1)]
    b = [sample_params("V32.1", np.random.default_rng(7), 1.5) for _ in range(1)]
    assert a == b
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        d = sample_params("V32.1", rng, 2.0)
        validate(d)
        make_opponent(d, 0)
        seen.add(d)
        _, vals = base_values("V32.1")
        for k, v in sample_overrides("V32.1", rng, 2.0).items():
            assert KNOBS[k].lo <= v <= KNOBS[k].hi and not KNOBS[k].structural
    assert len(seen) > 30


def test_changed_knobs_change_behaviour():
    base = match_hash("V32.1", 4_000_001, 1, 1500)
    for ov in ({"hunter_quota": 2}, {"threat_radius": 0.2}, {"danger_hunter_radius": 6.0}, {"steal_weight": 3.0}):
        assert match_hash(with_params("V32.1", **ov), 4_000_001, 1, 1500) != base, ov
