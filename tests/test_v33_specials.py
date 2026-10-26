"""V33 (reports/heur_v33.md): V32 + special-item weight knobs, dominance evasion and Hunter-vs-Hunter knobs; all
defaults = V32 exactly."""

import numpy as np

from blackout_env.heuristics.mixture import POLICY_REGISTRY
from blackout_env.heuristics.params import ALIASES, resolve_descriptor, split_descriptor
from blackout_env.heuristics.v32_four_hunters import StrategicHeuristicV32
from blackout_env.heuristics.v33_specials import StrategicHeuristicV33

from .test_v22_teacher import _fixtures
from .test_v32_four_hunters import ROLES, V31_KW


def test_v33_registered_and_alias():
    assert POLICY_REGISTRY["strategic_v33"] is StrategicHeuristicV33
    policy, kv = split_descriptor(resolve_descriptor("V33"))
    assert policy == "strategic_v33" and kv["commit_margin"] < 0 and kv["w_bspd"] > 1.0
    assert "V33" in ALIASES


def test_v33_special_weights_knobs():
    p = StrategicHeuristicV33(w_dspd=5.0, w_bspd=4.5)
    assert p.special_weights == (4.5, 5.0, 0.12, 0.12)
    assert StrategicHeuristicV33().special_weights == StrategicHeuristicV32().special_weights


def test_v33_default_is_v32_on_fixtures():
    for obs in _fixtures():
        x = StrategicHeuristicV32(**V31_KW, **ROLES).act(obs)
        y = StrategicHeuristicV33(**V31_KW, **ROLES).act(obs)
        assert set(x) == set(y) and all(np.array_equal(x[n], y[n]) for n in x)
