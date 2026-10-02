import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, StrategicHeuristicV22, StrategicHeuristicV23

from .test_v22_teacher import _fixtures


def _same(a, b):
    for obs in _fixtures():
        x, y = a.act(obs), b.act(obs)
        assert set(x) == set(y)
        for name in x:
            assert np.array_equal(x[name], y[name])


def test_v23_is_registered():
    assert isinstance(POLICY_REGISTRY["strategic_v23"](), StrategicHeuristicV23)


def test_without_the_opening_field_it_is_v22():
    _same(StrategicHeuristicV22(), StrategicHeuristicV23(opening_field=False))


def test_after_the_race_it_is_v22(monkeypatch):
    monkeypatch.setattr(StrategicHeuristicV22, "_racing", lambda self, ctx: False)
    _same(StrategicHeuristicV22(), StrategicHeuristicV23())


def test_v23_is_a_function_of_the_observation():
    played = StrategicHeuristicV23()
    for obs in _fixtures():
        x, y = played.act(obs), StrategicHeuristicV23().act(obs)
        for name in x:
            assert np.array_equal(x[name], y[name])
