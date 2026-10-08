"""S1.6 bootstrapped branch targets (blackout_env/value/branch.py)."""

import numpy as np
import pytest

from blackout_env.value.branch import H2, H5, H10, H20, bootstrap_targets, mirror_targets
from blackout_env.value.evaluate import symmetrize


def _v(final, d5, p):
    n = len(final)
    delta = np.zeros((n, 4), np.float32)
    delta[:, H5] = d5
    return {"final": np.asarray(final, np.float32), "delta": delta, "p": np.asarray(p, np.float32)}


def test_bootstrap_live_row():
    # s: lead +10; +2 s: +12; s' (+5 s): +20; V_sym(s') says final +35 (score/100 0.35), +5 s more = +4
    t = bootstrap_targets([10], [12], [20], [False], _v([0.35], [0.04], [[0.1, 0.0, 0.9]]))
    assert t["lead"][0] == pytest.approx(0.10)
    assert t["final"][0] == pytest.approx(0.35)
    assert t["remaining"][0] == pytest.approx(0.25)
    assert t["delta"][0, H2] == pytest.approx(0.02)
    assert t["delta"][0, H5] == pytest.approx(0.10)
    assert t["delta"][0, H10] == pytest.approx(0.14)
    assert not t["delta_mask"][0, H20] and t["delta_mask"][0, [H2, H5, H10]].all()
    np.testing.assert_allclose(t["p"][0], [0.1, 0.0, 0.9])


def test_bootstrap_terminal_row_uses_true_final_and_ignores_v():
    # ended inside the 5 s: lead5 is the final lead (-30); V at s' must not matter
    for v in (_v([5.0], [9.0], [[0, 0, 1]]), None):
        t = bootstrap_targets([10], [-30], [-30], [True], v)
        assert t["final"][0] == pytest.approx(-0.30)
        assert t["remaining"][0] == pytest.approx(-0.40)
        np.testing.assert_allclose(t["delta"][0], [-0.40] * 4, rtol=1e-6)
        assert t["delta_mask"][0].all()
        np.testing.assert_allclose(t["p"][0], [1, 0, 0])  # a loss
    draw = bootstrap_targets([0], [0], [0], [True], None)
    np.testing.assert_allclose(draw["p"][0], [0, 1, 0])
    with pytest.raises(ValueError):
        bootstrap_targets([0], [0], [0], [False], None)


def test_mirror_targets_antisymmetric():
    rng = np.random.default_rng(0)
    n = 50
    ended = rng.random(n) < 0.3
    lead = rng.integers(-60, 60, n)
    lead5 = lead + rng.integers(-20, 20, n)
    lead2 = lead + rng.integers(-10, 10, n)
    p = rng.dirichlet(np.ones(3), n)
    t = bootstrap_targets(lead, lead2, lead5, ended, _v(rng.normal(0, 0.3, n), rng.normal(0, 0.05, n), p))
    m = mirror_targets(t)
    for k in ("lead", "final", "remaining", "delta"):
        np.testing.assert_allclose(m[k], -t[k])
    np.testing.assert_allclose(m["p"], t["p"][:, ::-1])
    np.testing.assert_array_equal(m["delta_mask"], t["delta_mask"])
    np.testing.assert_allclose(m["p"].sum(1), 1, rtol=1e-6)
    # the mirror's targets are what bootstrapping from the other team's side gives: negated leads,
    # and a symmetric V at s' read from the other side (symmetrize(mir, own)) -> negated / reversed
    own = {"delta": rng.normal(0, 0.1, (n, 4)), "final": rng.normal(0, 0.3, n), "p": p}
    mir = {"delta": rng.normal(0, 0.1, (n, 4)), "final": rng.normal(0, 0.3, n), "p": rng.dirichlet(np.ones(3), n)}
    v_a, v_b = symmetrize(own, mir), symmetrize(mir, own)
    ta = bootstrap_targets(lead, lead2, lead5, ended, v_a)
    tb = bootstrap_targets(-lead, -lead2, -lead5, ended, v_b)
    mt = mirror_targets(ta)
    for k in ("final", "remaining", "delta", "p"):
        np.testing.assert_allclose(tb[k], mt[k], atol=1e-6)


def test_mirror_is_involution():
    t = bootstrap_targets([5, -7], [6, -7], [9, -20], [False, True], _v([0.2, 0.0], [0.01, 0.0], [[0.2, 0.1, 0.7]] * 2))
    mm = mirror_targets(mirror_targets(t))
    for k in t:
        np.testing.assert_allclose(mm[k], t[k])
