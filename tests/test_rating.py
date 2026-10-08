"""blackout_env/rating.py: Bradley-Terry fit, A-optimal picks, paired-seed stopping SE (examples/elo24.py)."""
import json
import math
from pathlib import Path

import numpy as np

from blackout_env.rating import ELO, choose_pairs, fit_bt, pair_seed_stats, plan_round, rating_se_elo, seeds_needed


def _games(true, n, rng, draw=0.0):
    i_idx, j_idx, sc = [], [], []
    k = len(true)
    for i in range(k):
        for j in range(i + 1, k):
            p = 1 / (1 + math.exp(-(true[i] - true[j])))
            for _ in range(n):
                i_idx.append(i)
                j_idx.append(j)
                sc.append(0.5 if rng.random() < draw else float(rng.random() < p))
    return i_idx, j_idx, sc


def test_fit_recovers_ratings_and_se_is_calibrated():
    rng = np.random.default_rng(0)
    true = np.array([0.0, 100, 250, 400, 600]) / ELO
    r, cov, wins, counts = fit_bt(5, *_games(true, 400, rng, draw=0.05), prior_sd_elo=4000)
    assert abs(r.sum()) < 1e-9
    err = (r - (true - true.mean())) * ELO
    se = rating_se_elo(cov)
    assert np.all(np.abs(err) < 4 * se), (err, se)
    assert np.allclose(wins + wins.T, counts)


def test_draws_count_half_and_prior_keeps_a_perfect_record_finite():
    r, _, wins, _ = fit_bt(2, [0, 0], [1, 1], [0.5, 0.5])
    assert abs(r[0] - r[1]) < 1e-9 and wins[0, 1] == 1.0
    r, cov, _, _ = fit_bt(2, [0] * 20, [1] * 20, [1.0] * 20, prior_sd_elo=400)
    assert np.isfinite(r).all() and r[0] > r[1] and np.isfinite(cov).all()


def test_sherman_morrison_matches_refit_information():
    """The variance drop a pick claims equals the drop of a direct inverse with that pair's information added."""
    rng = np.random.default_rng(1)
    true = np.array([0.0, 50, 300, 800]) / ELO
    i_idx, j_idx, sc = _games(true, 6, rng)
    r, cov, _, counts = fit_bt(4, i_idx, j_idx, sc)
    (i, j), = choose_pairs(r, cov, 1, 3, games_per_pair=8)
    p = 1 / (1 + math.exp(-(r[i] - r[j])))
    e = np.zeros(4)
    e[i], e[j] = 1, -1
    w = 8 * p * (1 - p)
    ce = cov @ e
    sm = cov - w / (1 + w * e @ ce) * np.outer(ce, ce)
    direct = np.linalg.pinv(np.linalg.pinv(cov) + w * np.outer(e, e))
    assert np.allclose(np.trace(sm), np.trace(direct), rtol=1e-6)
    gains = []  # it is the best single pick
    for a in range(4):
        for b in range(a + 1, 4):
            q = 1 / (1 + math.exp(-(r[a] - r[b])))
            wa = 8 * q * (1 - q)
            c = cov[:, a] - cov[:, b]
            gains.append(wa * c @ c / (1 + wa * (c[a] - c[b])))
    c = cov[:, i] - cov[:, j]
    assert math.isclose(w * c @ c / (1 + w * (c[i] - c[j])), max(gains))


def test_choose_pairs_respects_per_policy_and_distinct_pairs():
    rng = np.random.default_rng(2)
    true = rng.uniform(0, 600, 8) / ELO
    r, cov, _, _ = fit_bt(8, *_games(true, 2, rng))
    picks = choose_pairs(r, cov, 50, 2, 4)
    assert len(set(picks)) == len(picks) and all(i < j for i, j in picks)
    used = np.bincount(np.ravel(picks), minlength=8)
    assert used.max() <= 2 and len(picks) == 8  # 8 policies x 2 / 2


def test_stopping_se_smooths_lopsided_pairs():
    st = pair_seed_stats([1.0] * 10)  # 20-0
    assert st["se"] == 0 and math.isclose(st["se_stop"], 1 / 12)
    assert seeds_needed([1.0] * 10, 0.05) == 18
    assert pair_seed_stats([1.0] * 18)["se_stop"] <= 0.05 < pair_seed_stats([1.0] * 17)["se_stop"]
    even = [0.0, 0.5, 1.0, 0.5] * 25  # 100 seeds of independent coin flips
    assert pair_seed_stats(even)["se_stop"] < 0.05
    side_bound = [0.5] * 10  # the side decides every game: paired SE tiny, smoothing still asks for seeds
    assert pair_seed_stats(side_bound)["se_stop"] > 0.05


def test_plan_round_pairs_first_then_elo_then_done():
    k = 4
    rng = np.random.default_rng(3)
    true = np.array([0.0, 30, 60, 90]) / ELO
    scores = {(i, j): [float(rng.random() < 0.5) for _ in range(10)] for i in range(k) for j in range(i + 1, k)}
    i_idx, j_idx, sc = _games(true, 20, rng)
    r, cov, _, _ = fit_bt(k, i_idx, j_idx, sc)
    plan, why = plan_round(scores, r, cov, max_add=10)
    assert why == "pair_se" and set(plan) == set(scores) and all(2 <= n <= 10 for n in plan.values())
    settled = {p: [0.5] * 60 + [0.0, 1.0] * 20 for p in scores}
    assert all(pair_seed_stats(x)["se_stop"] <= 0.05 for x in settled.values())
    plan, why = plan_round(settled, r, cov, elo_se=1.0, elo_pairs=2, elo_seeds=4)
    assert why == "elo_se" and len(plan) == 2 and set(plan.values()) == {4}
    plan, why = plan_round(settled, r, cov, elo_se=1e9)
    assert (plan, why) == ({}, "done")


def test_config_heuristic_specs_match_the_adopted_constants():
    from blackout_env.heuristics.v32_four_hunters import V32_1
    from blackout_env.train.distill.ood_suite import V31

    cfg = json.loads((Path(__file__).resolve().parents[1] / "examples/elo24_config.json").read_text())
    spec = {p["name"]: p["spec"] for p in cfg["policies"]}
    assert len(spec) == 24 and spec["V32.1"] == V32_1 and spec["V31"] == V31
