"""Bradley-Terry ratings, A-optimal pair selection and paired-seed win-rate SE (examples/elo24.py).

Pure numpy, no simulator imports, so the maths is unit-tested on its own (tests/test_rating.py).

* ``fit_bt`` -- Newton MLE of Bradley-Terry with a weak Gaussian prior (as examples/elo_active.py); a game's score
  is 1 / 0.5 / 0 (win / draw / loss).  Returns centred ratings (natural log-odds units; ``* ELO`` for Elo) and the
  covariance projected onto centred ratings (only differences are identified by games).
* ``choose_pairs`` -- greedy A-optimal batch: each pick maximises the drop in summed rating variance, exact by
  Sherman-Morrison, the covariance updated after every pick.
* ``pair_seed_stats`` -- a pair's win rate from paired seeds (each seed = both side-swapped games; the seed score
  is their mean), SE over seeds (the repo's paired-seed convention), and the *stopping* SE: the same SE with two
  pseudo-seeds, one won and one lost (= +2 wins +2 losses, Agresti-Coull style), so a 20-0 pair is not called
  settled after 10 seeds (raw SE 0, smoothed 8.3 %).  All won: smoothed SE = 1 / (n + 2) -> 18 seeds for 5 %;
  a coin-flip pair with independent games needs ~50 seeds (100 games).
"""

from __future__ import annotations

import math

import numpy as np

ELO = 400.0 / math.log(10.0)
PSEUDO_SEEDS = (0.0, 1.0)  # stopping-SE smoothing: one lost + one won seed


def fit_bt(k: int, i_idx, j_idx, score, prior_sd_elo: float = 400.0, iterations: int = 100):
    """Games (i, j, score of i) -> (ratings r centred at 0, centred covariance, wins[k,k], counts[k,k])."""
    i_idx, j_idx = np.asarray(i_idx, int), np.asarray(j_idx, int)
    score = np.asarray(score, float)
    wins = np.zeros((k, k))
    counts = np.zeros((k, k))
    np.add.at(wins, (i_idx, j_idx), score)
    np.add.at(wins, (j_idx, i_idx), 1.0 - score)
    np.add.at(counts, (i_idx, j_idx), 1.0)
    np.add.at(counts, (j_idx, i_idx), 1.0)
    lam = 1.0 / (prior_sd_elo / ELO) ** 2
    r = np.zeros(k)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))
        grad = (wins - counts * p).sum(axis=1) - lam * r
        w = counts * p * (1.0 - p)
        hess = w.copy()
        np.fill_diagonal(hess, -w.sum(axis=1) - lam)
        step = np.linalg.solve(hess, -grad)
        r += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))
    w = counts * p * (1.0 - p)
    info = -w.copy()
    np.fill_diagonal(info, w.sum(axis=1) + lam)
    cov = np.linalg.inv(info)
    cov = cov - cov.mean(axis=0, keepdims=True) - cov.mean(axis=1, keepdims=True) + cov.mean()
    return r - r.mean(), cov, wins, counts


def rating_se_elo(cov) -> np.ndarray:
    return np.sqrt(np.maximum(np.diag(cov), 0.0)) * ELO


def focus_se_elo(cov, focus) -> np.ndarray:
    """Elo SEs of the ``focus`` ratings relative to their own mean (how well their order and gaps are known)."""
    f = np.asarray(focus, int)
    m = len(f)
    proj = np.eye(m) - 1.0 / m
    c = proj @ np.asarray(cov, float)[np.ix_(f, f)] @ proj
    return np.sqrt(np.maximum(np.diag(c), 0.0)) * ELO


def choose_pairs(r, cov, n_pairs: int, per_policy: int, games_per_pair: int, exclude=(), focus=None):
    """Greedy A-optimal batch of index pairs (i < j); each policy in at most ``per_policy`` picks.
    ``focus``: indices whose summed variance (relative to their own mean) is minimised instead of all ratings'."""
    f = None if focus is None else np.asarray(focus, int)
    k = len(r)
    cov = np.array(cov, float, copy=True)
    used = np.zeros(k, int)
    taken = set(exclude)
    chosen = []
    for _ in range(n_pairs):
        best = None
        for i in range(k):
            if used[i] >= per_policy:
                continue
            for j in range(i + 1, k):
                if used[j] >= per_policy or (i, j) in taken:
                    continue
                p = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
                w = games_per_pair * p * (1.0 - p)
                ce = cov[:, i] - cov[:, j]
                quad = ce[i] - ce[j]
                v = ce if f is None else ce[f] - ce[f].mean()
                gain = w * float(v @ v) / (1.0 + w * quad)
                if best is None or gain > best[0]:
                    best = (gain, i, j, w, ce, quad)
        if best is None:
            break
        _, i, j, w, ce, quad = best
        cov -= (w / (1.0 + w * quad)) * np.outer(ce, ce)
        used[i] += 1
        used[j] += 1
        taken.add((i, j))
        chosen.append((i, j))
    return chosen


def _sd(x) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def pair_seed_stats(seed_scores) -> dict:
    """Per-seed scores of one side of a pair (mean of its two side-swapped games, each 1 / 0.5 / 0) ->
    n seeds, win rate p, raw paired SE, smoothed stopping SE and its augmented variance."""
    x = np.asarray(seed_scores, float)
    n = len(x)
    aug = np.concatenate([x, PSEUDO_SEEDS])
    var_aug = float(np.var(aug, ddof=1))
    return {"n": n, "p": float(x.mean()) if n else float("nan"),
            "se": _sd(x) / math.sqrt(n) if n > 1 else float("nan"),
            "se_stop": math.sqrt(var_aug / len(aug)), "var_aug": var_aug}


def seeds_needed(seed_scores, target_se: float) -> int:
    """Smallest real-seed count whose stopping SE <= target if more seeds keep the current mean and variance
    (the pseudo-seeds' weight shrinks as n grows, so a 10-0 pair needs 18 seeds, not var_aug / target^2)."""
    x = np.asarray(seed_scores, float)
    n = len(x)
    m = float(x.mean()) if n else 0.5
    v = float(x.var()) if n else 0.125  # population variance of the real seeds; no data: a coin flip pair
    pseudo = np.asarray(PSEUDO_SEEDS)
    for n2 in range(max(n, 1), 100_000):
        tot = n2 + len(pseudo)
        mean = (n2 * m + pseudo.sum()) / tot
        ss = n2 * (v + (m - mean) ** 2) + float(((pseudo - mean) ** 2).sum())
        if math.sqrt(ss / (tot - 1) / tot) <= target_se:
            return n2
    return 100_000


def plan_round(seed_scores: dict, r, cov, *, pair_se: float = 0.05, elo_se: float = 12.0, min_add: int = 2,
               max_add: int = 10, elo_pairs: int = 24, per_policy: int = 3, elo_seeds: int = 4, focus=None):
    """One stage-2 round: {(i, j): extra seeds}.  Phase A (first): every pair whose stopping SE > ``pair_se`` gets
    clamp(seeds_needed - n, min_add, max_add) seeds.  Phase B (only when every pair passes): ``elo_pairs``
    A-optimal picks of ``elo_seeds`` seeds while some rating SE > ``elo_se``.  {} = both criteria met.
    ``seed_scores``: {(i, j): [seed scores of i]} for every pair i < j.  ``focus``: phase B targets only these
    ratings' SEs relative to their own mean (``focus_se_elo``)."""
    plan = {}
    for pair, xs in seed_scores.items():
        if pair_seed_stats(xs)["se_stop"] > pair_se:
            plan[pair] = int(np.clip(seeds_needed(xs, pair_se) - len(xs), min_add, max_add))
    if plan:
        return plan, "pair_se"
    se = rating_se_elo(cov) if focus is None else focus_se_elo(cov, focus)
    if se.max() > elo_se:
        for pair in choose_pairs(r, cov, elo_pairs, per_policy, 2 * elo_seeds, focus=focus):
            plan[pair] = elo_seeds
        return plan, "elo_se"
    return {}, "done"
