"""Pure helpers of PPO "league arm B" (see reports/rl/ppo_league_b_setup.md); the trainer wiring is in ppo.py.

* per-opponent advantage centering,
* snapshot gate (when to freeze the learner as a new opponent),
* snapshot sampling-share ramp and the opponent weight vector (anchor share floor),
* Monte-Carlo explained variance of the critic on finished games.
"""

from __future__ import annotations

import numpy as np


def center_by_opponent(adv: np.ndarray, row_opp: np.ndarray, games_per_opp: np.ndarray, min_games: int = 10):
    """Subtract each opponent's batch-mean advantage from its rows (only opponents with ``games_per_opp >= min_games``
    distinct games in the batch; others are left untouched).  No per-opponent std.  Returns (adv_centered, raw means [n_opp])."""
    out = adv.copy()
    n_opp = len(games_per_opp)
    means = np.full(n_opp, np.nan)
    for i in range(n_opp):
        m = row_opp == i
        if m.any():
            means[i] = adv[m].mean()
            if games_per_opp[i] >= min_games:
                out[m] -= means[i]
    return out, means


def fixed_gate(score: np.ndarray, w_fixed: np.ndarray) -> float:
    """PFSP-weighted mean of the fixed-pool opponents' EMA score: sum_i w_i s_i with w the *current normalised sampling
    weights of the fixed pool* (base weight x PFSP factor, renormalised over the fixed opponents)."""
    w = np.asarray(w_fixed, np.float64)
    return float((w / w.sum() * np.asarray(score, np.float64)).sum())


def gate_open(gate_score: float, it: int, last_snap_iter: int, n_snaps: int, *, thr: float, min_gap: int,
              min_start: int, max_snaps: int) -> bool:
    """All must hold: gate score >= thr, >= min_gap iterations since the last snapshot, >= min_start since the start,
    fewer than max_snaps snapshots.  (``last_snap_iter`` = 0 before the first snapshot, so min_start also covers min_gap.)"""
    return bool(gate_score >= thr and it - last_snap_iter >= min_gap and it >= min_start and n_snaps < max_snaps)


def snap_shares(levels: list[int], steps: list[float], anchor_min: float) -> np.ndarray:
    """Sampling share of each snapshot (its ramp level's share); scaled down together if their sum would leave the fixed
    (anchor) opponents with less than ``anchor_min`` of the games."""
    s = np.array([steps[l] for l in levels], np.float64)
    cap = 1.0 - anchor_min
    if s.sum() > cap:
        s = s * (cap / s.sum())
    return s


def league_weights(w_fixed: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Full opponent sampling vector: fixed pool (PFSP-weighted, normalised) gets the remaining mass, snapshots their shares."""
    w_fixed = np.asarray(w_fixed, np.float64)
    return np.concatenate([w_fixed / w_fixed.sum() * (1.0 - shares.sum()), shares])


def pooled_std(stats) -> float:
    """std from summed (n, sum, sum_sq) triples (an iterable of 3-tuples)."""
    n = s = ss = 0.0
    for a, b, c in stats:
        n += a; s += b; ss += c
    if n < 2:
        return float("nan")
    return float(np.sqrt(max(ss / n - (s / n) ** 2, 0.0)))


def ev_mc(fin) -> float:
    """1 - Var(R - V) / Var(R) from finished-game sums fin = [games, rows, sum R, sum R^2, sum e, sum e^2] (e = R - V,
    each row of a game carries the game's final reward R).  nan if there is no data / R is constant."""
    g, n, sr, srr, se, see = (float(x) for x in fin)
    if g <= 0 or n <= 1:
        return float("nan")
    vr = srr / n - (sr / n) ** 2
    if vr < 1e-9:
        return float("nan")
    ve = see / n - (se / n) ** 2
    return 1.0 - ve / vr


def ramp_ok(*, fin, snap_adv_win, fixed_adv_win, it: int, last_step_iter: int, level: int, n_levels: int,
            every: int, min_games: int, max_ratio: float, min_ev: float) -> tuple[bool, dict]:
    """May the snapshot advance one ramp step?  Needs: not at the last level, >= ``every`` iterations since its last step
    (or since it was added), >= ``min_games`` finished games, window adv-std ratio (snapshot rows / fixed-pool rows) <=
    ``max_ratio`` and EV_mc >= ``min_ev``.  An undefined EV (constant R) does not block; an undefined ratio does."""
    sa, fa = pooled_std(snap_adv_win), pooled_std(fixed_adv_win)
    ratio = sa / fa if fa == fa and fa > 0 else float("nan")
    ev = ev_mc(fin)
    info = {"ratio": ratio, "ev_mc": ev, "games": int(fin[0])}
    ok = (level < n_levels - 1 and it - last_step_iter >= every and fin[0] >= min_games
          and ratio == ratio and ratio <= max_ratio and not (ev < min_ev))
    return bool(ok), info
