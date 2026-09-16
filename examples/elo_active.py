"""Estimate Elo for every registered heuristic without filling the whole win-rate matrix.

Adaptive Bradley-Terry rating:

1. Seed round: every policy plays four others (two random cycles), one map seed from both sides.
2. Fit Bradley-Terry by Newton's method with a weak Gaussian prior (sd 400 Elo) so ratings stay
   defined before the comparison graph is dense; the inverse Hessian gives rating covariances.
3. Each later round greedily picks the pairs whose games most reduce the summed variance of all
   ratings (A-optimal design, exact via Sherman-Morrison).  Close pairs between uncertain
   ratings win most picks; a far-stronger policy still gets the lopsided games that anchor it.
   Each policy plays at most ``--per-policy`` pairs per round to keep the sample balanced.
4. Stop when every rating's standard error is below ``--target-se`` or the game budget is spent.
5. Hold-out check: play pairs never scheduled and compare their observed win rate with the
   fitted prediction.

Games use race_gauntlet's 64s truncation (matches are decided in the first minute; its winner
agreed with full 420s matches).  Draws count as half a win.  Elo is one-dimensional, so
deliberately non-transitive counters (V18 beats V17, V19 beats V18) show up as the largest
residuals rather than in the ratings.

Example
-------
./.venv/bin/python examples/elo_active.py --workers 18 --target-se 40 --output reports/elo_active
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY

try:  # direct ``python examples/...`` execution
    from race_gauntlet import _init_worker, _play
except ModuleNotFoundError:
    from examples.race_gauntlet import _init_worker, _play

ELO = 400.0 / math.log(10.0)


def fit_bradley_terry(policies, games, prior_sd_elo=400.0, iterations=50):
    """Return (ratings centred at 0, their covariance, win matrix, game-count matrix).

    Only rating differences are identified by games; a common shift is pinned only by the prior,
    with variance ~ prior_var / k.  The covariance is therefore projected onto centred ratings
    (P C P, P = I - 11^T/k); without it every standard error carries that unidentified shift
    (400/sqrt(13) = 111 Elo for 13 policies) and never shrinks however many games are played.
    """
    k = len(policies)
    index = {p: i for i, p in enumerate(policies)}
    wins = np.zeros((k, k))
    counts = np.zeros((k, k))
    for g in games:
        i, j = index[g["a"]], index[g["b"]]
        score = 1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0
        wins[i, j] += score
        wins[j, i] += 1.0 - score
        counts[i, j] += 1
        counts[j, i] += 1
    lam = 1.0 / (prior_sd_elo / ELO) ** 2
    r = np.zeros(k)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))
        grad = (wins - counts * p).sum(axis=1) - lam * r
        w = counts * p * (1.0 - p)
        hessian = w.copy()
        np.fill_diagonal(hessian, -w.sum(axis=1) - lam)
        step = np.linalg.solve(hessian, -grad)
        r += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))
    w = counts * p * (1.0 - p)
    information = -w.copy()
    np.fill_diagonal(information, w.sum(axis=1) + lam)
    covariance = np.linalg.inv(information)
    # P C P written out (row/column/grand means): same numbers, no large dense matmul.
    centred = (covariance - covariance.mean(axis=0, keepdims=True)
               - covariance.mean(axis=1, keepdims=True) + covariance.mean())
    return r - r.mean(), centred, wins, counts


def choose_pairs(policies, r, covariance, counts, n_pairs, per_policy, games_per_pair=2):
    """Greedy A-optimal batch: each pick maximises the drop in the summed rating variance.

    Playing ``g`` more games of pair (i, j) adds Fisher information ``w = g * p(1-p)`` along
    ``e = e_i - e_j``; by Sherman-Morrison the summed variance falls by
    ``w * |C e|^2 / (1 + w * e^T C e)``.  The covariance is updated after every pick so a batch
    does not pile onto the same uncertainty.  Unlike ranking by p(1-p) alone, this still buys the
    lopsided games that anchor a far-stronger policy to the pool when nothing else can.
    """
    k = len(policies)
    cov = covariance.copy()
    used = np.zeros(k, dtype=int)
    taken = set()
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
                gain = w * float(ce @ ce) / (1.0 + w * quad)
                if best is None or gain > best[0]:
                    best = (gain, i, j, w, ce, quad)
        if best is None:
            break
        _, i, j, w, ce, quad = best
        cov -= (w / (1.0 + w * quad)) * np.outer(ce, ce)
        used[i] += 1
        used[j] += 1
        taken.add((i, j))
        chosen.append((policies[i], policies[j]))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--policies", nargs="+", default=sorted(
        POLICY_REGISTRY, key=lambda p: (len(p), p)))
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--cutoff-steps", type=int, default=1600)
    parser.add_argument("--target-se", type=float, default=40.0, help="Elo standard error to stop at")
    parser.add_argument("--max-games", type=int, default=1500)
    parser.add_argument("--pairs-per-round", type=int, default=18)
    parser.add_argument("--seeds-per-pair", type=int, default=1, help="each seed is played from both sides")
    parser.add_argument("--per-policy", type=int, default=3)
    parser.add_argument("--holdout-pairs", type=int, default=12)
    parser.add_argument("--holdout-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path("reports/elo_active"))
    parser.add_argument("--refit", action="store_true",
                        help="rewrite the summary from games.jsonl without playing")
    args = parser.parse_args()

    policies = list(args.policies)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "games.jsonl"
    games: list[dict] = []
    if log_path.exists():  # resume
        games = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        games = [g for g in games if g["a"] in policies and g["b"] in policies and not g.get("holdout")]
        print(f"resumed {len(games)} games", flush=True)

    def play_pairs(pool, pairs, seeds_per_pair, holdout=False):
        tasks = []
        for a, b in pairs:
            for _ in range(seeds_per_pair):
                seed = int(rng.integers(1, 2**30))
                for swapped in (False, True):
                    tasks.append((a, (a, {}), b, seed, swapped, args.cutoff_steps))
        results = []
        with log_path.open("a") as log:
            for task, result in zip(tasks, pool.map(_play, tasks)):
                record = {"a": task[0], "b": task[2], "seed": task[3], "swapped": task[4],
                          "winner": result["winner"], "own": result["own"], "opp": result["opp"],
                          "holdout": holdout}
                log.write(json.dumps(record) + "\n")
                results.append(record)
        return results

    k = len(policies)
    holdout = []
    if args.refit:
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        records = [g for g in records if g["a"] in policies and g["b"] in policies]
        r, cov, wins, counts = fit_bradley_terry(policies, games)
        index = {p: i for i, p in enumerate(policies)}
        by_pair: dict[tuple[str, str], list[dict]] = {}
        for g in records:
            if g.get("holdout"):
                by_pair.setdefault((g["a"], g["b"]), []).append(g)
        for (a, b), pair_games in by_pair.items():
            i, j = index[a], index[b]
            observed = np.mean([1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0
                                for g in pair_games])
            holdout.append((a, b, 1.0 / (1.0 + math.exp(-(r[i] - r[j]))), float(observed), len(pair_games)))
        return _report(args, policies, games, r, cov, wins, counts, holdout)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(args.build, args.time_scale)) as pool:
        if not games:
            order = list(rng.permutation(k))
            seed_pairs = {tuple(sorted((order[i], order[(i + step) % k]))) for step in (1, 2) for i in range(k)}
            pairs = [(policies[i], policies[j]) for i, j in sorted(seed_pairs)]
            print(f"seed round: {len(pairs)} pairs", flush=True)
            games += play_pairs(pool, pairs, args.seeds_per_pair)

        round_index = 0
        while True:
            r, cov, wins, counts = fit_bradley_terry(policies, games)
            se = np.sqrt(np.diag(cov)) * ELO
            played_pairs = int(np.count_nonzero(np.triu(counts, 1)))
            print(f"round {round_index}: games={len(games)} pairs={played_pairs}/{k * (k - 1) // 2} "
                  f"max SE={se.max():.0f} mean SE={se.mean():.0f}", flush=True)
            if se.max() <= args.target_se or len(games) >= args.max_games:
                break
            pairs = choose_pairs(policies, r, cov, counts, args.pairs_per_round, args.per_policy,
                                 games_per_pair=2 * args.seeds_per_pair)
            games += play_pairs(pool, pairs, args.seeds_per_pair)
            round_index += 1

        # Hold-out: pairs never scheduled, predicted before they are played.
        r, cov, wins, counts = fit_bradley_terry(policies, games)
        unplayed = [(i, j) for i in range(k) for j in range(i + 1, k) if counts[i, j] == 0]
        if unplayed and args.holdout_pairs > 0:
            picks = rng.choice(len(unplayed), size=min(args.holdout_pairs, len(unplayed)), replace=False)
            holdout_pairs = [unplayed[int(x)] for x in picks]
            predictions = {(policies[i], policies[j]): 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
                           for i, j in holdout_pairs}
            results = play_pairs(pool, list(predictions), args.holdout_seeds, holdout=True)
            for (a, b), predicted in predictions.items():
                pair_games = [g for g in results if g["a"] == a and g["b"] == b]
                observed = np.mean([1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0
                                    for g in pair_games])
                holdout.append((a, b, predicted, float(observed), len(pair_games)))

    return _report(args, policies, games, r, cov, wins, counts, holdout)


def _report(args, policies, games, r, cov, wins, counts, holdout) -> int:
    k = len(policies)
    elo = 1500.0 + r * ELO
    se = np.sqrt(np.diag(cov)) * ELO
    games_per_policy = counts.sum(axis=1)
    order = np.argsort(-elo)
    short = {p: p.replace("strategic_", "") for p in policies}
    lines = [f"games={len(games)} (+{sum(h[4] for h in holdout)} hold-out), "
             f"pairs sampled={int(np.count_nonzero(np.triu(counts, 1)))}/{k * (k - 1) // 2}", "",
             f"{'rank':>4s} {'policy':10s} {'Elo':>6s} {'±SE':>5s} {'games':>6s}"]
    for rank, i in enumerate(order, 1):
        lines.append(f"{rank:4d} {short[policies[i]]:10s} {elo[i]:6.0f} {se[i]:5.0f} {int(games_per_policy[i]):6d}")

    residuals = []
    for i in range(k):
        for j in range(i + 1, k):
            if counts[i, j] >= 4:
                predicted = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
                observed = wins[i, j] / counts[i, j]
                residuals.append((abs(observed - predicted), short[policies[i]], short[policies[j]],
                                  observed, predicted, int(counts[i, j])))
    residuals.sort(reverse=True)
    lines += ["", "largest residuals on sampled pairs (non-transitivity shows up here)"]
    for _, a, b, observed, predicted, n in residuals[:8]:
        lines.append(f"  {a:>8s} vs {b:<8s} observed {observed:4.0%} predicted {predicted:4.0%} (n={n})")
    if holdout:
        errors = [abs(pred - obs) for _, _, pred, obs, _ in holdout]
        lines += ["", f"hold-out pairs (never used in the fit): mean |pred-obs| = {np.mean(errors):.0%}"]
        for a, b, predicted, observed, n in holdout:
            lines.append(f"  {short[a]:>8s} vs {short[b]:<8s} predicted {predicted:4.0%} observed {observed:4.0%} (n={n})")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (args.output / "summary.txt").write_text(text + "\n")
    (args.output / "ratings.json").write_text(json.dumps(
        {policies[i]: {"elo": float(elo[i]), "se": float(se[i]), "games": int(games_per_policy[i])}
         for i in range(k)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
