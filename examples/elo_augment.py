"""Add games where they shrink Elo standard errors most, on top of an existing tournament.

The round-robin in ``tournament_adaptive.py`` equalises *per-pair* win-rate standard error, which
is not what an Elo fit needs: a 100%-0% pair carries almost no Fisher information however many
games it holds, while a 50% pair between two uncertain policies carries the most.  This script
reads the existing games, fits Bradley-Terry, and then repeatedly plays the pairs that maximise
the drop in summed rating variance (greedy A-optimal, ``elo_active.choose_pairs``) until every
rating's standard error is under ``--target-se`` Elo.

Games use the same early decision as the tournament: the match stops once no battery is left on
the map, in storage or held by a unit, and the higher score wins.

Example
-------
./.venv/bin/python examples/elo_augment.py --workers 18 --target-se 8 \\
    --prior reports/heuristic_tournament_adaptive_se05_20260918/games.jsonl \\
    --output reports/elo_augment_20260918
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:  # direct ``python examples/...`` execution
    import gauntlet_heuristics
    from elo_active import ELO, choose_pairs, fit_bradley_terry
    from tournament_adaptive import _label, _play_one
except ModuleNotFoundError:
    from examples import gauntlet_heuristics
    from examples.elo_active import ELO, choose_pairs, fit_bradley_terry
    from examples.tournament_adaptive import _label, _play_one

DEFAULT_POLICIES = [f"strategic_v{i}" for i in range(1, 20)]


def _as_fit_games(records: list[dict]) -> list[dict]:
    return [{"a": r["candidate"], "b": r["opponent"], "winner": r["winner"]} for r in records]


def _read_games(path: Path, policies: set[str]) -> list[dict]:
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in records if r["candidate"] in policies and r["opponent"] in policies]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--prior", type=Path, nargs="*", default=[],
                        help="games.jsonl files from earlier runs, used as the starting fit")
    parser.add_argument("--target-se", type=float, default=8.0, help="stop once every rating's SE is below this (Elo)")
    parser.add_argument("--max-games", type=int, default=40000)
    parser.add_argument("--pairs-per-round", type=int, default=18)
    parser.add_argument("--per-policy", type=int, default=3, help="pairs one policy may take per round")
    parser.add_argument("--seeds-per-pair", type=int, default=1, help="each seed is played from both sides")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--seed-rng", type=int, default=20260918)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policies = list(dict.fromkeys(args.policies))
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "games.jsonl"
    prior = [record for path in args.prior for record in _read_games(path, set(policies))]
    played = _read_games(log_path, set(policies))  # resume
    rng = np.random.default_rng(args.seed_rng)

    ratings, covariance, _, counts = fit_bradley_terry(policies, _as_fit_games(prior + played))
    standard_errors = np.sqrt(np.diag(covariance)) * ELO
    print(f"prior games={len(prior)} resumed={len(played)} max SE={standard_errors.max():.2f} Elo", flush=True)

    with log_path.open("a") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=gauntlet_heuristics._init_worker, initargs=(args.build, args.time_scale),
    ) as pool:
        while standard_errors.max() > args.target_se and len(played) < args.max_games:
            pairs = choose_pairs(policies, ratings, covariance, counts, args.pairs_per_round,
                                 args.per_policy, games_per_pair=2 * args.seeds_per_pair)
            tasks = [(a, b, int(rng.integers(1, 2**30)), swapped)
                     for a, b in pairs for _ in range(args.seeds_per_pair) for swapped in (False, True)]
            for record in pool.map(_play_one, tasks):
                log.write(json.dumps(record) + "\n")
                played.append(record)
            log.flush()
            ratings, covariance, _, counts = fit_bradley_terry(policies, _as_fit_games(prior + played))
            standard_errors = np.sqrt(np.diag(covariance)) * ELO
            worst = int(np.argmax(standard_errors))
            print(f"[games +{len(played)}] max SE={standard_errors.max():.2f} ({_label(policies[worst])}) "
                  f"mean={standard_errors.mean():.2f} | last round: "
                  + ", ".join(f"{_label(a)}v{_label(b)}" for a, b in pairs[:6]), flush=True)

    order = np.argsort(-ratings)
    summary = {"policies": policies, "target_se": args.target_se,
               "prior_games": len(prior), "added_games": len(played),
               "ratings": {policies[i]: {"elo": float(ratings[i] * ELO + 1500.0),
                                         "se": float(standard_errors[i]),
                                         "games": int(counts[i].sum())} for i in order}}
    (args.output / "ratings.json").write_text(json.dumps(summary, indent=1))
    lines = [f"{'policy':16s} {'Elo':>8s} {'SE':>6s} {'95% interval':>17s} {'games':>7s}"]
    for i in order:
        elo = ratings[i] * ELO + 1500.0
        lines.append(f"{policies[i]:16s} {elo:8.1f} {standard_errors[i]:6.2f} "
                     f"{elo - 1.96 * standard_errors[i]:8.1f}-{elo + 1.96 * standard_errors[i]:<8.1f} "
                     f"{int(counts[i].sum()):7d}")
    text = "\n".join(lines)
    (args.output / "ratings.txt").write_text(text + "\n")
    print("\n" + text, flush=True)
    print(f"output={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
