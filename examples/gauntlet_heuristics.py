"""One candidate heuristic against many opponents, game-level parallel over Unity processes.

Unlike ``tournament_heuristics.py`` (full round robin, one process per pair) this plays only
candidate-vs-opponent games and spreads individual (opponent, seed, side) games over a pool
of long-lived workers, each owning one Unity process.  Every seed is played from both
physical sides, so spawn-side advantage cancels.

Example
-------
./.venv/bin/python examples/gauntlet_heuristics.py --candidate strategic_v17 --workers 18 \
    --n-seeds 4 --output reports/gauntlet_v17
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from multiprocessing import util as mp_util

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.heuristics import POLICY_REGISTRY

try:  # direct ``python examples/...`` execution
    from benchmark_heuristics import play
except ModuleNotFoundError:
    from examples.benchmark_heuristics import play

_ENV = None


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                       additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    # Pool workers exit through os._exit(), so mlagents' atexit close never runs, and the
    # interpreter-exit join then blocks forever on the gRPC thread still waiting inside
    # Exchange() for our next message -- the worker, its Unity process and pool.shutdown()
    # all hang.  multiprocessing finalizers run before that join; closing sends Unity the
    # shutdown message, which also releases the blocked gRPC thread.
    mp_util.Finalize(_ENV, _ENV.close, exitpriority=10)


def _play_one(task: tuple[str, str, int, bool]) -> dict:
    candidate, opponent, seed, swapped = task
    game = play(_ENV, seed, swapped, POLICY_REGISTRY[candidate], POLICY_REGISTRY[opponent])
    return {
        "candidate": candidate, "opponent": opponent, "seed": seed, "swapped": swapped,
        "winner": game.winner,  # 0 candidate, 1 opponent, None draw
        "candidate_score": round(game.candidate_score * 100),
        "opponent_score": round(game.baseline_score * 100),
        "steps": game.steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--candidate", default="strategic_v17", choices=sorted(POLICY_REGISTRY))
    parser.add_argument("--opponents", nargs="+",
                        default=[f"strategic_v{i}" for i in range(1, 17)])
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--seed-rng", type=int, default=20260916)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    unknown = [o for o in args.opponents if o not in POLICY_REGISTRY]
    if unknown:
        parser.error(f"unknown opponents: {unknown}")
    seeds = args.seeds or np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.n_seeds, replace=False
    ).tolist()
    tasks = [(args.candidate, opponent, int(seed), swapped)
             for opponent in args.opponents for seed in seeds for swapped in (False, True)]
    output = args.output or Path("reports") / (
        f"gauntlet_{args.candidate}_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} games, seeds={seeds}, output={output}", flush=True)

    results = []
    with (output / "games.jsonl").open("w") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=_init_worker, initargs=(args.build, args.time_scale),
    ) as pool:
        futures = [pool.submit(_play_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            mark = {0: "W", 1: "L", None: "D"}[result["winner"]]
            print(f"[{len(results):3d}/{len(tasks)}] vs {result['opponent']:14s} "
                  f"seed={result['seed']} swap={int(result['swapped'])} {mark} "
                  f"{result['candidate_score']}-{result['opponent_score']} "
                  f"steps={result['steps']}", flush=True)

    by_opponent = defaultdict(list)
    for result in results:
        by_opponent[result["opponent"]].append(result)
    summary = {}
    lines = [f"candidate={args.candidate} seeds={seeds}", "",
             f"{'opponent':14s} {'W-L-D':>9s} {'win%':>6s} {'margin':>7s} {'score':>11s}"]
    for opponent in args.opponents:
        games = by_opponent[opponent]
        wins = sum(g["winner"] == 0 for g in games)
        losses = sum(g["winner"] == 1 for g in games)
        draws = len(games) - wins - losses
        margin = float(np.mean([g["candidate_score"] - g["opponent_score"] for g in games]))
        own = float(np.mean([g["candidate_score"] for g in games]))
        opp = float(np.mean([g["opponent_score"] for g in games]))
        rate = (wins + 0.5 * draws) / len(games)
        summary[opponent] = {"wins": wins, "losses": losses, "draws": draws,
                             "win_rate": rate, "mean_margin": margin}
        lines.append(f"{opponent:14s} {wins:3d}-{losses}-{draws:<3d} {rate:6.0%} "
                     f"{margin:+7.1f} {own:5.1f}-{opp:<5.1f}")
    total_w = sum(s["wins"] for s in summary.values())
    total_l = sum(s["losses"] for s in summary.values())
    total_d = sum(s["draws"] for s in summary.values())
    lines += ["", f"total {total_w}-{total_l}-{total_d}"]
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (output / "summary.txt").write_text(text + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
