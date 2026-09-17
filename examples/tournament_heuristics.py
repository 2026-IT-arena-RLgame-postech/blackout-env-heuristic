"""Parallel, side-swapped round-robin evaluation for every named heuristic policy.

The result matrix stores the row policy's win rate against the column policy.  Every
unordered pairing is played in both physical sides for each map seed, so spawn-side
advantage does not become a fake strategy advantage.  The script intentionally uses
the same ``play`` and movement-failure monitor as ``benchmark_heuristics.py``.

Example
-------
./.venv/bin/python examples/tournament_heuristics.py --n-seeds 5 --workers 4 --time-scale 200
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# The desktop sandbox does not expose the normal per-user matplotlib cache.  The parent uses
# this for the final PNG only; keeping pyplot out of spawned workers avoids concurrent font
# cache construction during the actual Unity tournament.
_MPL_CACHE = Path("/private/tmp/blackout_heuristic_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.heuristics import (
    StrategicHeuristicV1, StrategicHeuristicV2, StrategicHeuristicV3,
    StrategicHeuristicV4, StrategicHeuristicV5, StrategicHeuristicV6,
    StrategicHeuristicV7, StrategicHeuristicV8, StrategicHeuristicV9,
    StrategicHeuristicV10, StrategicHeuristicV11, StrategicHeuristicV12,
    StrategicHeuristicV13, StrategicHeuristicV14, StrategicHeuristicV15, StrategicHeuristicV16,
    StrategicHeuristicV17, StrategicHeuristicV18, StrategicHeuristicV19, V4PolicyFamily,
)
try:  # direct ``python examples/...`` execution
    from benchmark_heuristics import aggregate, play
except ModuleNotFoundError:  # package/module import used by tests and notebooks
    from examples.benchmark_heuristics import aggregate, play


POLICIES = {
    "v1": StrategicHeuristicV1,
    "v2": StrategicHeuristicV2,
    "v3": StrategicHeuristicV3,
    "v4": StrategicHeuristicV4,
    "v4-near": V4PolicyFamily,
    "v5": StrategicHeuristicV5,
    "v6": StrategicHeuristicV6,
    "v7": StrategicHeuristicV7,
    "v8": StrategicHeuristicV8,
    "v9": StrategicHeuristicV9,
    "v10": StrategicHeuristicV10,
    "v11": StrategicHeuristicV11,
    "v12": StrategicHeuristicV12,
    "v13": StrategicHeuristicV13,
    "v14": StrategicHeuristicV14,
    "v15": StrategicHeuristicV15,
    "v16": StrategicHeuristicV16,
    "v17": StrategicHeuristicV17,
    "v18": StrategicHeuristicV18,
    "v19": StrategicHeuristicV19,
}


def _run_pair(task: tuple[str, str, list[int], str, float]) -> dict[str, Any]:
    """Run all side-swapped games for one unordered pair in an isolated Unity process."""
    row_id, column_id, seeds, build, time_scale = task
    env = BlackOutEnv(
        build, time_scale=time_scale, no_graphics=True, additional_args=["-logFile", "/dev/null"],
        unity_shaping=False,
    )
    games = []
    try:
        for seed in seeds:
            for swapped in (False, True):
                games.append(play(env, int(seed), swapped, POLICIES[row_id], POLICIES[column_id]))
    finally:
        env.close()

    margins = np.asarray(
        [(game.candidate_score - game.baseline_score) * 100.0 for game in games], dtype=np.float64
    )
    wins = sum(game.winner == 0 for game in games)
    draws = sum(game.winner is None for game in games)
    return {
        "row": row_id,
        "column": column_id,
        "games": len(games),
        "wins": wins,
        "losses": len(games) - wins - draws,
        "draws": draws,
        "win_rate": (wins + 0.5 * draws) / len(games),
        "mean_margin": float(margins.mean()),
        "paired_margins": [float(margins[i:i + 2].mean()) for i in range(0, len(margins), 2)],
        "row_reliability": aggregate(games, "candidate_failures"),
        "column_reliability": aggregate(games, "baseline_failures"),
        "row_modes": _mode_summary(games),
    }


def _mode_summary(games) -> dict[str, Any]:
    mode_ticks: dict[str, int] = {}
    transition_count = 0
    for game in games:
        transition_count += len(game.strategy_transitions)
        for mode, ticks in game.candidate_mode_ticks.items():
            mode_ticks[mode] = mode_ticks.get(mode, 0) + ticks
    total = sum(mode_ticks.values())
    return {
        "mode_ticks": mode_ticks,
        "mode_share": {mode: ticks / total for mode, ticks in mode_ticks.items()} if total else {},
        "transitions": transition_count,
    }


def _write_csv(results: list[dict[str, Any]], output: Path) -> None:
    fields = ["row", "column", "games", "wins", "losses", "draws", "win_rate", "mean_margin"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def _plot_heatmap(policy_ids: list[str], results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    values = np.full((len(policy_ids), len(policy_ids)), np.nan, dtype=np.float64)
    index = {policy: i for i, policy in enumerate(policy_ids)}
    for result in results:
        row, column = index[result["row"]], index[result["column"]]
        values[row, column] = result["win_rate"]
        values[column, row] = 1.0 - result["win_rate"]
    np.fill_diagonal(values, 0.5)

    fig, axis = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axis.set_xticks(range(len(policy_ids)), policy_ids, rotation=45, ha="right")
    axis.set_yticks(range(len(policy_ids)), policy_ids)
    axis.set_xlabel("column policy (opponent)")
    axis.set_ylabel("row policy")
    axis.set_title("Heuristic round-robin win rate (row policy wins)")
    for row in range(len(policy_ids)):
        for column in range(len(policy_ids)):
            value = values[row, column]
            label = "—" if row == column else f"{value:.0%}"
            color = "white" if value < 0.25 or value > 0.75 else "black"
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("side-swapped win rate")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build/mac/BlackOut.app"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-rng", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-scale", type=float, default=200.0)
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=list(POLICIES))
    parser.add_argument(
        "--pairs",
        nargs="+",
        metavar="ROW:OPPONENT",
        help="run only these unordered policy pairs; useful for targeted replications",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted output directory")
    args = parser.parse_args()
    if args.n_seeds < 5:
        parser.error("--n-seeds must be at least 5")
    if len(args.policies) < 2:
        parser.error("provide at least two policies")
    if args.workers < 1:
        parser.error("--workers must be positive")

    policy_ids = list(dict.fromkeys(args.policies))
    seeds = np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.n_seeds, replace=False
    ).astype(int).tolist()
    output_dir = args.output_dir or Path("reports") / f"heuristic_tournament_{datetime.now():%Y%m%d_%H%M%S}"
    if args.pairs:
        requested_pairs: set[tuple[str, str]] = set()
        policy_order = {policy: position for position, policy in enumerate(policy_ids)}
        for specification in args.pairs:
            try:
                row_id, column_id = specification.split(":", maxsplit=1)
            except ValueError:
                parser.error(f"invalid --pairs value {specification!r}; use ROW:OPPONENT")
            if row_id not in policy_order or column_id not in policy_order:
                parser.error(f"unknown policy in --pairs value {specification!r}")
            if row_id == column_id:
                parser.error(f"a policy cannot play itself: {specification!r}")
            requested_pairs.add(
                (row_id, column_id) if policy_order[row_id] < policy_order[column_id] else (column_id, row_id)
            )
        all_tasks = [
            (row_id, column_id, seeds, str(args.build), args.time_scale)
            for row_id, column_id in sorted(requested_pairs, key=lambda pair: (policy_order[pair[0]], policy_order[pair[1]]))
        ]
    else:
        all_tasks = [
            (policy_ids[left], policy_ids[right], seeds, str(args.build), args.time_scale)
            for left in range(len(policy_ids))
            for right in range(left + 1, len(policy_ids))
        ]
    partial_path = output_dir / "completed_pair_results.jsonl"
    manifest_path = output_dir / "progress.json"
    results: list[dict[str, Any]] = []
    if args.resume:
        if not output_dir.is_dir() or not partial_path.is_file() or not manifest_path.is_file():
            parser.error("--resume requires an existing tournament output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("policy_ids") != policy_ids or manifest.get("seeds") != seeds:
            parser.error("--resume policy IDs or generated seeds do not match the existing run")
        results = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line]
        completed = {(result["row"], result["column"]) for result in results}
        tasks = [task for task in all_tasks if (task[0], task[1]) not in completed]
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        tasks = all_tasks
        manifest = {
            "policy_ids": policy_ids,
            "seeds": seeds,
            "n_seeds": args.n_seeds,
            "side_swapped": True,
            "time_scale": args.time_scale,
            "workers": args.workers,
            "total_pairs": len(all_tasks),
            "completed_pairs": 0,
            "complete": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"policies={policy_ids}", flush=True)
    print(f"seeds={seeds} remaining_pairs={len(tasks)} games={len(tasks) * len(seeds) * 2} workers={args.workers}", flush=True)
    # A full 13-policy tournament is intentionally long-running.  Persist each completed
    # pair immediately so an interrupted desktop session never gets mistaken for a complete
    # heatmap and the completed evidence remains usable for a later resume/export.
    context = mp.get_context("spawn")
    with partial_path.open("a" if args.resume else "w", encoding="utf-8") as partial_handle:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_pair, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                partial_handle.write(json.dumps(result) + "\n")
                partial_handle.flush()
                manifest["completed_pairs"] = len(results)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(
                    f"[{len(results)}/{len(all_tasks)}] {result['row']} vs {result['column']}: "
                    f"{result['wins']}-{result['losses']}-{result['draws']} "
                    f"win={result['win_rate']:.1%} margin={result['mean_margin']:+.2f}",
                    flush=True,
                )

    results.sort(key=lambda result: (policy_ids.index(result["row"]), policy_ids.index(result["column"])))
    metadata = {
        "policy_ids": policy_ids,
        "seeds": seeds,
        "n_seeds": args.n_seeds,
        "side_swapped": True,
        "time_scale": args.time_scale,
        "workers": args.workers,
        "results": results,
    }
    (output_dir / "tournament.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_csv(results, output_dir / "pair_results.csv")
    _plot_heatmap(policy_ids, results, output_dir / "win_rate_heatmap.png")
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
