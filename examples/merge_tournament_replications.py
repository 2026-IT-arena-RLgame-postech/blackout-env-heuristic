"""Merge targeted tournament replications into a complete tournament result.

This keeps the original full round robin immutable.  A derived output combines the
extra side-swapped games only for the requested pairs and can be used directly by
the heatmap and Elo diagnostic tools.

Example
-------
./.venv/bin/python examples/merge_tournament_replications.py \
  reports/full/tournament.json reports/targeted/tournament.json \
  --output-dir reports/full_with_targeted_replications
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

try:
    from tournament_heuristics import _plot_heatmap, _write_csv
except ModuleNotFoundError:
    from examples.tournament_heuristics import _plot_heatmap, _write_csv


def _merge_reliability(base: dict[str, Any], extra: dict[str, Any], base_games: int, extra_games: int) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in set(base) | set(extra):
        prior, added = base.get(name, {}), extra.get(name, {})
        total_games = base_games + extra_games
        merged[name] = {
            "incidents": int(prior.get("incidents", 0)) + int(added.get("incidents", 0)),
            "incident_ticks": int(prior.get("incident_ticks", 0)) + int(added.get("incident_ticks", 0)),
            "worst_ticks": max(int(prior.get("worst_ticks", 0)), int(added.get("worst_ticks", 0))),
            # Both evaluations use equal-length game budgets, so game-weighting is a
            # transparent approximation without attempting to reconstruct hidden unit ticks.
            "per_1000_unit_ticks": (
                float(prior.get("per_1000_unit_ticks", 0.0)) * base_games
                + float(added.get("per_1000_unit_ticks", 0.0)) * extra_games
            ) / total_games,
        }
    return merged


def _merge_modes(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for source in (base.get("mode_ticks", {}), extra.get("mode_ticks", {})):
        for mode, ticks in source.items():
            counts[mode] = counts.get(mode, 0) + int(ticks)
    total = sum(counts.values())
    return {
        "mode_ticks": counts,
        "mode_share": {mode: ticks / total for mode, ticks in counts.items()} if total else {},
        "transitions": int(base.get("transitions", 0)) + int(extra.get("transitions", 0)),
    }


def _merge_pair(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    if (base["row"], base["column"]) != (extra["row"], extra["column"]):
        raise ValueError("pair orientation differs between source tournaments")
    previous_games, added_games = int(base["games"]), int(extra["games"])
    games = previous_games + added_games
    wins = int(base["wins"]) + int(extra["wins"])
    losses = int(base["losses"]) + int(extra["losses"])
    draws = int(base["draws"]) + int(extra["draws"])
    return {
        "row": base["row"], "column": base["column"], "games": games,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": (wins + 0.5 * draws) / games,
        "mean_margin": (
            float(base["mean_margin"]) * previous_games + float(extra["mean_margin"]) * added_games
        ) / games,
        "paired_margins": list(base.get("paired_margins", [])) + list(extra.get("paired_margins", [])),
        "row_reliability": _merge_reliability(
            base.get("row_reliability", {}), extra.get("row_reliability", {}), previous_games, added_games
        ),
        "column_reliability": _merge_reliability(
            base.get("column_reliability", {}), extra.get("column_reliability", {}), previous_games, added_games
        ),
        "row_modes": _merge_modes(base.get("row_modes", {}), extra.get("row_modes", {})),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="complete source tournament.json")
    parser.add_argument("replications", type=Path, nargs="+", help="targeted tournament.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_data = json.loads(args.base.read_text(encoding="utf-8"))
    policies = list(base_data["policy_ids"])
    results = {(row["row"], row["column"]): copy.deepcopy(row) for row in base_data["results"]}
    if len(results) != len(base_data["results"]):
        raise ValueError("base tournament contains duplicate pairs")

    sources = [str(args.base.resolve())]
    for replication_path in args.replications:
        replication = json.loads(replication_path.read_text(encoding="utf-8"))
        if list(replication["policy_ids"]) != policies:
            raise ValueError(f"policy pool does not match base: {replication_path}")
        sources.append(str(replication_path.resolve()))
        for extra in replication["results"]:
            key = (extra["row"], extra["column"])
            if key not in results:
                raise ValueError(f"replication pair not present in base: {key}")
            results[key] = _merge_pair(results[key], extra)

    ordered = sorted(results.values(), key=lambda row: (policies.index(row["row"]), policies.index(row["column"])))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "policy_ids": policies,
        "side_swapped": True,
        "results": ordered,
        "aggregate_sources": sources,
        "description": "base full round robin plus selected targeted replications",
    }
    (output_dir / "tournament.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_csv(ordered, output_dir / "pair_results.csv")
    _plot_heatmap(policies, ordered, output_dir / "win_rate_heatmap.png")
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
