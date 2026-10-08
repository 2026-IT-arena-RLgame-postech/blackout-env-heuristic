"""Explain where a single-axis Elo fit disagrees with a heuristic tournament.

The tournament scores a draw as half a win.  This script fits a Bradley--Terry model
on that same convention, then refits it once per pair with that pair withheld.  The
leave-one-pair-out (LOO) probability is deliberately used for diagnostics: it answers
"what would Elo have predicted before seeing this matchup?", rather than rewarding a
model for the outcome it was fitted on.

Example
-------
./.venv/bin/python examples/analyze_elo_diagnostics.py \
  reports/heuristic_tournament_all17_workers18_20260913/tournament.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

# The desktop sandbox does not expose the regular per-user matplotlib cache.
_MPL_CACHE = Path("/private/tmp/blackout_elo_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE = Path("/private/tmp/blackout_elo_cache")
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib.pyplot as plt
import numpy as np


def _probability(logit_difference: float) -> float:
    if logit_difference >= 0:
        return 1.0 / (1.0 + math.exp(-logit_difference))
    exp_difference = math.exp(logit_difference)
    return exp_difference / (1.0 + exp_difference)


def _fit_elo(policy_ids: list[str], results: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Fit centred natural-logit ratings and return their covariance matrix."""
    count = len(policy_ids)
    free_count = count - 1
    index = {policy: i for i, policy in enumerate(policy_ids)}
    free_ratings = np.zeros(free_count, dtype=np.float64)

    for _ in range(100):
        ratings = np.concatenate((free_ratings, [-float(free_ratings.sum())]))
        gradient_full = np.zeros(count, dtype=np.float64)
        information_full = np.zeros((count, count), dtype=np.float64)
        for result in results:
            row, column = index[result["row"]], index[result["column"]]
            games = int(result["games"])
            points = int(result["wins"]) + 0.5 * int(result["draws"])
            probability = _probability(float(ratings[row] - ratings[column]))
            gradient = points - games * probability
            curvature = games * probability * (1.0 - probability)
            gradient_full[row] += gradient
            gradient_full[column] -= gradient
            information_full[row, row] += curvature
            information_full[column, column] += curvature
            information_full[row, column] -= curvature
            information_full[column, row] -= curvature

        # The final rating is -sum(free ratings), enforcing a zero-mean pool.
        gradient = gradient_full[:-1] - gradient_full[-1]
        information = (
            information_full[:-1, :-1]
            - information_full[:-1, -1:]
            - information_full[-1:, :-1]
            + information_full[-1, -1]
        )
        update = np.linalg.solve(information, gradient)
        # A leave-one-pair-out fit can start from a flat rating vector while its held-out
        # result was the only strong bridge between two styles.  Damping avoids a transient
        # Newton overshoot into near-perfect separation, where curvature collapses and the
        # covariance would become numerically meaningless.
        max_update = float(np.max(np.abs(update)))
        if max_update > 0.5:
            update *= 0.5 / max_update
        free_ratings += update
        if float(np.max(np.abs(update))) < 1e-11:
            break

    # Re-evaluate at the solution for the observed information / covariance.
    ratings = np.concatenate((free_ratings, [-float(free_ratings.sum())]))
    information_full = np.zeros((count, count), dtype=np.float64)
    for result in results:
        row, column = index[result["row"]], index[result["column"]]
        probability = _probability(float(ratings[row] - ratings[column]))
        curvature = int(result["games"]) * probability * (1.0 - probability)
        information_full[row, row] += curvature
        information_full[column, column] += curvature
        information_full[row, column] -= curvature
        information_full[column, row] -= curvature
    information = (
        information_full[:-1, :-1]
        - information_full[:-1, -1:]
        - information_full[-1:, :-1]
        + information_full[-1, -1]
    )
    covariance_free = np.linalg.inv(information)
    # The final rating is constrained to the negative sum of the free ratings.
    # Expand the covariance explicitly rather than multiplying by the constraint
    # transform.  Besides making the relationship clearer, this avoids an
    # intermittent BLAS overflow warning observed for otherwise finite matrices.
    covariance = np.empty((count, count), dtype=np.float64)
    covariance[:-1, :-1] = covariance_free
    final_cross_covariance = -covariance_free.sum(axis=0)
    covariance[-1, :-1] = final_cross_covariance
    covariance[:-1, -1] = final_cross_covariance
    covariance[-1, -1] = covariance_free.sum()
    return ratings, covariance


def _observed_standard_error(result: dict[str, Any]) -> float:
    """Empirical SE of win points; a draw contributes the observed score 0.5."""
    games = int(result["games"])
    observed = float(result["win_rate"])
    values = np.array(
        [1.0] * int(result["wins"])
        + [0.5] * int(result["draws"])
        + [0.0] * int(result["losses"]),
        dtype=np.float64,
    )
    if games < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(games))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output: Path, pairs: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    expected = np.array([float(pair["loo_expected_win_rate"]) for pair in pairs])
    observed = np.array([float(pair["observed_win_rate"]) for pair in pairs])
    residual = observed - expected
    z_score = np.array([float(pair["loo_standardized_residual"]) for pair in pairs])
    scale = np.maximum(20.0, np.abs(z_score) * 38.0)

    scatter = axes[0].scatter(expected, observed, c=np.abs(z_score), s=scale, cmap="magma", alpha=0.84)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1, label="Elo perfect prediction")
    axes[0].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="LOO Elo expected win rate", ylabel="Observed win rate", title="Held-out Elo prediction vs experiment")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.colorbar(scatter, ax=axes[0], label="|standardized residual|")

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhspan(-1.96, 1.96, color="#b8e3b8", alpha=0.45, label="approx. 95% consistency band")
    axes[1].scatter(expected, z_score, c=np.where(np.abs(z_score) >= 1.96, "#c33", "#3c78b4"), s=42, alpha=0.85)
    axes[1].set(xlim=(-0.02, 1.02), xlabel="LOO Elo expected win rate", ylabel="Standardized residual", title="Where observed results depart from Elo")
    axes[1].legend(loc="upper right", fontsize=8)

    ordered = sorted(policy_rows, key=lambda row: float(row["loo_mae"]), reverse=True)
    names = [row["policy"] for row in ordered]
    errors = [float(row["loo_mae"]) for row in ordered]
    axes[2].barh(names[::-1], errors[::-1], color="#7451a6")
    axes[2].set(xlabel="Mean absolute LOO prediction error", title="Policies least explained by one-axis Elo")
    axes[2].set_xlim(left=0)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tournament", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.tournament.read_text(encoding="utf-8"))
    policies = list(data["policy_ids"])
    results = list(data["results"])
    output_dir = args.output_dir or args.tournament.parent / "elo_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {policy: i for i, policy in enumerate(policies)}
    elo_scale = 400.0 / math.log(10.0)

    ratings, covariance = _fit_elo(policies, results)
    pair_rows: list[dict[str, Any]] = []
    for held_out, result in enumerate(results):
        training = results[:held_out] + results[held_out + 1:]
        loo_ratings, loo_covariance = _fit_elo(policies, training)
        row, column = index[result["row"]], index[result["column"]]
        delta = float(loo_ratings[row] - loo_ratings[column])
        expected = _probability(delta)
        observed = float(result["win_rate"])
        rating_difference_variance = float(
            loo_covariance[row, row] + loo_covariance[column, column] - 2.0 * loo_covariance[row, column]
        )
        prediction_se = expected * (1.0 - expected) * math.sqrt(max(0.0, rating_difference_variance))
        observed_se = _observed_standard_error(result)
        total_se = math.sqrt(prediction_se ** 2 + observed_se ** 2)
        residual = observed - expected
        pair_rows.append({
            "row": result["row"], "column": result["column"], "games": int(result["games"]),
            "wins": int(result["wins"]), "losses": int(result["losses"]), "draws": int(result["draws"]),
            "observed_win_rate": observed, "loo_expected_win_rate": expected,
            "residual_observed_minus_expected": residual,
            "observed_se": observed_se, "elo_prediction_se": prediction_se,
            "combined_se": total_se,
            "loo_standardized_residual": residual / total_se if total_se > 1e-12 else 0.0,
            "elo_expected_win_rate_in_sample": _probability(float(ratings[row] - ratings[column])),
        })

    policy_rows: list[dict[str, Any]] = []
    for policy, policy_index in index.items():
        related = [
            row for row in pair_rows if row["row"] == policy or row["column"] == policy
        ]
        residuals = [abs(float(row["residual_observed_minus_expected"])) for row in related]
        standardized = [abs(float(row["loo_standardized_residual"])) for row in related]
        policy_rows.append({
            "policy": policy,
            "elo": 1500.0 + float(ratings[policy_index]) * elo_scale,
            "elo_standard_error": math.sqrt(float(covariance[policy_index, policy_index])) * elo_scale,
            "elo_95_low": 1500.0 + (float(ratings[policy_index]) - 1.96 * math.sqrt(float(covariance[policy_index, policy_index]))) * elo_scale,
            "elo_95_high": 1500.0 + (float(ratings[policy_index]) + 1.96 * math.sqrt(float(covariance[policy_index, policy_index]))) * elo_scale,
            "matchups": len(related), "loo_mae": float(np.mean(residuals)),
            "loo_rmse": float(math.sqrt(np.mean(np.square(residuals)))),
            "max_abs_standardized_residual": max(standardized),
            "significant_like_mismatches_abs_z_ge_1_96": sum(value >= 1.96 for value in standardized),
        })

    pair_rows.sort(key=lambda row: abs(float(row["loo_standardized_residual"])), reverse=True)
    policy_rows.sort(key=lambda row: float(row["elo"]), reverse=True)
    _write_csv(output_dir / "elo_pair_diagnostics.csv", pair_rows, list(pair_rows[0]))
    _write_csv(output_dir / "elo_policy_diagnostics.csv", policy_rows, list(policy_rows[0]))
    _plot(output_dir / "elo_diagnostics.png", pair_rows, policy_rows)

    with (output_dir / "elo_diagnostic_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Elo leave-one-pair-out diagnostics\n\n")
        handle.write("Each expected rate is estimated without the displayed matchup. Draws count as half a win.\n\n")
        handle.write("## Largest Elo prediction mismatches\n\n")
        handle.write("| Row policy | Opponent | Observed | LOO Elo | Residual | z | W-L-D |\n|---|---|---:|---:|---:|---:|---|\n")
        for row in pair_rows[:20]:
            handle.write(
                f"| {row['row']} | {row['column']} | {row['observed_win_rate']:.1%} | "
                f"{row['loo_expected_win_rate']:.1%} | {row['residual_observed_minus_expected']:+.1%} | "
                f"{row['loo_standardized_residual']:+.2f} | {row['wins']}-{row['losses']}-{row['draws']} |\n"
            )
        handle.write("\n## Policy rating uncertainty and Elo misfit\n\n")
        handle.write("| Policy | Elo | 95% interval | LOO MAE | Large mismatches |\n|---|---:|---:|---:|---:|\n")
        for row in policy_rows:
            handle.write(
                f"| {row['policy']} | {row['elo']:.1f} | {row['elo_95_low']:.1f}–{row['elo_95_high']:.1f} | "
                f"{row['loo_mae']:.1%} | {row['significant_like_mismatches_abs_z_ge_1_96']} |\n"
            )
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
