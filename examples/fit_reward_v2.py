"""
Check and calibrate reward v2's value against recorded heuristic match outcomes (E3-E5).

For team A at time t:  X(t) = (confirmed_A - confirmed_B) + (V_A - V_B)   (points)
A potential is a good shaping target when it anticipates the result, so:

E3  across the `diverse` suite, how well X(t) at t = 2..40 s predicts the 64 s score difference
    (Pearson r) and the winner (AUC), against the displayed score difference and the old team
    potential Psi, and whether V adds anything once the displayed score is known (partial r)
E4  `matchups`: the candidate's mean X(t) should be positive exactly where it wins
E5  `v17_variants`: the ordering of mean X(t) should follow the ordering of win rates

--fit runs a coordinate search over RewardV2Config weights, maximising the mean E3 correlation over
5-30 s on the diverse suite, and reports E3-E5 for the fitted config. Every --holdout-every-th diverse
match is left out of the search and E3 is reported on it separately, so overfitting shows as a gap.

Usage:
    python examples/fit_reward_v2.py --root reports/value_matches [--fit]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from blackout_env.train.reward_v2 import (
    BATTERY, INF, STORAGE_ENEMY, STORAGE_OWN, TARGET_SCORE, RewardV2Config, compute_potentials,
    geometry_for, interpolated_rows, unit_positions,
)

SECONDS = (2, 5, 10, 15, 20, 30, 40)
DECISIONS_PER_SECOND = 25


def load_suite(root: Path, suite: str) -> list[dict]:
    matches = []
    for path in sorted((root / suite).glob("match_*.npz")):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        step = z["step"]
        picks = [int(np.argmin(np.abs(step - s * DECISIONS_PER_SECOND))) for s in SECONDS]
        graphic = z["graphic"][picks].astype(np.float32)
        graphic[..., 8] /= 15.0
        matches.append({"meta": meta, "graphic": graphic, "agent_states": z["agent_states"][picks],
                        "team_state": z["team_state"][picks], "valid": step[picks] <= np.array(SECONDS) * DECISIONS_PER_SECOND + 10})
    return matches


def stack(matches: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.concatenate([m["graphic"] for m in matches])
    s = np.concatenate([m["agent_states"] for m in matches])
    t = np.concatenate([m["team_state"] for m in matches])
    return g, s, t


def value_x(matches: list[dict], cfg: RewardV2Config) -> np.ndarray:
    """[M, len(SECONDS)] X(t) for team A."""
    g, s, t = stack(matches)
    out = []
    for i in range(0, len(g), 2048):
        p = compute_potentials(g[i : i + 2048], s[i : i + 2048], t[i : i + 2048], cfg)
        v = p.team_value()
        out.append(p.confirmed[:, 0] - p.confirmed[:, 1] + v[:, 0] - v[:, 1])
    return np.concatenate(out).reshape(len(matches), len(SECONDS))


def old_psi(matches: list[dict]) -> np.ndarray:
    """The Unity team potential this replaces: tanh((displayed diff + provisional)/40), provisional =
    carried and stored batteries * exp(-0.05 * tau / nearest enemy of ANY class). Note it adds stored
    batteries on top of a displayed score that already counts them."""
    g, s, t = stack(matches)
    geo = geometry_for(g[0])
    own = s[..., 2] > 0
    row, col = unit_positions(s)
    rows = np.where(own[..., None], interpolated_rows(geo.dist[0], geo.walkable[0], row, col),
                    interpolated_rows(geo.dist[1], geo.walkable[1], row, col))
    tau = 20.0 * t[:, 3]
    battery = np.rint(g[..., BATTERY] * 15).reshape(len(g), -1)
    cargo = np.rint(np.maximum(s[..., 4], 0) * 15)
    x = (t[:, 0] - t[:, 1]) * TARGET_SCORE
    for team, sign in ((0, 1.0), (1, -1.0)):
        enemies = own if team == 1 else ~own
        d_enemy_to_cell = np.where(enemies[..., None], rows, INF).min(1)  # [B, cells]
        storage = (g[..., STORAGE_OWN if team == 0 else STORAGE_ENEMY] > 0.5).reshape(len(g), -1)
        keep = np.exp(-0.05 * tau[:, None] / np.maximum(d_enemy_to_cell, 0.5))
        stored = np.where(storage, battery * keep, 0).sum(1)
        cells = (np.rint(row) * 24 + np.rint(col)).astype(int)
        mine = own if team == 0 else ~own
        carried_keep = np.take_along_axis(keep, cells, 1)
        carried = np.where(mine, cargo * carried_keep, 0).sum(1)
        x = x + sign * (stored + carried)
    return np.tanh(x / 40.0).reshape(len(matches), len(SECONDS))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def e3(matches: list[dict], x: np.ndarray, label: str) -> dict[str, list[float]]:
    y = np.array([m["meta"]["score_a"] - m["meta"]["score_b"] for m in matches], dtype=np.float64)
    win = np.array([m["meta"]["winner"] for m in matches], dtype=object)
    decided = np.array([w in (0, 1) for w in win])
    a_won = np.array([w == 0 for w in win])
    displayed = np.stack([(m["team_state"][:, 0] - m["team_state"][:, 1]) * TARGET_SCORE for m in matches])
    rows = {"r": [], "auc": [], "partial_r": []}
    for k in range(len(SECONDS)):
        ok = np.array([m["valid"][k] for m in matches])
        xs, ys = x[ok, k], y[ok]
        rows["r"].append(float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 else float("nan"))
        rows["auc"].append(auc(xs[decided[ok]], a_won[ok][decided[ok]]))
        d = displayed[ok, k]
        if d.std() > 0 and xs.std() > 0:
            beta = np.polyfit(d, ys, 1)
            resid_y = ys - np.polyval(beta, d)
            resid_x = xs - np.polyval(np.polyfit(d, xs, 1), d)
            rows["partial_r"].append(float(np.corrcoef(resid_x, resid_y)[0, 1]) if resid_x.std() > 1e-9 else 0.0)
        else:
            rows["partial_r"].append(float("nan"))
    return rows


def print_e3(results: dict[str, dict[str, list[float]]]) -> None:
    print("\nE3 predicting the 64 s score difference (r) / winner (AUC) from the state at t")
    header = "".join(f"{s:>7d}s" for s in SECONDS)
    for metric in ("r", "auc", "partial_r"):
        print(f"  {metric:9s}{header}")
        for name, rows in results.items():
            print(f"    {name:22s}" + "".join(f"{v:8.3f}" for v in rows[metric]))


def candidate_view(matches: list[dict], x: np.ndarray) -> dict[str, dict[str, list]]:
    """Per label (':swapped' folded in): candidate's X(t) and whether it won."""
    groups = defaultdict(lambda: {"x": [], "won": []})
    for m, xs in zip(matches, x):
        label = m["meta"]["label"]
        swapped = label.endswith(":swapped")
        base = label.removesuffix(":swapped")
        sign = -1.0 if swapped else 1.0
        cand_team = 1 if swapped else 0
        groups[base]["x"].append(sign * xs)
        w = m["meta"]["winner"]
        groups[base]["won"].append(0.5 if w not in (0, 1) else float(w == cand_team))
    return groups


def print_candidates(title: str, matches: list[dict], x: np.ndarray, expected: dict[str, float] | None = None) -> None:
    groups = candidate_view(matches, x)
    print(f"\n{title}: candidate mean X(t) (points) and win rate")
    print(f"  {'label':18s} {'games':>5s} {'win':>5s} {'exp':>5s}" + "".join(f"{s:>7d}s" for s in SECONDS))
    order = []
    for label, gdict in sorted(groups.items()):
        xs = np.stack(gdict["x"])
        win = float(np.mean(gdict["won"]))
        exp = f"{expected[label]:5.0%}" if expected and label in expected else "    -"
        print(f"  {label:18s} {len(xs):5d} {win:5.0%} {exp}" + "".join(f"{v:8.1f}" for v in xs.mean(0)))
        order.append((label, win, xs.mean(0)))
    if len(order) >= 3:
        wins = np.array([o[1] for o in order])
        for k, s in enumerate(SECONDS):
            vals = np.array([o[2][k] for o in order])
            rho = np.corrcoef(np.argsort(np.argsort(wins)), np.argsort(np.argsort(vals)))[0, 1]
            if s in (10, 20, 30):
                print(f"  rank correlation (win rate vs X at {s}s): {rho:+.2f}")


def objective(matches: list[dict], cfg: RewardV2Config) -> float:
    rows = e3(matches, value_x(matches, cfg), "")
    return float(np.nanmean([rows["r"][k] for k, s in enumerate(SECONDS) if 5 <= s <= 30]))


def coordinate_search(matches: list[dict], cfg: RewardV2Config, sweeps: int = 1) -> RewardV2Config:
    grid = {
        "steal_hazard": [0.05, 0.14, 0.3, 0.6],
        "carry_hazard": [0.05, 0.14, 0.3, 0.6],
        "lambda_rho": [0.2, 0.5, 0.8, 0.95],
        "lambda_length": [6.0, 12.0, 24.0],
        "hunt_beta": [0.2, 0.5, 0.8, 0.95],
        "hunt_length": [2.0, 4.0, 8.0],
        "exit_value": [0.0, 15.0, 30.0, 45.0, 60.0],
        "exit_length": [2.0, 3.0, 5.0],
        "travel_value": [0.0, 0.5, 1.5, 3.0],
        "travel_cap_seconds": [5.0, 10.0, 20.0],
        "hunter_value": [0.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        "carrier_value": [0.0, 3.0, 10.0],
    }
    best = objective(matches, cfg)
    print(f"\nfit: start objective {best:.4f}")
    for sweep in range(sweeps):
        for name, values in grid.items():
            for v in values:
                if name == "hunter_value":
                    trial = replace(cfg, class_value=(cfg.class_value[0], v, cfg.class_value[2]))
                elif name == "carrier_value":
                    trial = replace(cfg, class_value=(cfg.class_value[0], cfg.class_value[1], v))
                else:
                    trial = replace(cfg, **{name: v})
                score = objective(matches, trial)
                if score > best + 1e-4:
                    best, cfg = score, trial
                    print(f"  sweep {sweep} {name}={v}: {score:.4f}", flush=True)
    print(f"fit: best objective {best:.4f}\n{json.dumps(asdict(cfg))}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("reports/value_matches"))
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--save", type=Path, help="write the fitted RewardV2Config as json")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--from-fitted", action="store_true", help="start the search from reward_v2.FITTED_20260917B")
    parser.add_argument("--holdout-every", type=int, default=5, help="hold out every n-th diverse match (0: none)")
    args = parser.parse_args()

    diverse = load_suite(args.root, "diverse")
    matchups = load_suite(args.root, "matchups")
    variants = load_suite(args.root, "v17_variants")
    print(f"matches: diverse {len(diverse)}, matchups {len(matchups)}, v17_variants {len(variants)}")
    k = args.holdout_every
    holdout = [m for i, m in enumerate(diverse) if k and i % k == 0]
    fit_set = [m for i, m in enumerate(diverse) if not (k and i % k == 0)]
    print(f"diverse split: fit {len(fit_set)}, holdout {len(holdout)}")

    configs = {"v2 default": RewardV2Config()}
    if args.fit:
        from blackout_env.train.reward_v2 import FITTED_20260917B

        configs["v2 fitted (previous)"] = FITTED_20260917B
        configs["v2 fitted"] = coordinate_search(fit_set, FITTED_20260917B if args.from_fitted else RewardV2Config(), sweeps=args.sweeps)
        if args.save:
            args.save.write_text(json.dumps(asdict(configs["v2 fitted"]), indent=2) + "\n")

    for split, ms in (("fit set", fit_set), ("holdout", holdout)):
        if not ms:
            continue
        displayed = np.stack([(m["team_state"][:, 0] - m["team_state"][:, 1]) * TARGET_SCORE for m in ms])
        results = {"displayed score diff": e3(ms, displayed, ""), "old Psi": e3(ms, old_psi(ms), "")}
        for name, cfg in configs.items():
            results[name] = e3(ms, value_x(ms, cfg), name)
        print(f"\n----- diverse {split} ({len(ms)} matches) -----")
        print_e3(results)
        print("  objective (mean r 5-30 s): " + ", ".join(f"{n} {objective(ms, c):.4f}" for n, c in configs.items()))

    expected = {"v17-vs-v1..16": 0.94, "v18-vs-v17": 0.78, "v19-vs-v18": 0.72}
    expected_variants = {"quota0-mixed": 0.50, "quota1-mixed": 0.66, "quota2-mixed": 0.86, "quota3-mixed": 0.96, "quota3-hunt": 0.86, "quota3-camp": 0.53}
    for name, cfg in configs.items():
        print(f"\n===== {name} =====")
        print_candidates("E4 matchups", matchups, value_x(matchups, cfg), expected)
        print_candidates("E5 V17 variants (findings §3.1 win rates vs V1-V16)", variants, value_x(variants, cfg), expected_variants)
    print("\n===== displayed score diff (reference) =====")
    for title, ms in (("E4 matchups", matchups), ("E5 V17 variants", variants)):
        print_candidates(title, ms, np.stack([(m["team_state"][:, 0] - m["team_state"][:, 1]) * TARGET_SCORE for m in ms]))


if __name__ == "__main__":
    main()
