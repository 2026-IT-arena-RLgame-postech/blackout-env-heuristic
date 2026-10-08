"""Round-robin win-rate matrix with per-pair adaptive sample size.

Every unordered pair starts with ``--min-seeds`` map seeds, each played from both physical sides.
Whenever a pair has no games in flight its win-rate standard error is re-estimated; pairs above
``--target-se`` get more seeds (projected from SE ~ 1/sqrt(n)) until they reach it or
``--max-seeds``.  Lopsided pairs stop near the minimum, close pairs take the most games.

Games are spread over long-lived workers (one Unity process each), so the pool stays saturated
while pairs finish at different times.  Draws count as half a win.

Early decision: batteries are spawned once and never respawn, and only batteries score.  Once no
battery is left anywhere -- on map tiles (storages included, graphic channel 8) or held by any
unit (agent_states column 4) -- the score is frozen, so the game stops and the higher score wins
(equal scores draw).  Checked on 28 full 420s matches: the score never moved after that point and
Unity's final winner always matched the leader, so results equal full-length play.

Standard error (conservative): max of
  * the paired SE over seeds -- each seed's two side-swapped games averaged into one sample, and
  * a Laplace-smoothed binomial SE sqrt(p(1-p)/n), p = (score + 1) / (n + 2), so a 10-0 start
    does not report SE 0.

Every game is appended to ``games.jsonl``; rerunning with the same ``--output`` resumes.

Example
-------
./.venv/bin/python examples/tournament_adaptive.py --workers 18 --target-se 0.05 \\
    --output reports/heuristic_tournament_adaptive
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

_MPL_CACHE = Path("/private/tmp/blackout_heuristic_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY

from blackout_env.env.constants import team_a_agents, team_b_agents

try:  # direct ``python examples/...`` execution
    import gauntlet_heuristics
except ModuleNotFoundError:
    from examples import gauntlet_heuristics

_init_worker = gauntlet_heuristics._init_worker
BATTERY_CHANNEL = 8     # graphic: battery count per tile, storages included
HELD_BATTERY_COLUMN = 4  # agent_states: holding_item one-hot slot 1 = battery count

DEFAULT_POLICIES = [f"strategic_v{i}" for i in range(1, 20)]


def _play_one(task: tuple[str, str, int, bool]) -> dict:
    """One game; stops as soon as no battery is left anywhere (score can no longer change)."""
    candidate_id, opponent_id, seed, swapped = task
    env = gauntlet_heuristics._ENV
    candidate = POLICY_REGISTRY[candidate_id]()
    opponent = POLICY_REGISTRY[opponent_id]()
    candidate_names = set(team_b_agents() if swapped else team_a_agents())
    candidate_team = 1 if swapped else 0
    obs, _ = env.reset(seed=seed)
    steps, scores, winner, decided_early = 0, (0.0, 0.0), None, False
    while env.agents:
        infos = None
        if not obs:
            obs, _, _, _, infos = env.step({})
        else:
            first = next(iter(obs))
            team_state = obs[first]["team_state"]
            if int(first.split("_")[1]) < 5:
                scores = (float(team_state[0]), float(team_state[1]))
            else:
                scores = (float(team_state[1]), float(team_state[0]))
            if (not obs[first]["graphic"][:, :, BATTERY_CHANNEL].any()
                    and not obs[first]["agent_states"][:, HELD_BATTERY_COLUMN].any()):
                decided_early = True
                break
            actions = {}
            candidate_obs = {n: o for n, o in obs.items() if n in candidate_names}
            opponent_obs = {n: o for n, o in obs.items() if n not in candidate_names}
            if candidate_obs:
                actions.update(candidate.act(candidate_obs))
            if opponent_obs:
                actions.update(opponent.act(opponent_obs))
            obs, _, _, _, infos = env.step(actions)
        steps += 1
        if infos and not env.agents:
            info = next(iter(infos.values()))
            scores = (float(info.get("score_0", scores[0])), float(info.get("score_1", scores[1])))
            physical = info.get("winner")
            winner = None if physical in (None, -1) else 0 if int(physical) == candidate_team else 1
    own, opp = round(scores[candidate_team] * 100), round(scores[1 - candidate_team] * 100)
    if decided_early:
        winner = 0 if own > opp else 1 if opp > own else None
    return {"candidate": candidate_id, "opponent": opponent_id, "seed": seed, "swapped": swapped,
            "winner": winner, "candidate_score": own, "opponent_score": opp, "steps": steps,
            "decided_early": decided_early}


def _score(game: dict) -> float:
    return 1.0 if game["winner"] == 0 else 0.5 if game["winner"] is None else 0.0


def pair_stats(games: list[dict]) -> dict:
    """Win rate of the pair's first policy with its conservative SE, over complete seeds only."""
    by_seed = defaultdict(list)
    for game in games:
        by_seed[game["seed"]].append(_score(game))
    paired = np.array([np.mean(s) for s in by_seed.values() if len(s) == 2])
    m = len(paired)
    if m == 0:
        return {"seeds": 0, "games": 0, "win_rate": float("nan"), "se": float("inf")}
    n = 2 * m
    rate = float(paired.mean())
    paired_se = float(paired.std(ddof=1) / math.sqrt(m)) if m > 1 else float("inf")
    smoothed = (rate * n + 1.0) / (n + 2.0)
    binomial_se = math.sqrt(smoothed * (1.0 - smoothed) / n)
    return {"seeds": m, "games": n, "win_rate": rate,
            "paired_se": paired_se, "binomial_se": binomial_se, "se": max(paired_se, binomial_se)}


def plan_extra_seeds(stats: dict, target_se: float, max_seeds: int) -> int:
    if stats["se"] <= target_se or stats["seeds"] >= max_seeds:
        return 0
    if not math.isfinite(stats["se"]):
        return min(stats["seeds"], max_seeds - stats["seeds"])
    projected = math.ceil(stats["seeds"] * (stats["se"] / target_se) ** 2)
    # At most double per step: early SE estimates from 5 seeds are noisy.
    wanted = min(projected, 2 * stats["seeds"], max_seeds)
    return max(1, wanted - stats["seeds"])


def _label(policy_id: str) -> str:
    return policy_id.removeprefix("strategic_")


def plot(policies: list[str], stats: dict, output: Path, target_se: float) -> None:
    import matplotlib.pyplot as plt

    k = len(policies)
    rate = np.full((k, k), np.nan)
    games = np.zeros((k, k), dtype=int)
    se = np.full((k, k), np.nan)
    for (i, j), s in stats.items():
        rate[i, j], rate[j, i] = s["win_rate"], 1.0 - s["win_rate"]
        games[i, j] = games[j, i] = s["games"]
        se[i, j] = se[j, i] = s["se"]
    labels = [_label(p) for p in policies]
    total_games = int(games[np.triu_indices(k, 1)].sum())

    fig, axis = plt.subplots(figsize=(15.0, 12.5), constrained_layout=True)
    shown = np.where(np.eye(k, dtype=bool), np.nan, rate)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#fffde7")
    image = axis.imshow(shown, vmin=0.0, vmax=1.0, cmap=cmap)
    axis.set_xticks(range(k), labels, rotation=45, ha="right")
    axis.set_yticks(range(k), labels)
    axis.set_xlabel("column policy (opponent)")
    axis.set_ylabel("row policy")
    for i in range(k):
        for j in range(k):
            if i == j:
                axis.text(j, i, "—", ha="center", va="center", color="black", fontsize=9)
                continue
            value = rate[i, j]
            color = "white" if value < 0.25 or value > 0.75 else "black"
            axis.text(j, i - 0.21, f"{value:.0%}", ha="center", va="center",
                      color=color, fontsize=9)
            axis.text(j, i + 0.12, f"n={games[i, j]}", ha="center", va="center",
                      color=color, fontsize=6.2)
            axis.text(j, i + 0.32, f"SE {se[i, j]:.3f}".replace("0.", "."),
                      ha="center", va="center", color=color, fontsize=6.2)
    mean_rate = np.nanmean(shown, axis=1)
    for i in range(k):
        axis.text(k - 0.3, i, f"{mean_rate[i]:.0%}", ha="left", va="center", fontsize=9)
    axis.text(k - 0.3, -0.85, "mean", ha="left", va="center", fontsize=8)
    axis.set_xlim(-0.5, k + 0.7)
    axis.set_title(
        "Heuristic round-robin win rate (row policy wins, draws = 1/2)\n"
        f"side-swapped, stopped once no battery remains; per-pair SE ≤ {target_se:.2f} "
        f"(max {np.nanmax(se):.3f}), {total_games} games",
        fontsize=12,
    )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("side-swapped win rate")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--target-se", type=float, default=0.05)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--max-seeds", type=int, default=60)
    parser.add_argument("--seed-rng", type=int, default=20260918)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unknown = [p for p in args.policies if p not in POLICY_REGISTRY]
    if unknown:
        parser.error(f"unknown policies: {unknown}")
    if args.min_seeds < 1 or args.max_seeds < args.min_seeds:
        parser.error("need 1 <= --min-seeds <= --max-seeds")
    policies = list(dict.fromkeys(args.policies))
    # One shared seed sequence: pair p's k-th seed is the same map for every pair.
    seeds = np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.max_seeds, replace=False).tolist()
    pairs = [(i, j) for i in range(len(policies)) for j in range(i + 1, len(policies))]

    args.output.mkdir(parents=True, exist_ok=True)
    config = {"policies": policies, "seeds": seeds, "target_se": args.target_se,
              "min_seeds": args.min_seeds, "max_seeds": args.max_seeds,
              "time_scale": args.time_scale, "full_length": True, "side_swapped": True}
    config_path, log_path = args.output / "config.json", args.output / "games.jsonl"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if (previous["policies"], previous["seeds"]) != (policies, seeds):
            parser.error("--output holds a run with different policies or seeds")
    config_path.write_text(json.dumps(config, indent=1))

    index = {p: i for i, p in enumerate(policies)}
    games: dict[tuple[int, int], list[dict]] = defaultdict(list)
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line:
                game = json.loads(line)
                games[(index[game["candidate"]], index[game["opponent"]])].append(game)
    done_keys = {(g["candidate"], g["opponent"], g["seed"], g["swapped"])
                 for gs in games.values() for g in gs}
    in_flight: dict[tuple[int, int], int] = defaultdict(int)
    print(f"{len(policies)} policies, {len(pairs)} pairs, resumed games={len(done_keys)}", flush=True)

    with log_path.open("a") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=_init_worker, initargs=(args.build, args.time_scale),
    ) as pool:
        futures = {}

        def submit(pair, seed_indices):
            for k in seed_indices:
                for swapped in (False, True):
                    key = (policies[pair[0]], policies[pair[1]], int(seeds[k]), swapped)
                    if key in done_keys:
                        continue
                    futures[pool.submit(_play_one, key)] = pair
                    in_flight[pair] += 1

        def top_up(pair):
            # Called only with nothing in flight; seeds 0..wanted-1 minus finished games are
            # (re)submitted, which also redoes half-played seeds from an interrupted run.
            stats = pair_stats(games[pair])
            if stats["seeds"] < args.min_seeds:
                wanted = args.min_seeds
            else:
                wanted = stats["seeds"] + plan_extra_seeds(stats, args.target_se, args.max_seeds)
            submit(pair, range(wanted))

        for pair in pairs:
            top_up(pair)
        finished_pairs = sum(in_flight[p] == 0 for p in pairs)
        played = 0
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                pair = futures.pop(future)
                game = future.result()
                log.write(json.dumps(game) + "\n")
                log.flush()
                games[pair].append(game)
                done_keys.add((game["candidate"], game["opponent"], game["seed"], game["swapped"]))
                in_flight[pair] -= 1
                played += 1
                if in_flight[pair] == 0:
                    top_up(pair)
                    if in_flight[pair] == 0:
                        finished_pairs += 1
                        s = pair_stats(games[pair])
                        print(f"[pairs {finished_pairs}/{len(pairs)} | games {played} | "
                              f"queued {len(futures)}] {_label(policies[pair[0]])} vs "
                              f"{_label(policies[pair[1]])}: {s['win_rate']:.1%} "
                              f"SE={s['se']:.3f} n={s['games']}", flush=True)

    stats = {pair: pair_stats(games[pair]) for pair in pairs}
    rows = []
    for (i, j), s in stats.items():
        pair_games = games[(i, j)]
        rows.append({"row": policies[i], "column": policies[j], **s,
                     "wins": sum(g["winner"] == 0 for g in pair_games),
                     "losses": sum(g["winner"] == 1 for g in pair_games),
                     "draws": sum(g["winner"] is None for g in pair_games),
                     "mean_margin": float(np.mean([g["candidate_score"] - g["opponent_score"]
                                                   for g in pair_games]))})
    (args.output / "pairs.json").write_text(json.dumps(rows, indent=1))
    k = len(policies)
    matrix = np.full((k, k), np.nan)
    for (i, j), s in stats.items():
        matrix[i, j], matrix[j, i] = s["win_rate"], 1.0 - s["win_rate"]
    (args.output / "win_rate_matrix.json").write_text(json.dumps(
        {"policies": policies, "win_rate": [[None if np.isnan(v) else v for v in r] for r in matrix]},
        indent=1))
    plot(policies, stats, args.output / "win_rate_heatmap.png", args.target_se)
    capped = [r for r in rows if r["se"] > args.target_se]
    print(f"total games={sum(r['games'] for r in rows)}, pairs above target SE={len(capped)}", flush=True)
    for r in capped:
        print(f"  {r['row']} vs {r['column']}: SE={r['se']:.3f} n={r['games']}", flush=True)
    print(f"output={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
