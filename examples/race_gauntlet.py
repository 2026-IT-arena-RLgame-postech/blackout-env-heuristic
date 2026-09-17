"""Fast truncated gauntlet: candidate variants vs opponents, scored at an early cutoff.

A BlackOut match is decided in its first minute: the ~200 battery points are spawned once and
never respawn, so the field is empty within ~40s and scores barely move after the third
absorption (60s).  Playing only the first ``--cutoff-steps`` (default 1600 = 64s) is therefore a
close, ~6x cheaper proxy for the full 420s result.  A match that ends early (100 points) keeps
Unity's real winner; otherwise the live scores at the cutoff decide.

Variants are ``name=policy_id`` or ``name=policy_id:{json kwargs}``.

Example
-------
./.venv/bin/python examples/race_gauntlet.py --workers 18 --n-seeds 4 \\
    --variants 'base=strategic_v17' 'risk=strategic_v17:{"threat_mode": "risk"}'
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import util as mp_util
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import make_heuristic

_ENV = None


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                       additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    # See gauntlet_heuristics.py: without an explicit close the worker never exits.
    mp_util.Finalize(_ENV, _ENV.close, exitpriority=10)


def _build(spec: tuple[str, dict], seed: int):
    policy_id, kwargs = spec
    if policy_id == "strategic_v4_near":
        return make_heuristic(policy_id, seed=seed, **kwargs)
    return make_heuristic(policy_id, **kwargs)


def _play(task) -> dict:
    variant, spec, opponent, seed, swapped, cutoff = task
    candidate = _build(spec, seed)
    baseline = _build((opponent, {}), seed + 1_000_003)
    cand_names = set(team_b_agents() if swapped else team_a_agents())
    base_names = set(team_a_agents() if swapped else team_b_agents())
    cand_team = 1 if swapped else 0
    obs, _ = _ENV.reset(seed=seed)
    steps = 0
    scores = (0.0, 0.0)
    winner = None
    ended = False
    while _ENV.agents and steps < cutoff:
        if not obs:
            obs, _, _, _, infos = _ENV.step({})
            steps += 1
        else:
            first = next(iter(obs))
            team_state = obs[first]["team_state"]
            if int(first.split("_")[1]) < 5:
                scores = (float(team_state[0]), float(team_state[1]))
            else:
                scores = (float(team_state[1]), float(team_state[0]))
            actions = {}
            cand_obs = {n: obs[n] for n in obs if n in cand_names}
            base_obs = {n: obs[n] for n in obs if n in base_names}
            if cand_obs:
                actions.update(candidate.act(cand_obs))
            if base_obs:
                actions.update(baseline.act(base_obs))
            obs, _, _, _, infos = _ENV.step(actions)
            steps += 1
        if infos and not _ENV.agents:
            info = next(iter(infos.values()))
            ended = True
            physical = info.get("winner")
            if physical not in (None, -1):
                winner = 0 if int(physical) == cand_team else 1
            scores = (float(info.get("score_0", scores[0])), float(info.get("score_1", scores[1])))
    own, opp = scores[cand_team] * 100, scores[1 - cand_team] * 100
    if not ended:
        winner = 0 if own > opp + 0.5 else 1 if opp > own + 0.5 else None
    return {"variant": variant, "opponent": opponent, "seed": seed, "swapped": swapped,
            "winner": winner, "own": round(own), "opp": round(opp), "steps": steps, "ended": ended}


def _parse_variant(text: str) -> tuple[str, tuple[str, dict]]:
    name, _, rest = text.partition("=")
    policy_id, _, raw = rest.partition(":")
    return name, (policy_id, json.loads(raw) if raw else {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--opponents", nargs="+", default=[f"strategic_v{i}" for i in range(1, 17)])
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--seed-rng", type=int, default=20260916)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--cutoff-steps", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    variants = dict(_parse_variant(v) for v in args.variants)
    seeds = args.seeds or np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.n_seeds, replace=False).tolist()
    tasks = [(name, spec, opponent, int(seed), swapped, args.cutoff_steps)
             for name, spec in variants.items() for opponent in args.opponents
             for seed in seeds for swapped in (False, True)]
    print(f"{len(tasks)} games, seeds={seeds}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker,
                             initargs=(args.build, args.time_scale)) as pool:
        futures = [pool.submit(_play, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    table = defaultdict(lambda: defaultdict(list))
    for r in results:
        table[r["variant"]][r["opponent"]].append(r)
    short = [o.replace("strategic_", "") for o in args.opponents]
    print("\nwin rate (draw=0.5) per opponent, then total win%, mean margin, worst opponent")
    print(f"{'variant':18s} " + " ".join(f"{s:>4s}" for s in short) + "   total  margin  worst")
    for name in variants:
        rates, margins = [], []
        cells = []
        for opponent in args.opponents:
            games = table[name][opponent]
            rate = np.mean([1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0 for g in games])
            rates.append(rate)
            margins.extend(g["own"] - g["opp"] for g in games)
            cells.append(f"{rate * 100:4.0f}")
        worst = int(np.argmin(rates))
        print(f"{name:18s} " + " ".join(cells) +
              f"   {np.mean(rates) * 100:5.1f}  {np.mean(margins):+6.1f}  {short[worst]}={rates[worst] * 100:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
