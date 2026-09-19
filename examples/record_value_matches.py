"""
Record observation snapshots from heuristic matches, for fitting and checking reward v2 against
match outcomes (docs/reward_v2_design.md §5, E3-E5).

Each match is played to a 64 s cutoff (race_gauntlet.py: the first minute decides the match) and
saves team A's observation every --every decisions plus the result, one compressed .npz per match:

  graphic      uint8 [S, 24, 24, 13]  channel 8 as the battery count (0-15), the rest 0/1
  agent_states float32 [S, 10, 12]
  team_state   float32 [S, 4]
  step         int32 [S]             decision index of each snapshot
  meta         json: suite, policy ids/kwargs per team, seed, final displayed scores, winner

Suites
  diverse       policy ids for both teams drawn from the mixture's default weights
  matchups      known results: V17 vs V1-V16 (94%), V18 vs V17 (78%), V19 vs V18 (72%)
  v17_variants  V17 Hunter quota 0-3 (mixed), 3 all hunting, 3 all camping, vs V4/V7/V13
                (findings §3.1: 50 / 66 / 86 / 96, 83-89, 53 %)

Snapshots are what fit_reward_v2.py reads: every --every decisions (default 10 = 0.4 s) up to
--cutoff-steps (default 1600 = 64 s). Unity's own shaping is off (unity_shaping=False); only
observations and the result are kept, so the reward itself is recomputed later.

Refit workflow for reward v2's weights (run from the repo root; how FITTED_20260917B was made):

  0. Unity build at build/mac/BlackOut.app (models/run11_step80k/run11_pipeline.sh build), or pass
     --build to step 1.
  1. Record heuristic matches, one call per suite, all into the same --out root (each suite gets
     its own subdirectory; use a fresh root, index.jsonl is appended to). The 20260917b refit used
     1,000 / 480 / 360 matches, ~9 min on 18 workers:
       ./.venv/bin/python examples/record_value_matches.py --suite diverse      --games 1000 --out reports/value_matches_NEW
       ./.venv/bin/python examples/record_value_matches.py --suite matchups     --games 480  --out reports/value_matches_NEW
       ./.venv/bin/python examples/record_value_matches.py --suite v17_variants --games 360  --out reports/value_matches_NEW
  2. Fit (coordinate search on 4/5 of `diverse`, E3 on the held-out 1/5, E4/E5 on the other suites;
     add --from-fitted to start from FITTED_20260917B instead of the defaults):
       ./.venv/bin/python examples/fit_reward_v2.py --root reports/value_matches_NEW --fit --save reports/value_matches_NEW/fitted.json
     Check the report: fit vs holdout objective gap, values sitting at a grid edge (see the P2
     refit in docs/reward_v2_design.md for how two such values were pulled back by hand).
  3. Unit-level check on a stored dataset (no Unity; without --config it evaluates the DEFAULT
     weights -- Run 11's are docs/design/reward_v2_fitted_20260917b.json):
       ./.venv/bin/python examples/evaluate_reward_v2.py --dataset-dir datasets/heuristic_mixv6_live_20260917 --config reports/value_matches_NEW/fitted.json
  4. Paste the json values into blackout_env/train/reward_v2.py as a new RewardV2Config constant
     (keep FITTED_20260917B for reproducing Run 11), point `--reward v2-fitted` at it in
     blackout_env/train/offline_pretrain.py, and record the numbers in docs/reward_v2_design.md.
     The dataset's reward cache is keyed by the config, so a new config re-annotates on its own.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import util as mp_util
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv
from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics import HeuristicPolicyMixture, make_heuristic

_ENV = None


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True, additional_args=["-logFile", "/dev/null"],
                       unity_shaping=False)
    mp_util.Finalize(_ENV, _ENV.close, exitpriority=10)  # see gauntlet_heuristics.py


def _build(policy_id: str, kwargs: dict, seed: int):
    if policy_id == "strategic_v4_near":
        return make_heuristic(policy_id, seed=seed, **kwargs)
    return make_heuristic(policy_id, **kwargs)


def _pack_graphic(graphic: np.ndarray) -> np.ndarray:
    packed = np.rint(graphic).astype(np.uint8)
    packed[..., 8] = np.rint(graphic[..., 8] * 15.0).astype(np.uint8)
    return packed


def _play(task: dict) -> dict:
    team_a = _build(task["a"][0], task["a"][1], task["seed"])
    team_b = _build(task["b"][0], task["b"][1], task["seed"] + 1_000_003)
    names_a, names_b = set(team_a_agents()), set(team_b_agents())
    first_a = sorted(names_a)[0]
    obs, _ = _ENV.reset(seed=task["seed"])
    graphics, states, teams, steps = [], [], [], []
    scores, winner, ended, step = (0.0, 0.0), None, False, 0
    while _ENV.agents and step < task["cutoff"]:
        infos = {}
        if not obs:
            obs, _, _, _, infos = _ENV.step({})
        else:
            view = obs.get(first_a) or next((obs[n] for n in obs if n in names_a), None)
            if view is not None:
                scores = (float(view["team_state"][0]), float(view["team_state"][1]))
                if step % task["every"] == 0:
                    graphics.append(_pack_graphic(view["graphic"]))
                    states.append(view["agent_states"].astype(np.float32))
                    teams.append(view["team_state"].astype(np.float32))
                    steps.append(step)
            actions = {}
            obs_a = {n: obs[n] for n in obs if n in names_a}
            obs_b = {n: obs[n] for n in obs if n in names_b}
            if obs_a:
                actions.update(team_a.act(obs_a))
            if obs_b:
                actions.update(team_b.act(obs_b))
            obs, _, _, _, infos = _ENV.step(actions)
        step += 1
        if infos and not _ENV.agents:
            info = next(iter(infos.values()))
            ended = True
            physical = info.get("winner")
            winner = None if physical in (None, -1) else int(physical)
            scores = (float(info.get("score_0", scores[0])), float(info.get("score_1", scores[1])))
    if not ended:
        winner = 0 if scores[0] > scores[1] + 0.005 else 1 if scores[1] > scores[0] + 0.005 else None
    meta = {**{k: task[k] for k in ("suite", "label", "a", "b", "seed")},
            "score_a": round(scores[0] * 100), "score_b": round(scores[1] * 100), "winner": winner,
            "ended_early": ended, "steps": step}
    out = Path(task["path"])
    np.savez_compressed(out, graphic=np.stack(graphics), agent_states=np.stack(states), team_state=np.stack(teams),
                        step=np.asarray(steps, dtype=np.int32), meta=json.dumps(meta))
    return meta


def _tasks(suite: str, games: int, rng: np.random.Generator, cutoff: int, every: int, out: Path) -> list[dict]:
    seeds = lambda n: rng.choice(np.arange(1, 2**30, dtype=np.int64), size=n, replace=False).tolist()
    pairs: list[tuple[str, tuple, tuple]] = []
    if suite == "diverse":
        mixture = HeuristicPolicyMixture(seed=0)
        names = list(mixture.weights)
        p = np.asarray([mixture.weights[n] for n in names], dtype=np.float64)
        p /= p.sum()
        for _ in range(games):
            a, b = rng.choice(names, p=p), rng.choice(names, p=p)
            pairs.append((f"{a}-vs-{b}", (str(a), {}), (str(b), {})))
    elif suite == "matchups":
        lineup = [("v17-vs-v1..16", "strategic_v17", f"strategic_v{i}") for i in (1, 4, 7, 10, 13, 16)]
        lineup += [("v18-vs-v17", "strategic_v18", "strategic_v17"), ("v19-vs-v18", "strategic_v19", "strategic_v18")]
        per = max(1, games // (len(lineup) * 2))
        for label, cand, opp in lineup:
            for _ in range(per):
                pairs.append((label, (cand, {}), (opp, {})))
                pairs.append((label + ":swapped", (opp, {}), (cand, {})))
    elif suite == "v17_variants":
        variants = {f"quota{q}-mixed": {"hunter_quota": q, "hunter_mode": "mixed"} for q in (0, 1, 2, 3)}
        variants["quota3-hunt"] = {"hunter_quota": 3, "hunter_mode": "hunt"}
        variants["quota3-camp"] = {"hunter_quota": 3, "hunter_mode": "camp"}
        opponents = ["strategic_v4", "strategic_v7", "strategic_v13"]
        per = max(1, games // (len(variants) * len(opponents) * 2))
        for label, kwargs in variants.items():
            for opp in opponents:
                for _ in range(per):
                    pairs.append((label, ("strategic_v17", kwargs), (opp, {})))
                    pairs.append((label + ":swapped", (opp, {}), ("strategic_v17", kwargs)))
    else:
        raise ValueError(suite)
    tasks = []
    for i, ((label, a, b), seed) in enumerate(zip(pairs, seeds(len(pairs)))):
        tasks.append({"suite": suite, "label": label, "a": list(a), "b": list(b), "seed": int(seed),
                      "cutoff": cutoff, "every": every, "path": str(out / f"match_{i:05d}.npz")})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["diverse", "matchups", "v17_variants"], required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("reports/value_matches"))
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--cutoff-steps", type=int, default=1600)
    parser.add_argument("--every", type=int, default=10, help="decisions between snapshots (25 = 1 s)")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    out = args.out / args.suite
    out.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args.suite, args.games, np.random.default_rng(args.seed), args.cutoff_steps, args.every, out)
    print(f"{len(tasks)} games -> {out}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(args.build, args.time_scale)) as pool, \
            open(out / "index.jsonl", "a") as index:
        for future in as_completed([pool.submit(_play, t) for t in tasks]):
            meta = future.result()
            index.write(json.dumps(meta) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
