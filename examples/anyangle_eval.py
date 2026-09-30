"""Any-angle vs 8-direction movement: paired, side-swapped heuristic matches with movement stats.

Plays ``--candidate`` (default ``strategic_v21_anyangle``) against ``--baseline`` (default
``strategic_v21``) headless, every seed from both physical sides, over a small worker pool, and
records per side:

margin      final score difference (candidate - baseline, x100), mean and SE over games
deliveries  cargo -> none within 1.5 tiles of an own storage tile, per game and per 1000 steps
first       env step of the side's first delivery
tiles/dlv   tiles walked by non-Hunters per delivery
speed eff   distance moved / (speed x step time) over unit-steps with a non-zero command, by
            class (1.0 = no time lost to walls or corners; respawns and class changes skipped)
deaths      own units respawning (a jump of more than 2 tiles between decisions)

It also saves each seed's walls and bases (``maps.npz``) for ``examples/anyangle_probe.py``.

    .venv/bin/python examples/anyangle_eval.py --n-seeds 6 --workers 2 --output reports/anyangle_v21
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
from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.heuristics import POLICY_REGISTRY
from blackout_env.train.reward_v2 import CARGO_COL, CLASS_COLS, STORAGE_OWN, unit_positions

STEP_SECONDS = 0.04  # decisionPeriod 2 x fixedDeltaTime 0.02
SPEED = {0: 4.0, 1: 6.0, 2: 6.0}  # Collector, Hunter, Carrier BaseSpeed (tiles/s)
DELIVERY_RADIUS = 1.5
_ENV = None


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                       additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    mp_util.Finalize(_ENV, _ENV.close, exitpriority=10)


class Side:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.prev: dict[int, tuple] = {}
        self.stats = {"deliveries": 0, "first_delivery": None, "walked_economy": 0.0,
                      "moved": {0: 0.0, 1: 0.0, 2: 0.0}, "expected": {0: 0.0, 1: 0.0, 2: 0.0},
                      "commands": 0, "stand": 0, "deaths": 0}

    def act(self, obs, step: int):
        actions = self.policy.act(obs)
        view = next(iter(obs.values()))
        states, graphic = view["agent_states"], view["graphic"]
        storage = np.argwhere(graphic[..., STORAGE_OWN] > 0.5)
        row, col = unit_positions(states[None])
        for agent, vec in actions.items():
            u = unit_index(agent)
            r, c = float(row[0, u]), float(col[0, u])
            cls = int(states[u, CLASS_COLS].argmax())
            cargo = states[u, CARGO_COL] > 1e-3
            moving = float(np.linalg.norm(vec)) > 1e-6
            self.stats["commands"] += 1
            self.stats["stand"] += int(not moving)
            prev = self.prev.get(u)
            if prev is not None:
                pr, pc, pcls, pcargo, pmoving = prev
                moved = float(np.hypot(r - pr, c - pc))
                if moved > 2.0:
                    self.stats["deaths"] += 1  # respawn teleport
                if moved <= 1.0 and cls == pcls:
                    if pmoving:
                        self.stats["moved"][cls] += moved
                        self.stats["expected"][cls] += SPEED[cls] * STEP_SECONDS
                    if cls != 1:
                        self.stats["walked_economy"] += moved
                    if pcargo and not cargo and len(storage) and \
                            float(np.min(np.hypot(storage[:, 0] - pr, storage[:, 1] - pc))) <= DELIVERY_RADIUS:
                        self.stats["deliveries"] += 1
                        if self.stats["first_delivery"] is None:
                            self.stats["first_delivery"] = step
            self.prev[u] = (r, c, cls, cargo, moving)
        return actions


def _play_one(task):
    candidate_id, baseline_id, seed, swapped, kwargs = task
    env = _ENV
    cand, base = Side(POLICY_REGISTRY[candidate_id](**kwargs)), Side(POLICY_REGISTRY[baseline_id]())
    names_a, names_b = set(team_a_agents()), set(team_b_agents())
    cand_names = names_b if swapped else names_a
    cand_team = 1 if swapped else 0
    obs, _ = env.reset(seed=seed)
    walls = None
    last = (0.0, 0.0)
    step = 0
    final = {}
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            step += 1
            if infos:
                final = next(iter(infos.values()))
            continue
        view = next(iter(obs.values()))
        if walls is None:
            walls = view["graphic"][..., 1] > 0.5
        if unit_index(next(iter(obs))) < 5:
            last = (float(view["team_state"][0]), float(view["team_state"][1]))
        co = {n: obs[n] for n in obs if n in cand_names}
        bo = {n: obs[n] for n in obs if n not in cand_names}
        actions = {}
        if co:
            actions.update(cand.act(co, step))
        if bo:
            actions.update(base.act(bo, step))
        obs, _, _, _, infos = env.step(actions)
        step += 1
        if infos:
            final = next(iter(infos.values()))
            last = (float(final.get("score_0", last[0])), float(final.get("score_1", last[1])))
    winner = final.get("winner")
    winner = None if winner in (-1, None) else (0 if int(winner) == cand_team else 1)
    return {
        "seed": seed, "swapped": swapped, "steps": step, "winner": winner,
        "candidate_score": last[cand_team] * 100, "baseline_score": last[1 - cand_team] * 100,
        "candidate": cand.stats, "baseline": base.stats,
        "anyangle_stats": dict(getattr(cand.policy, "anyangle_stats", {})),
        "walls": walls.astype(np.uint8).tolist() if walls is not None else None,
    }


def _mean_se(values) -> tuple[float, float]:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def summarise(games: list[dict], candidate: str, baseline: str) -> str:
    lines = [f"{candidate} vs {baseline}: {len(games)} games"]
    wins = sum(g["winner"] == 0 for g in games)
    losses = sum(g["winner"] == 1 for g in games)
    m, se = _mean_se([g["candidate_score"] - g["baseline_score"] for g in games])
    lines.append(f"W-L-D {wins}-{losses}-{len(games) - wins - losses}  margin {m:+.1f} +- {se:.1f} (SE)")
    # paired by seed: mean of the two sides
    by_seed: dict[int, list[float]] = {}
    for g in games:
        by_seed.setdefault(g["seed"], []).append(g["candidate_score"] - g["baseline_score"])
    pm, pse = _mean_se([np.mean(v) for v in by_seed.values() if len(v) == 2])
    lines.append(f"seed-paired margin {pm:+.1f} +- {pse:.1f} (SE over {len(by_seed)} seeds)")
    for side in ("candidate", "baseline"):
        name = candidate if side == "candidate" else baseline
        d = [g[side]["deliveries"] for g in games]
        rate = [1000 * g[side]["deliveries"] / max(1, g["steps"]) for g in games]
        first = [g[side]["first_delivery"] for g in games]
        per = [g[side]["walked_economy"] / g[side]["deliveries"] for g in games if g[side]["deliveries"]]
        eff = {}
        for cls, label in ((0, "collector"), (1, "hunter"), (2, "carrier")):
            moved = sum(g[side]["moved"][str(cls)] if str(cls) in g[side]["moved"] else g[side]["moved"][cls] for g in games)
            exp = sum(g[side]["expected"][str(cls)] if str(cls) in g[side]["expected"] else g[side]["expected"][cls] for g in games)
            eff[label] = moved / exp if exp else float("nan")
        deaths = [g[side]["deaths"] for g in games]
        lines.append(
            f"  {name:24s} deliveries/game {_mean_se(d)[0]:.1f}+-{_mean_se(d)[1]:.1f}"
            f"  /1000 steps {_mean_se(rate)[0]:.2f}+-{_mean_se(rate)[1]:.2f}"
            f"  first {_mean_se(first)[0]:.0f}+-{_mean_se(first)[1]:.0f}"
            f"  tiles/dlv {_mean_se(per)[0]:.1f}+-{_mean_se(per)[1]:.1f}"
            f"  deaths {_mean_se(deaths)[0]:.1f}+-{_mean_se(deaths)[1]:.1f}"
            f"  speed eff " + " ".join(f"{k} {v:.3f}" for k, v in eff.items()))
    stats: dict[str, int] = {}
    for g in games:
        for k, v in g["anyangle_stats"].items():
            stats[k] = stats.get(k, 0) + v
    if stats:
        total = sum(stats.values())
        lines.append("  any-angle moves: " + ", ".join(f"{k} {v / total:.1%}" for k, v in sorted(stats.items())))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--candidate", default="strategic_v21_anyangle", choices=sorted(POLICY_REGISTRY))
    parser.add_argument("--baseline", default="strategic_v21", choices=sorted(POLICY_REGISTRY))
    parser.add_argument("--candidate-kwargs", type=json.loads, default={},
                        help='constructor keywords as JSON, e.g. \'{"exact_hunt": false}\'')
    parser.add_argument("--n-seeds", type=int, default=6)
    parser.add_argument("--seed-rng", type=int, default=20260930)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = args.seeds or np.random.default_rng(args.seed_rng).choice(
        np.arange(1, 2**30, dtype=np.int64), size=args.n_seeds, replace=False).tolist()
    tasks = [(args.candidate, args.baseline, int(s), sw, args.candidate_kwargs) for s in seeds for sw in (False, True)]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} games, seeds={seeds}", flush=True)
    games, maps = [], {}
    with (args.output / "games.jsonl").open("w") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=_init_worker, initargs=(args.build, args.time_scale),
    ) as pool:
        for future in as_completed([pool.submit(_play_one, t) for t in tasks]):
            g = future.result()
            walls = g.pop("walls")
            if walls is not None:
                maps[f"seed{g['seed']}_swap{int(g['swapped'])}"] = np.array(walls, dtype=np.uint8)
            games.append(g)
            log.write(json.dumps(g) + "\n")
            log.flush()
            print(f"[{len(games):2d}/{len(tasks)}] seed={g['seed']} swap={int(g['swapped'])} "
                  f"{g['candidate_score']:.0f}-{g['baseline_score']:.0f} steps={g['steps']} "
                  f"dlv {g['candidate']['deliveries']}-{g['baseline']['deliveries']}", flush=True)
    np.savez_compressed(args.output / "maps.npz", **maps)
    text = summarise(games, args.candidate + (json.dumps(args.candidate_kwargs) if args.candidate_kwargs else ""), args.baseline)
    print("\n" + text, flush=True)
    (args.output / "summary.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
