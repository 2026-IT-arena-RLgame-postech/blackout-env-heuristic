"""
Elo for trained checkpoints with as few games as possible, on the heuristic Elo scale.

Heuristic anchors keep their full-pool ratings (train/policy_strength.ELO_20260916, V4 = 1500);
only the checkpoints' ratings are fitted (Bradley-Terry, Gaussian prior). Each round greedily
schedules the games that most reduce the summed variance of the checkpoint ratings -- checkpoint
vs anchor near its current estimate, or checkpoint vs checkpoint when they are close -- and the
run stops once every checkpoint's standard error is below --target-se. Games are race_gauntlet's
64 s truncation, one seed per game pair played from both sides; draws count half.

Usage:
    python examples/elo_checkpoints.py --checkpoints run11=checkpoints/offline/<run>/final.pt \\
        run13=checkpoints/offline/<run>/final.pt --output reports/elo_checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import util as mp_util
from pathlib import Path

import numpy as np

ELO = 400.0 / math.log(10.0)
DEFAULT_ANCHORS = ["strategic_v1", "strategic_v14", "strategic_v2", "strategic_v5", "strategic_v6",
                   "strategic_v3", "strategic_v4", "strategic_v12"]

_ENV = None
_CHECKPOINTS: dict[str, str] = {}
_POLICIES: dict = {}


_DEVICE = "cpu"


def _init_worker(build: str, time_scale: float, checkpoints: dict[str, str], device: str = "cpu") -> None:
    global _ENV, _CHECKPOINTS, _DEVICE
    _DEVICE = device
    import torch

    from blackout_env import BlackOutEnv

    torch.set_num_threads(1)
    _CHECKPOINTS = checkpoints
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True, additional_args=["-logFile", "/dev/null"],
                       unity_shaping=False)
    mp_util.Finalize(_ENV, _ENV.close, exitpriority=10)


def _player(name: str, seed: int):
    from blackout_env.heuristics import make_heuristic

    if name in _CHECKPOINTS:
        if name not in _POLICIES:
            from blackout_env.model.my_policy import MyPolicy
            from blackout_env.train.qmix_trainer import QMIXConfig, QMIXTrainer

            trainer = QMIXTrainer(env=None, config=QMIXConfig(buffer_capacity=1, device=_DEVICE, tb_log_dir=None))
            trainer.load(_CHECKPOINTS[name])
            trainer.net.eval()
            _POLICIES[name] = MyPolicy(trainer.net, device=_DEVICE)
        return _POLICIES[name]
    return make_heuristic(name, seed=seed) if name == "strategic_v4_near" else make_heuristic(name)


def _play(task) -> dict:
    """(a, b, seed, swapped): a plays team B when swapped. Returns winner 0 = a, 1 = b, None = draw."""
    from blackout_env.env.constants import team_a_agents, team_b_agents

    a, b, seed, swapped, cutoff = task
    pa, pb = _player(a, seed), _player(b, seed + 1_000_003)
    for p in (pa, pb):
        if hasattr(p, "reset"):
            p.reset()
    a_names = set(team_b_agents() if swapped else team_a_agents())
    a_team = 1 if swapped else 0
    obs, _ = _ENV.reset(seed=seed)
    steps, scores, winner, ended = 0, (0.0, 0.0), None, False
    while _ENV.agents and steps < cutoff:
        infos = {}
        if not obs:
            obs, _, _, _, infos = _ENV.step({})
        else:
            first = next(iter(obs))
            ts = obs[first]["team_state"]
            scores = (float(ts[0]), float(ts[1])) if int(first.split("_")[1]) < 5 else (float(ts[1]), float(ts[0]))
            oa = {n: obs[n] for n in obs if n in a_names}
            ob = {n: obs[n] for n in obs if n not in a_names}
            actions = {}
            if oa:
                actions.update(pa.act(oa))
            if ob:
                actions.update(pb.act(ob))
            obs, _, _, _, infos = _ENV.step(actions)
        steps += 1
        if infos and not _ENV.agents:
            info = next(iter(infos.values()))
            ended = True
            physical = info.get("winner")
            if physical not in (None, -1):
                winner = 0 if int(physical) == a_team else 1
            scores = (float(info.get("score_0", scores[0])), float(info.get("score_1", scores[1])))
    own, opp = scores[a_team] * 100, scores[1 - a_team] * 100
    if not ended:
        winner = 0 if own > opp + 0.5 else 1 if opp > own + 0.5 else None
    return {"a": a, "b": b, "seed": seed, "swapped": swapped, "winner": winner, "a_score": round(own),
            "b_score": round(opp), "steps": steps, "ended": ended}


def fit(free: list[str], fixed: dict[str, float], games: list[dict], prior_mean: float, prior_sd: float):
    """Bradley-Terry over the free players with anchors held at `fixed` (Elo). Returns (elo, cov)."""
    k = len(free)
    idx = {p: i for i, p in enumerate(free)}
    r = np.full(k, prior_mean / ELO)
    lam = 1.0 / (prior_sd / ELO) ** 2
    for _ in range(100):
        grad = -lam * (r - prior_mean / ELO)
        hess = -lam * np.eye(k)
        for g in games:
            s = 1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0
            ra = r[idx[g["a"]]] if g["a"] in idx else fixed[g["a"]] / ELO
            rb = r[idx[g["b"]]] if g["b"] in idx else fixed[g["b"]] / ELO
            p = 1.0 / (1.0 + math.exp(-(ra - rb)))
            w = p * (1.0 - p)
            for name, sign in ((g["a"], 1.0), (g["b"], -1.0)):
                if name in idx:
                    grad[idx[name]] += sign * (s - p)
            ia, ib = idx.get(g["a"]), idx.get(g["b"])
            for i in (ia, ib):
                if i is not None:
                    hess[i, i] -= w
            if ia is not None and ib is not None:
                hess[ia, ib] += w
                hess[ib, ia] += w
        step = np.linalg.solve(hess, -grad)
        r += step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(-hess) * ELO ** 2
    return r * ELO, cov


def floored(free, fixed, games, min_games=6):
    """Checkpoints that have scored nothing in >= min_games against the weakest anchor: their absolute
    rating is only bounded by the prior, so more anchor games buy nothing."""
    anchors = {a: v for a, v in fixed.items() if a.startswith("strategic_")}
    if not anchors:
        return set()
    weakest = min(anchors, key=anchors.get)
    out = set()
    for n in free:
        rows = [g for g in games if {g["a"], g["b"]} == {n, weakest}]
        won = sum((g["winner"] == 0) if g["a"] == n else (g["winner"] == 1) for g in rows)
        draws = sum(g["winner"] is None for g in rows)
        if len(rows) >= min_games and won + draws == 0:
            out.add(n)
    return out


def choose(free, fixed, elo, cov, per_round, games_per_pair=2, skip_anchors=frozenset()):
    """Greedy A-optimal picks over (free, anchor) and (free, free) pairs (Sherman-Morrison).
    Checkpoints in skip_anchors only get checkpoint-vs-checkpoint games."""
    k = len(free)
    cov = cov / ELO ** 2
    r = elo / ELO
    picks = []
    for _ in range(per_round):
        best = None
        for i in range(k):
            if free[i] in skip_anchors and k == 1:
                continue
            for opp in ([] if free[i] in skip_anchors else list(fixed)) + free[i + 1:]:
                e = np.zeros(k)
                e[i] = 1.0
                ro = fixed[opp] / ELO if opp in fixed else r[free.index(opp)]
                if opp not in fixed:
                    e[free.index(opp)] = -1.0
                p = 1.0 / (1.0 + math.exp(-(r[i] - ro)))
                w = games_per_pair * p * (1.0 - p)
                ce = cov @ e
                gain = w * float(ce @ ce) / (1.0 + w * float(e @ ce))
                if best is None or gain > best[0]:
                    best = (gain, free[i], opp, w, ce, e)
        if best is None:
            break
        _, a, b, w, ce, e = best
        cov = cov - (w / (1.0 + w * float(e @ ce))) * np.outer(ce, ce)
        picks.append((a, b))
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path")
    parser.add_argument("--anchors", nargs="*", default=DEFAULT_ANCHORS, help="heuristic anchors; pass none for checkpoints only")
    parser.add_argument("--reference", help="checkpoint label pinned at --reference-elo (relative rating among checkpoints)")
    parser.add_argument("--reference-elo", type=float, default=0.0)
    parser.add_argument("--target-se", type=float, default=50.0)
    parser.add_argument("--max-games", type=int, default=400)
    parser.add_argument("--pairs-per-round", type=int, default=9, help="each pair = 2 games (both sides)")
    parser.add_argument("--prior-mean", type=float, default=1300.0)
    parser.add_argument("--prior-sd", type=float, default=400.0)
    parser.add_argument("--cutoff-steps", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--build", default="build/mac/BlackOut.app")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, default=Path("reports/elo_checkpoints"))
    parser.add_argument("--device", default="cpu", help="inference device per worker; batch-1 CPU (1 thread) measured "
                        "1.9 ms/act vs MPS 3.8 ms/act, so cpu is the default")
    parser.add_argument("--resume", action="store_true", help="start from the games already in <output>/games.jsonl")
    args = parser.parse_args()

    from blackout_env.train.policy_strength import ELO_20260916

    checkpoints = dict(spec.split("=", 1) for spec in args.checkpoints)
    free = [c for c in checkpoints if c != args.reference]
    fixed = {a: float(ELO_20260916[a]) for a in args.anchors}
    if args.reference:
        fixed[args.reference] = args.reference_elo
        args.prior_mean = args.reference_elo
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "games.jsonl"
    games: list[dict] = []
    if args.resume and log_path.exists():
        players = set(free) | set(fixed)
        games = [g for g in map(json.loads, log_path.read_text().splitlines()) if g["a"] in players and g["b"] in players]
        print(f"resumed {len(games)} games from {log_path}", flush=True)
    log = open(log_path, "a")
    elo, cov = fit(free, fixed, games, args.prior_mean, args.prior_sd)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             initargs=(args.build, args.time_scale, checkpoints, args.device)) as pool:
        round_no = 0
        while len(games) < args.max_games:
            se = np.sqrt(np.diag(cov))
            low = floored(free, fixed, games)
            if len(games) and np.all(se < args.target_se):
                break
            if low and low == set(free):
                # every checkpoint is below the anchor scale: only their differences are measurable
                diff_se = [math.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j]) for i in range(len(free)) for j in range(i + 1, len(free))]
                if not diff_se or max(diff_se) < args.target_se:
                    break
            pairs = choose(free, fixed, elo, cov, args.pairs_per_round, skip_anchors=frozenset(low))
            if not pairs:
                break
            tasks = []
            for a, b in pairs:
                seed = int(rng.integers(1, 2**30))
                tasks += [(a, b, seed, False, args.cutoff_steps), (a, b, seed, True, args.cutoff_steps)]
            for fut in as_completed([pool.submit(_play, t) for t in tasks]):
                g = fut.result()
                games.append(g)
                log.write(json.dumps(g) + "\n")
            log.flush()
            elo, cov = fit(free, fixed, games, args.prior_mean, args.prior_sd)
            round_no += 1
            se = np.sqrt(np.diag(cov))
            print(f"round {round_no}: {len(games)} games | " + ", ".join(f"{n} {e:.0f}±{s:.0f}" for n, e, s in zip(free, elo, se)), flush=True)

    se = np.sqrt(np.diag(cov))
    low = floored(free, fixed, games)
    if low:
        print(f"\nbelow the weakest anchor with no points scored: {sorted(low)} -- "
              "their absolute Elo is set by the prior; read only the differences")
    print(f"\nfinal after {len(games)} games (fixed: " + ", ".join(f"{a.replace('strategic_', '')} {v:.0f}" for a, v in fixed.items()) + ")")
    for n, e, s in sorted(zip(free, elo, se), key=lambda x: -x[1]):
        print(f"  {n:12s} Elo {e:6.0f} ± {s:3.0f}")
    if len(free) > 1:
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                d = elo[i] - elo[j]
                sd = math.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])
                print(f"  {free[i]} - {free[j]}: {d:+.0f} ± {sd:.0f} Elo")
    print("\nrecord (score for the checkpoint, draws half):")
    for n in free:
        for opp in list(fixed) + [f for f in free if f != n]:
            rows = [g for g in games if {g["a"], g["b"]} == {n, opp}]
            if rows:
                s = sum((1.0 if g["winner"] == 0 else 0.5 if g["winner"] is None else 0.0) if g["a"] == n
                        else (1.0 if g["winner"] == 1 else 0.5 if g["winner"] is None else 0.0) for g in rows)
                print(f"  {n:10s} vs {opp.replace('strategic_', ''):10s} {s:4.1f}/{len(rows)}")
    (args.output / "ratings.json").write_text(json.dumps(
        {"games": len(games), "anchors": fixed, "ratings": {n: {"elo": float(e), "se": float(s)} for n, e, s in zip(free, elo, se)}},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
