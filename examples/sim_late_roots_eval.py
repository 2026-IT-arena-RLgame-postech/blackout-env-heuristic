"""Late-game paired comparison in the simulator: same root states, learner variants continue.

Full matches (sim_teacher_eval.py) are chaotic: any change before the race is decided moves
the whole match, and a +-100 point swing on one seed buries a rule that only acts late.  Here
the opening is fixed: ``strategic_v23`` (the root learner) plays each opponent from each seed,
both sides, and at ``--root-seconds`` (if the match is still on) the state is snapshotted --
the simulator and a deep copy of the opponent (history-dependent heuristics keep their
history).  From every root each ``--learners`` variant then continues against its copy of the
opponent for ``--horizon`` seconds (or to the end); the reseeded special-item RNG is the same
for every variant (common random numbers).  The result is the lead at the end of the
continuation; ``report`` gives each variant's paired gap against ``--baseline`` per opponent
and root time.

    PYTHONPATH=. .venv/bin/python examples/sim_late_roots_eval.py play --learners strategic_v23 strategic_v24 \\
        --opponents strategic_v23 strategic_v22 --seeds 24 --workers 6 --out reports/teacher_v24/roots
    PYTHONPATH=. .venv/bin/python examples/sim_late_roots_eval.py report --out reports/teacher_v24/roots
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DPS = 25
MATCH_SECONDS = 420
ROOT_LEARNER = "strategic_v23"


def team_names(team: int) -> list[str]:
    return [f"unit_{i}" for i in range(5 * team, 5 * team + 5)]


def _moves(obs, sides) -> np.ndarray:
    mv = np.zeros((10, 2), np.float32)
    for team, policy in sides.items():
        for name, a in policy.act({n: obs[n] for n in team_names(team)}).items():
            mv[int(name.split("_")[1])] = np.clip(np.asarray(a, np.float32), -1, 1)
    return mv


def run_task(task: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import SimState
    from blackout_env.train.distill.opponents import make_opponent

    side, seed = task["side"], task["seed"]
    sim = SimState.from_seed(seed)
    learner = make_opponent(ROOT_LEARNER, seed)
    opponent = make_opponent(task["opponent"], seed + 7919)
    root_ticks = sorted(int(s * DPS) for s in task["root_seconds"])
    roots = []
    tick = 0
    while not sim.over and tick <= root_ticks[-1]:
        if tick in root_ticks:
            roots.append((tick, sim.copy(), copy.deepcopy(opponent)))
        sim.step(_moves(sim.obs(), {side: learner, 1 - side: opponent}))
        tick += 1
    out = []
    for tick, sim0, opp0 in roots:
        reseed = zlib.crc32(f"{seed}:{side}:{task['opponent']}:{tick}".encode()) % 1_000_000_007
        lead0 = sim0.scores[side] - sim0.scores[1 - side]
        for desc in task["learners"]:
            started = time.time()
            s = sim0.copy()
            s.reseed(reseed)
            pol, opp = make_opponent(desc, seed), copy.deepcopy(opp0)
            limit = min(MATCH_SECONDS * DPS, tick + int(task["horizon"] * DPS)) - tick
            steps = 0
            while not s.over and steps < limit:
                s.step(_moves(s.obs(), {side: pol, 1 - side: opp}))
                steps += 1
            sc = s.scores
            out.append({"learner": desc, "opponent": task["opponent"], "seed": seed, "side": side,
                        "root_s": tick / DPS, "lead_root": lead0, "lead": sc[side] - sc[1 - side],
                        "over": s.over, "steps": steps, "seconds": round(time.time() - started, 2)})
    return out


def key_of(row: dict) -> tuple:
    return row["learner"], row["opponent"], row["seed"], row["side"], row["root_s"]


def load_rows(out: Path) -> list[dict]:
    path = out / "roots.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_play(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    tasks = [{"opponent": o, "seed": args.seed_base + j, "side": side, "root_seconds": args.root_seconds,
              "horizon": args.horizon, "learners": args.learners}
             for o in args.opponents for j in range(args.seeds) for side in (0, 1)
             if (o, args.seed_base + j, side) not in done]
    print(f"{len(tasks)} root matches to play", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as ex, (out / "roots.jsonl").open("a") as fh:
        futures = [ex.submit(run_task, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}  {time.time() - started:.0f}s", flush=True)


def se(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def report(rows: list[dict], baseline: str) -> str:
    by = defaultdict(dict)
    for r in rows:
        by[r["learner"]][key_of(r)[1:]] = r
    base = by[baseline]
    lines = [f"roots: {len(base)} (baseline {baseline}); gap = lead at the end of the continuation, variant - baseline"]
    for learner in sorted(by, key=lambda n: (n != baseline, n)):
        if learner == baseline:
            continue
        lines.append(f"== {learner}")
        rows_l = by[learner]
        groups = defaultdict(list)
        for k, r in rows_l.items():
            if k in base:
                g = r["lead"] - base[k]["lead"]
                groups[("opp", k[0])].append(g)
                groups[("root", k[3])].append(g)
                groups[("all", "")].append(g)
        for gk in sorted(groups, key=lambda x: (x[0] != "opp", x[0] != "root", str(x[1]))):
            g = groups[gk]
            label = f"{gk[0]} {gk[1]}"[-30:]
            lines.append(f"  {label:>30}  n={len(g):4d}  gap {np.mean(g):+6.2f}±{se(g):.2f}  (+{sum(x > 0 for x in g)}/-{sum(x < 0 for x in g)})")
    return "\n".join(lines)


def cmd_report(args) -> None:
    text = report(load_rows(Path(args.out)), args.baseline)
    print(text)
    (Path(args.out) / "summary.txt").write_text(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--learners", nargs="+", required=True)
    a.add_argument("--opponents", nargs="+", required=True)
    a.add_argument("--seeds", type=int, default=24)
    a.add_argument("--seed-base", type=int, default=970000)
    a.add_argument("--root-seconds", type=float, nargs="+", default=[30, 45, 60, 90, 120])
    a.add_argument("--horizon", type=float, default=120.0)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_play)
    r = sub.add_parser("report")
    r.add_argument("--baseline", default=ROOT_LEARNER)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
