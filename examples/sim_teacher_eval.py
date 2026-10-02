"""Paired teacher comparison in the simulator: every learner plays the same seeds, both sides.

Each (learner, opponent, seed, side) is one full match in ``blackout_env.sim`` (bit-exact with
the Unity build): to 100 points or the 7-minute clock (``--max-seconds`` caps it lower).  The
learner and the opponent are opponents.py descriptors (``strategic_v24:relief=false``,
``ckpt:<path>``, ...).  Rows append to ``<out>/matches.jsonl`` (a rerun skips rows already
there); ``report`` prints each learner's margin and its paired gap against ``--baseline`` per
opponent, with the standard error over pairs.

    PYTHONPATH=. .venv/bin/python examples/sim_teacher_eval.py play --learners strategic_v23 strategic_v24 \\
        --opponents strategic_v23 strategic_v22 --seeds 24 --workers 6 --out reports/teacher_v24/sim
    PYTHONPATH=. .venv/bin/python examples/sim_teacher_eval.py report --baseline strategic_v23 --out reports/teacher_v24/sim
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DPS = 25  # decisions per second
MATCH_SECONDS = 420


def team_names(team: int) -> list[str]:
    return [f"unit_{i}" for i in range(5 * team, 5 * team + 5)]


def play(task: dict) -> dict:
    """One match; the learner plays team ``side``.  Returns the row for matches.jsonl."""
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import SimState
    from blackout_env.sim.mapdata import HUNTER
    from blackout_env.train.distill.opponents import make_opponent

    started = time.time()
    side = task["side"]
    sim = SimState.from_seed(task["seed"])
    learner = make_opponent(task["learner"], task["seed"])
    opponent = make_opponent(task["opponent"], task["seed"] + 7919)
    for policy in (learner, opponent):
        if hasattr(policy, "reset"):
            policy.reset()
    sides = {side: learner, 1 - side: opponent}
    limit = int(task["max_seconds"] * DPS)
    first_hunter = [None, None]
    tick = 0
    lead_at = {}
    while not sim.over and tick < limit:
        obs = sim.obs()
        moves = np.zeros((10, 2), np.float32)
        for team, policy in sides.items():
            acts = policy.act({n: obs[n] for n in team_names(team)})
            for name, a in acts.items():
                moves[int(name.split("_")[1])] = np.clip(np.asarray(a, np.float32), -1, 1)
        sim.step(moves)
        tick += 1
        for i, team in enumerate((side, 1 - side)):
            if first_hunter[i] is None and (sim.ui[5 * team: 5 * team + 5, 0] == HUNTER).any():
                first_hunter[i] = tick
        if tick % (30 * DPS) == 0:
            s = sim.scores
            lead_at[tick // DPS] = s[side] - s[1 - side]
    s = sim.scores
    return {**task, "learner_score": s[side], "opponent_score": s[1 - side], "margin": s[side] - s[1 - side],
            "ticks": tick, "over": sim.over, "first_hunter": first_hunter, "lead_at": lead_at,
            "seconds": round(time.time() - started, 2)}


def key_of(row: dict) -> tuple:
    return row["learner"], row["opponent"], row["seed"], row["side"]


def load_rows(out: Path) -> list[dict]:
    path = out / "matches.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_play(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {key_of(r) for r in load_rows(out)}
    tasks = []
    for opponent in args.opponents:
        for j in range(args.seeds):
            for side in (0, 1):
                for learner in args.learners:
                    t = {"learner": learner, "opponent": opponent, "seed": args.seed_base + j, "side": side,
                         "max_seconds": args.max_seconds}
                    if key_of(t) not in done:
                        tasks.append(t)
    print(f"{len(tasks)} matches to play ({len(done)} already in {out / 'matches.jsonl'})", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as ex, (out / "matches.jsonl").open("a") as fh:
        futures = [ex.submit(play, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}  {time.time() - started:.0f}s", flush=True)


def se(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def report(rows: list[dict], baseline: str, learners: list[str] | None = None) -> str:
    by = defaultdict(dict)  # learner -> (opponent, seed, side) -> row
    for r in rows:
        by[r["learner"]][(r["opponent"], r["seed"], r["side"])] = r
    learners = learners or sorted(by, key=lambda n: (n != baseline, n))
    opponents = sorted({k[0] for k in by.get(baseline, {})})
    lines = []
    base = by.get(baseline, {})
    for learner in learners:
        rows_l = by[learner]
        lines.append(f"== {learner}")
        all_gaps = []
        for opp in opponents:
            keys = [k for k in rows_l if k[0] == opp]
            if not keys:
                continue
            margins = [rows_l[k]["margin"] for k in keys]
            wins = np.mean([np.sign(m) for m in margins])
            capped = np.mean([not rows_l[k]["over"] for k in keys])
            line = f"  {opp[-26:]:>26}  n={len(keys):3d}  margin {np.mean(margins):+6.1f}  W-L {wins:+.2f}  unfinished {capped:.2f}"
            pair = [k for k in keys if k in base]
            if learner != baseline and pair:
                gaps = [rows_l[k]["margin"] - base[k]["margin"] for k in pair]
                all_gaps += gaps
                line += f"  gap {np.mean(gaps):+6.2f}±{se(gaps):.2f} (+{sum(g > 0 for g in gaps)}/-{sum(g < 0 for g in gaps)})"
            lines.append(line)
        if all_gaps:
            lines.append(f"  {'all':>26}  pairs {len(all_gaps)}  gap {np.mean(all_gaps):+6.2f}±{se(all_gaps):.2f}"
                         f" (+{sum(g > 0 for g in all_gaps)}/-{sum(g < 0 for g in all_gaps)})")
    return "\n".join(lines)


def cmd_report(args) -> None:
    text = report(load_rows(Path(args.out)), args.baseline, args.learners)
    print(text)
    (Path(args.out) / "summary.txt").write_text(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--learners", nargs="+", required=True)
    a.add_argument("--opponents", nargs="+", required=True)
    a.add_argument("--seeds", type=int, default=24)
    a.add_argument("--seed-base", type=int, default=960000)
    a.add_argument("--max-seconds", type=float, default=MATCH_SECONDS)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_play)
    r = sub.add_parser("report")
    r.add_argument("--baseline", default="strategic_v23")
    r.add_argument("--learners", nargs="*")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
