"""H9 rule check: does a V23 variant (``strategic_v25``: exact race walk / last-step commit) beat V23
against the same opponents?  Simulator only, full matches (docs/policy_v2_design_ko.md §28).

``play``: for every ``--ours`` descriptor and (seed, side), our heuristic plays the opponent
(a heuristic descriptor, ``r5`` or ``p11``; nets on CPU) from ``SimState.from_seed`` until the
match ends or ``--cap-s``.  ``--workers`` processes split the seeds (lockstep ``BatchSim`` chunk
each).  Rows append to ``<out>/matches.jsonl``; a rerun skips rows already there.

``report``: per (ours, opponent) the margin minus the baseline's (``--base``, default
strategic_v23) on the same (seed, side), SE over seeds (both sides averaged first), race
(first Hunter tick) win/tie/loss.

    PYTHONPATH=. .venv/bin/python examples/race_rule_eval.py play --opponent strategic_v23 \\
        --ours strategic_v23,strategic_v25 --seeds 128 --workers 4
    PYTHONPATH=. .venv/bin/python examples/race_rule_eval.py report
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

R5 = "checkpoints/rollout_net/r5.pt"
P11 = "checkpoints/distill/frozen/p11_v22.pt"


def _policy(name: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name == "r5":
        from blackout_env.sim.rollout_net import RolloutNet

        return NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device="cpu")
    if name == "p11":
        return NetPolicy.from_checkpoint(str(REPO / P11), device="cpu")
    return HeuristicPolicy(name)


def play_chunk(job):
    """(ours, opponent, seeds, sides, cap_s) -> result rows."""
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import HUNTER, H, W
    from blackout_env.sim.state import SimState

    ours_name, opp_name, seeds, sides, cap_s = job
    F32 = np.float32
    t0 = time.time()
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    ours, opp = _policy(ours_name), _policy(opp_name)
    ours.reset(n)
    opp.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    need = getattr(opp, "needs_inputs", True)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    first = np.full((n, 2), -1, np.int64)
    for t in range(cap_s * 25):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        if need:
            bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if not sel.any():
                    continue
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), s_in)[team]
        bs.step(moves, live)
        for team in (0, 1):
            hit = (first[live, team] < 0) & (bs.ui[live, 5 * team:5 * team + 5, core.U_CLS] == HUNTER).any(axis=1)
            first[live[hit], team] = t + 1
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        o, p = int(sc[j, sd]), int(sc[j, 1 - sd])
        rows.append({"ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd), "margin": o - p,
                     "score_ours": o, "score_opp": p, "over": bool(bs.over[j]), "ticks": int(bs.ticks[j]),
                     "first_ours": int(first[j, sd]), "first_opp": int(first[j, 1 - sd]),
                     "chunk_wall_s": time.time() - t0, "chunk_n": n})
    return rows


def load_rows(out: Path) -> list[dict]:
    path = out / "matches.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_play(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["ours"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    jobs = []
    for ours in args.ours.split(";"):
        todo = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)
                if (ours, args.opponent, s, sd) not in done]
        for c in range(0, len(todo), args.chunk):
            ch = todo[c:c + args.chunk]
            jobs.append((ours, args.opponent, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s))
    print(f"{len(jobs)} chunks", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(play_chunk, jobs):
            with (out / "matches.jsonl").open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            m = np.mean([r["margin"] for r in rows])
            print(f"[{rows[0]['ours']} vs {rows[0]['opponent']}] {len(rows)} matches margin {m:+.1f} "
                  f"wall {rows[0]['chunk_wall_s']:.0f}s", flush=True)


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def race_wtl(rows):
    a = np.array([r["first_ours"] if r["first_ours"] >= 0 else 10 ** 9 for r in rows])
    b = np.array([r["first_opp"] if r["first_opp"] >= 0 else 10 ** 9 for r in rows])
    s = np.sign(b - a)
    return int((s > 0).sum()), int((s == 0).sum()), int((s < 0).sum())


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    lines = []
    for (ours, opp), d in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = by.get((args.base, opp), {})
        keys = sorted(k for k in d if k in base)
        ms = [d[k]["margin"] for k in d]
        line = (f"{ours:44s} vs {opp:16s} n={len(d):4d} margin {np.mean(ms):+6.1f}±{se(ms):4.1f} "
                f"win {np.mean([m > 0 for m in ms]):.2f} race w/t/l {race_wtl(list(d.values()))}")
        if keys and ours != args.base:
            per = defaultdict(list)
            for k in keys:
                per[k[0]].append(d[k]["margin"] - base[k]["margin"])
            ps = [np.mean(v) for v in per.values()]
            line += f"  − {args.base}: {np.mean(ps):+6.1f} ± {se(ps):4.1f} ({len(ps)} seeds)"
            # race outcome transitions vs base
            tr = defaultdict(int)
            for k in keys:
                ra = race_wtl([base[k]])
                rb = race_wtl([d[k]])
                tr[(ra.index(1) if 1 in ra else 1, rb.index(1) if 1 in rb else 1)] += 1
            name = {0: "w", 1: "t", 2: "l"}
            line += "  race " + " ".join(f"{name[a]}>{name[b]}:{v}" for (a, b), v in sorted(tr.items()) if a != b)
        lines.append(line)
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--ours", default="strategic_v23;strategic_v25", help="';'-separated heuristic descriptors")
    q.add_argument("--opponent", default="strategic_v23")
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=32)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--out", default=str(REPO / "reports/race_rule_eval"))
    q = sub.add_parser("report")
    q.add_argument("--base", default="strategic_v23")
    q.add_argument("--out", default=str(REPO / "reports/race_rule_eval"))
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
