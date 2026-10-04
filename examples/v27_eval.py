"""Q6 rule check: does a V25 variant (``strategic_v27``) beat V25 against the same opponents?
(docs/policy_v2_design_ko.md §33.)  Simulator only, full matches, paired by (seed, side).

``play``: every ``--ours`` descriptor (';'-separated) plays ``--opponent`` (a heuristic
descriptor, ``p11`` or ``p14b``; nets on CPU) from ``SimState.from_seed`` until the match ends or
``--cap-s``; ``--workers`` processes split the chunks.  Per match: margin, first Hunter decision of
both teams, the *second race* (each team's first Hunter transform after the opening trades:
later than its first Hunter + 8 decisions), Hunter-seconds of both teams, our nearest
non-walker's distance to the Hunter sanctuary at decision 100.  Rows append to
``<out>/matches.jsonl`` (a rerun skips rows already there).

``report``: per (ours, opponent) margin minus ``--base``'s on the same (seed, side), SE over seeds
(both sides averaged first), race and second-race win/tie/loss.

    PYTHONPATH=. .venv/bin/python examples/v27_eval.py play --opponent strategic_v25 \\
        --ours 'strategic_v25;strategic_v27' --seeds 128 --workers 3
    PYTHONPATH=. .venv/bin/python examples/v27_eval.py report --base strategic_v25
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

NETS = {"p11": "checkpoints/distill/frozen/p11_v22.pt", "p14b": "checkpoints/distill/frozen/p14_v25.pt"}
SANCT = (11.0, 13.0)


def _policy(name: str):
    from blackout_env.sim.batch import HeuristicPolicy, NetPolicy

    if name in NETS:
        return NetPolicy.from_checkpoint(str(REPO / NETS[name]), device="cpu")
    return HeuristicPolicy(name)


def sanct_dist(p):
    dx = np.maximum(np.maximum(SANCT[0] - p[..., 0], p[..., 0] - SANCT[1]), 0)
    dy = np.maximum(np.maximum(SANCT[0] - p[..., 1], p[..., 1] - SANCT[1]), 0)
    return np.hypot(dx, dy)


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
    need = getattr(opp, "needs_inputs", True) or getattr(ours, "needs_inputs", True)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    first = np.full((n, 2), -1, np.int64)
    second = np.full((n, 2), -1, np.int64)
    hsec = np.zeros((n, 2))
    reserve = np.full(n, np.nan)
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
        if t == 98:  # our units not in the opening walk (> 1 tile from the sanctuary)
            for b in live:
                u0 = 5 * int(sides[b])
                d = sanct_dist(bs.uf[b, u0:u0 + 5])
                far = d[d > 1.0]
                reserve[b] = float(far.min()) if len(far) else np.nan
        bs.step(moves, live)
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if e[1] == core.EV_TRANSFORM and e[3] == HUNTER:
                    team = 0 if e[2] < 5 else 1
                    if first[b, team] >= 0 and second[b, team] < 0 and t + 1 > first[b, team] + 8:
                        second[b, team] = t + 1
        for team in (0, 1):
            h = (bs.ui[live, 5 * team:5 * team + 5, core.U_CLS] == HUNTER)
            hsec[live, team] += h.sum(1) / 25.0
            hit = (first[live, team] < 0) & h.any(axis=1)
            first[live[hit], team] = t + 1
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        o, p = int(sc[j, sd]), int(sc[j, 1 - sd])
        rows.append({"ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd), "margin": o - p,
                     "score_ours": o, "score_opp": p, "over": bool(bs.over[j]), "ticks": int(bs.ticks[j]),
                     "first_ours": int(first[j, sd]), "first_opp": int(first[j, 1 - sd]),
                     "second_ours": int(second[j, sd]), "second_opp": int(second[j, 1 - sd]),
                     "hunter_s": float(hsec[j, sd]), "opp_hunter_s": float(hsec[j, 1 - sd]),
                     "reserve100": float(reserve[j]), "chunk_wall_s": time.time() - t0, "chunk_n": n})
    return rows


def load_rows(out: Path) -> list[dict]:
    path = out / "matches.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_play(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["ours"], r["opponent"], r["seed"], r["side"]) for r in load_rows(out)}
    jobs = []
    for opp in args.opponent.split(";"):
        for ours in args.ours.split(";"):
            todo = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)
                    if (ours, opp, s, sd) not in done]
            for c in range(0, len(todo), args.chunk):
                ch = todo[c:c + args.chunk]
                jobs.append((ours, opp, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s))
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


def wtl(rows, a="first_ours", b="first_opp"):
    x = np.array([r[a] if r[a] >= 0 else 10 ** 9 for r in rows])
    y = np.array([r[b] if r[b] >= 0 else 10 ** 9 for r in rows])
    s = np.sign(y - x)
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
        v = list(d.values())
        line = (f"{ours:52s} vs {opp:14s} n={len(d):4d} margin {np.mean(ms):+6.1f}±{se(ms):4.1f} "
                f"win {np.mean([m > 0 for m in ms]):.2f} race {wtl(v)} 2nd {wtl(v, 'second_ours', 'second_opp')} "
                f"H-s {np.mean([r['hunter_s'] for r in v]):.0f}/{np.mean([r['opp_hunter_s'] for r in v]):.0f} "
                f"res100 {np.nanmedian([r['reserve100'] for r in v]):.1f}")
        if keys and ours != args.base:
            per = defaultdict(list)
            tied = defaultdict(list)
            for k in keys:
                per[k[0]].append(d[k]["margin"] - base[k]["margin"])
            ps = [np.mean(x) for x in per.values()]
            line += f"\n{'':52s}    - {args.base}: {np.mean(ps):+6.1f} ± {se(ps):4.1f} ({len(ps)} seeds)"
            for k in keys:
                if wtl([base[k]])[1] == 1 and wtl([d[k]])[1] == 1:
                    tied[k[0]].append(d[k]["margin"] - base[k]["margin"])
            if tied:
                pt = [np.mean(x) for x in tied.values()]
                line += f"; both race-tied {sum(len(x) for x in tied.values())}: {np.mean(pt):+6.1f} ± {se(pt):4.1f}"
            same = np.mean([d[k]["margin"] == base[k]["margin"] and d[k]["ticks"] == base[k]["ticks"] for k in keys])
            line += f"; identical {same:.2f}"
        lines.append(line)
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--ours", default="strategic_v25;strategic_v27", help="';'-separated heuristic descriptors")
    q.add_argument("--opponent", default="strategic_v25", help="';'-separated")
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=32)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--out", default=str(REPO / "reports/late_search_why/v27"))
    q = sub.add_parser("report")
    q.add_argument("--base", default="strategic_v25")
    q.add_argument("--out", default=str(REPO / "reports/late_search_why/v27"))
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
