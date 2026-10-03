"""P15: after losing the opening Hunter race, should our late Hunter trade itself off against the
enemy Hunter?  Simulator only, full matches (docs/policy_v2_design_ko.md §30).

``play``: as ``race_rule_eval.py play`` (our heuristic descriptor vs a heuristic / ``p11`` / ``r5``
from ``SimState.from_seed`` until the end or ``--cap-s``), but every row also records what
happens around the Hunters: first Hunter tick per team, the first enemy Hunter's fate (traded /
killed / alive), Hunter-vs-Hunter trades (first tick, count), kills by each team's Hunters split
by victim class, Hunter-decisions per team and decisions where only one team has a Hunter,
scores at 10 / 30 / 60 s, and at the decision the opponent's first Hunter appears: our nearest
Collector's exact path distance to the sanctuary and whether we had a Hunter 1 / 3 / 10 s later.

``report``: per (ours, opponent) the margin and its paired difference to ``--base`` (same seed
and side, SE over seeds after averaging sides), overall and split by the base's race outcome
(won / tied / lost), plus the Hunter statistics.

    PYTHONPATH=. .venv/bin/python examples/late_hunter_eval.py play --opponent strategic_v25 \\
        --ours 'strategic_v25;strategic_v26' --seeds 128 --workers 4 --out reports/p15_eval
    PYTHONPATH=. .venv/bin/python examples/late_hunter_eval.py report --base strategic_v25 --out reports/p15_eval
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
MARKS = {"10s": 250, "30s": 750, "60s": 1500}


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
    from blackout_env.heuristics.race_path import RacePlanner
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import COLLECTOR, HUNTER, H, W
    from blackout_env.sim.state import SimState

    ours_name, opp_name, seeds, sides, cap_s = job
    F32 = np.float32
    t0 = time.time()
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    ours, opp = _policy(ours_name), _policy(opp_name)
    ours.reset(n)
    opp.reset(n)
    sims = [SimState.from_seed(int(s)) for s in seeds]
    bs = BatchSim(sims)
    need = getattr(ours, "needs_inputs", True) or getattr(opp, "needs_inputs", True)
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    first = np.full((n, 2), -1, np.int64)
    first_unit = np.full((n, 2), -1, np.int64)       # unit index of each team's first Hunter
    first_fate = [["alive", "alive"] for _ in range(n)]
    first_death = np.full((n, 2), -1, np.int64)
    trade_first = np.full(n, -1, np.int64)
    trades = np.zeros(n, np.int64)
    kills = np.zeros((n, 2, 3), np.int64)            # [game, killer team, victim class]
    hunter_dec = np.zeros((n, 2), np.int64)
    solo_dec = np.zeros((n, 2), np.int64)            # decisions only this team has a Hunter
    marks = {k: np.zeros((n, 2), np.int64) for k in MARKS}
    at_first_opp = [None] * n                        # (nearest own Collector path distance, ...)
    hunter_after = np.full((n, 3), False)            # we had a Hunter 1 / 3 / 10 s after the enemy's first
    planners = {}

    def planner(b, team):
        sim = sims[b]
        mask = sim.layout.walk[team].T[::-1].copy()
        key = (team, mask.tobytes())
        if key not in planners:
            planners[key] = RacePlanner(mask, [(23 - y, x) for x in (11, 12) for y in (11, 12)], 0.275)
        return planners[key]

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
        for b in live:
            sd = int(sides[b])
            ne = int(bs.g[b, core.G_NEV])
            dead_hunters = []
            for e in bs.ev[b, :ne]:
                kind = int(e[1])
                if kind == core.EV_TRANSFORM and int(e[3]) == HUNTER:
                    u = int(e[2])
                    tm = 0 if u < 5 else 1
                    rel = 0 if tm == sd else 1
                    if first_unit[b, rel] < 0:
                        first_unit[b, rel] = u
                elif kind == core.EV_KILL:
                    killer, victim, vcls, kcls = (int(x) for x in e[2:6])
                    vrel = 0 if (victim < 5) == (sd == 0) else 1
                    if killer < 0 and vcls == HUNTER:
                        dead_hunters.append(victim)
                    elif killer >= 0 and kcls == HUNTER:
                        kills[b, 1 - vrel, vcls] += 1
                    for rel in (0, 1):
                        if victim == first_unit[b, rel] and first_death[b, rel] < 0 and vcls == HUNTER:
                            first_death[b, rel] = t + 1
                            first_fate[b][rel] = "traded" if killer < 0 else "killed"
            if dead_hunters:
                trades[b] += len(dead_hunters) // 2
                if trade_first[b] < 0:
                    trade_first[b] = t + 1
            cls = bs.ui[b, :, core.U_CLS]
            nh = [(cls[5 * sd:5 * sd + 5] == HUNTER).sum(), (cls[5 * (1 - sd):5 * (1 - sd) + 5] == HUNTER).sum()]
            hunter_dec[b] += nh
            if nh[0] and not nh[1]:
                solo_dec[b, 0] += 1
            elif nh[1] and not nh[0]:
                solo_dec[b, 1] += 1
            for rel in (0, 1):
                if first[b, rel] < 0 and nh[rel]:
                    first[b, rel] = t + 1
            if first[b, 1] == t + 1 and at_first_opp[b] is None:
                pl = planner(b, sd)
                ds = [pl.distance((23.5 - float(bs.uf[b, u, 1]), float(bs.uf[b, u, 0]) - 0.5))
                      for u in range(5 * sd, 5 * sd + 5) if cls[u] == COLLECTOR]
                at_first_opp[b] = float(min(ds)) if ds else None
            if first[b, 1] >= 0:
                for k, lag in enumerate((25, 75, 250)):
                    if t + 1 == first[b, 1] + lag:
                        hunter_after[b, k] = bool(nh[0])
            for k, tick in MARKS.items():
                if t + 1 == tick:
                    marks[k][b] = bs.scores[b, [sd, 1 - sd]]
    sc = bs.scores
    rows = []
    for j, (s, sd) in enumerate(zip(seeds, sides)):
        o, p = int(sc[j, sd]), int(sc[j, 1 - sd])
        for k, tick in MARKS.items():
            if bs.ticks[j] // 2 < tick:  # ended before the mark: final score
                marks[k][j] = (o, p)
        rows.append({
            "ours": ours_name, "opponent": opp_name, "seed": int(s), "side": int(sd), "margin": o - p,
            "score_ours": o, "score_opp": p, "over": bool(bs.over[j]), "ticks": int(bs.ticks[j]),
            "first_ours": int(first[j, 0]), "first_opp": int(first[j, 1]),
            "first_fate": first_fate[j], "first_death": [int(x) for x in first_death[j]],
            "trade_first": int(trade_first[j]), "trades": int(trades[j]),
            "kills_ours": [int(x) for x in kills[j, 0]], "kills_opp": [int(x) for x in kills[j, 1]],
            "hunter_dec": [int(x) for x in hunter_dec[j]], "solo_dec": [int(x) for x in solo_dec[j]],
            "marks": {k: [int(x) for x in v[j]] for k, v in marks.items()},
            "near_at_first_opp": at_first_opp[j], "hunter_after": [bool(x) for x in hunter_after[j]],
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
    for opponent in args.opponent.split(";"):
        for ours in args.ours.split(";"):
            todo = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)
                    if (ours, opponent, s, sd) not in done]
            for c in range(0, len(todo), args.chunk):
                ch = todo[c:c + args.chunk]
                jobs.append((ours, opponent, [s for s, _ in ch], [sd for _, sd in ch], args.cap_s))
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


def race(r) -> str:
    a = r["first_ours"] if r["first_ours"] >= 0 else 10 ** 9
    b = r["first_opp"] if r["first_opp"] >= 0 else 10 ** 9
    return "w" if a < b else "t" if a == b else "l"


def paired(d, base, keys):
    """Mean ± SE over seeds of (ours - base) margin, sides averaged per seed first."""
    per = defaultdict(list)
    for k in keys:
        per[k[0]].append(d[k]["margin"] - base[k]["margin"])
    ps = [np.mean(v) for v in per.values()]
    return (float(np.mean(ps)), se(ps), len(keys)) if ps else (float("nan"), float("nan"), 0)


def hunter_stats(rows) -> str:
    if not rows:
        return ""
    lost = [r for r in rows if race(r) == "l"]
    f = lambda key, i: np.mean([r[key][i] for r in rows])
    s = (f"trades {np.mean([r['trades'] for r in rows]):.2f} "
         f"Hkills ours/opp C {f('kills_ours', 0):.1f}/{f('kills_opp', 0):.1f} "
         f"Hdec ours/opp {f('hunter_dec', 0) / 25:.0f}/{f('hunter_dec', 1) / 25:.0f}s "
         f"solo ours/opp {f('solo_dec', 0) / 25:.0f}/{f('solo_dec', 1) / 25:.0f}s")
    if lost:
        fate = defaultdict(int)
        for r in lost:
            fate[r["first_fate"][1]] += 1
        after = np.mean([r["hunter_after"] for r in lost], axis=0)
        near = [r["near_at_first_opp"] for r in lost if r["near_at_first_opp"] is not None]
        s += (f" | lost n={len(lost)}: enemy 1st H {dict(fate)}, ours H +1/3/10s {after[0]:.2f}/{after[1]:.2f}/{after[2]:.2f}, "
              f"nearest walker {np.median(near) if near else float('nan'):.2f} tiles (<=1: {np.mean([x <= 1 for x in near]) if near else 0:.2f})")
    return s


def cmd_report(args):
    rows = load_rows(Path(args.out))
    by = defaultdict(dict)
    for r in rows:
        by[(r["ours"], r["opponent"])][(r["seed"], r["side"])] = r
    lines = []
    for (ours, opp), d in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ms = [r["margin"] for r in d.values()]
        wtl = [sum(race(r) == c for r in d.values()) for c in "wtl"]
        lines.append(f"{ours} vs {opp}: n={len(d)} margin {np.mean(ms):+.1f}±{se(ms):.1f} "
                     f"win {np.mean([m > 0 for m in ms]):.2f} race w/t/l {wtl}")
        lines.append("    " + hunter_stats(list(d.values())))
        base = by.get((args.base, opp), {})
        keys = sorted(k for k in d if k in base)
        if keys and ours != args.base:
            m, e, nk = paired(d, base, keys)
            line = f"    - {args.base}: {m:+.1f} ± {e:.1f} (n={nk})"
            for c, nm in (("w", "won"), ("t", "tied"), ("l", "lost")):
                sub = [k for k in keys if race(base[k]) == c]
                if sub:
                    m, e, nk = paired(d, base, sub)
                    mb = np.mean([base[k]["margin"] for k in sub])
                    line += f" | base race {nm} n={nk}: {m:+.1f} ± {e:.1f} (base margin {mb:+.1f})"
            lines.append(line)
    txt = "\n".join(lines)
    print(txt)
    (Path(args.out) / "report.txt").write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("play")
    q.add_argument("--ours", default="strategic_v25;strategic_v26", help="';'-separated heuristic descriptors")
    q.add_argument("--opponent", default="strategic_v25", help="';'-separated opponents")
    q.add_argument("--seeds", type=int, default=128)
    q.add_argument("--seed0", type=int, default=4_000_000)
    q.add_argument("--chunk", type=int, default=32)
    q.add_argument("--cap-s", type=int, default=150)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--out", default=str(REPO / "reports/p15_eval"))
    q = sub.add_parser("report")
    q.add_argument("--base", default="strategic_v25")
    q.add_argument("--out", default=str(REPO / "reports/p15_eval"))
    args = p.parse_args()
    {"play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
