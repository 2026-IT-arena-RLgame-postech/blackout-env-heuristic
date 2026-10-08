"""D1 (§48): the students' output distribution for the staged reserve in the race-join decisions (ticks 80-99).

Plays ``--ours`` (net aliases of d1_why) against ``--opponent`` with the V30 shadow for the first ``--ticks``
decisions and, for every tick 80-99 and every unit the shadow has staged, stores the student's softmax mass on
STAND, on the shadow's heading (its K = 32 bin +- 1), on the back half-plane (> 90 deg from the shadow's heading),
the arg-max category, the shadow's STAND / joining flags and the unit's sanctuary distance.

    PYTHONPATH=. python examples/d1_probe.py --ours 'p19;p18bp2' --opponent v28 --seeds 32
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

F32 = np.float32


def run(job):
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, mirror_order
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState
    from d1_why import make_policy, sanct_dist

    ours_name, opp_name, seeds, sides, ticks = job
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    ours, opp, sh = make_policy(ours_name), make_policy(opp_name), make_policy("v30")
    for p in (ours, opp, sh):
        p.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    rows = []
    for t in range(ticks):
        live = np.nonzero(~bs.over)[0]
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        own = np.zeros((len(live), 5, 2), F32)
        q_all = None
        for team in (0, 1):
            sel = sides[live] == team
            if not sel.any():
                continue
            s_in = tuple(x[sel] for x in sub)
            own[sel] = ours.moves(bs, live[sel], (team,), s_in)[team]
            if 80 <= t < 100:
                g, tm, a = (s_in[0][:, team], s_in[1][:, team], s_in[2][:, team])
                q, _ = ours.forward(np.ascontiguousarray(g), np.ascontiguousarray(tm), np.ascontiguousarray(a))
                width = q.shape[-1]
                k = width - (width % 2)
                order = mirror_order(k)
                if width != k:
                    order = np.r_[order, k]
                qw = q[..., order] if team == 1 else q
                if q_all is None:
                    q_all = np.zeros((len(live), 5, width), F32)
                q_all[sel] = qw
        shm = np.zeros((len(live), 5, 2), F32)
        for team in (0, 1):
            sel = sides[live] == team
            if sel.any():
                shm[sel] = sh.moves(bs, live[sel], (team,), tuple(x[sel] for x in sub))[team]
        if 80 <= t < 100 and q_all is not None:
            width = q_all.shape[-1]
            k = width - (width % 2)
            p = np.exp(q_all - q_all.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            ang_bins = 2 * math.pi * np.arange(k) / k
            for j, b in enumerate(live):
                inst = sh.instances.get((int(b), int(sides[b])))
                if inst is None:
                    continue
                base = 5 * int(sides[b])
                staged = [int(nm.split("_")[1]) - base for nm in (getattr(inst, "staged", []) or [])]
                for u in staged:
                    tv = shm[j, u]
                    tstand = float(np.linalg.norm(tv)) < 1e-3
                    pu = p[j, u]
                    p_stand = float(pu[k]) if width > k else 0.0
                    if tstand:
                        p_arc = p_back = float("nan")
                    else:
                        ta = math.atan2(tv[1], tv[0])
                        d = np.abs((ang_bins - ta + math.pi) % (2 * math.pi) - math.pi)
                        p_arc = float(pu[:k][d <= 1.5 * 2 * math.pi / k].sum())
                        p_back = float(pu[:k][d > math.pi / 2].sum())
                    am = int(pu.argmax())
                    if am == k:
                        cat = "stand"
                    elif tstand:
                        cat = "move"
                    else:
                        da = abs((ang_bins[am] - ta + math.pi) % (2 * math.pi) - math.pi)
                        cat = "arc" if da <= 1.5 * 2 * math.pi / k else ("side" if da <= math.pi / 2 else "back")
                    us = u + base
                    dist = float(sanct_dist(bs.uf[b, us].astype(F32)))
                    rows.append((int(seeds[b]), int(sides[b]), t, u, int(tstand), int(bool(getattr(inst, "joining", False))),
                                 p_stand, p_arc, p_back, cat, dist))
        oo = np.zeros((len(live), 5, 2), F32)
        for team in (0, 1):
            sel = sides[live] == 1 - team
            if sel.any():
                oo[sel] = opp.moves(bs, live[sel], (team,), tuple(x[sel] for x in sub))[team]
        for team in (0, 1):
            mine = (sides[live] == team)[:, None, None]
            moves[live, 5 * team:5 * team + 5] = np.where(mine, own, oo)
        bs.step(moves, live)
        if t == 98:  # the state after decision 99 (index 99 of d1_why's recordings), before decision 100's trades
            h99 = [int((bs.ui[b, 5 * sides[b]:5 * sides[b] + 5, core.U_CLS] == 1).sum()) for b in range(n)]
    return ours_name, rows, list(zip(seeds, sides.tolist(), h99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="p19;p18bp2")
    ap.add_argument("--opponent", default="v28")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=1100000)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    jobs = []
    for ours in args.ours.split(";"):
        tasks = [(s, sd) for s in range(args.seed0, args.seed0 + args.seeds) for sd in (0, 1)]
        for c in range(0, len(tasks), args.chunk):
            ch = tasks[c:c + args.chunk]
            jobs.append((ours, args.opponent, [s for s, _ in ch], [sd for _, sd in ch], 100))
    res = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for ours, rows, h in pool.imap_unordered(run, jobs):
            r = res.setdefault(ours, ([], []))
            r[0].extend(rows)
            r[1].extend(h)
    lines = [f"staged-reserve output distribution, ticks 80-99, vs {args.opponent} ({args.seeds} seeds x 2 sides)"]
    for ours, (rows, h) in res.items():
        h99 = {(s, sd): x for s, sd, x in h}
        lines.append(f"== {ours}: 4 Hunters at 99 {sum(x == 4 for x in h99.values())}/{len(h99)}")
        for lo, hi in ((80, 90), (90, 95), (95, 100)):
            for tst in (0, 1):
                for jn in (0, 1):
                    sel = [r for r in rows if lo <= r[2] < hi and r[4] == tst and r[5] == jn]
                    if len(sel) < 10:
                        continue
                    ps = np.mean([r[6] for r in sel])
                    pa = np.nanmean([r[7] for r in sel]) if not tst else float("nan")
                    pb = np.nanmean([r[8] for r in sel]) if not tst else float("nan")
                    cats = {c: sum(r[9] == c for r in sel) / len(sel) for c in ("arc", "side", "back", "stand", "move")}
                    lines.append(f"  t{lo}-{hi} teacher {'STAND' if tst else 'move '} joining={jn} n={len(sel):4d}  P(stand) {ps:.2f}"
                                 f"  P(teacher arc) {pa:.2f}  P(back) {pb:.2f}  argmax " +
                                 " ".join(f"{c} {v:.2f}" for c, v in cats.items() if v > 0))
        # games that end with 3 Hunters although the teacher wanted to join at 95-99
        miss = {(r[0], r[1]) for r in rows if r[2] >= 95 and r[5] == 1 and r[4] == 0}
        lost = [g for g in miss if h99.get(g) != 4]
        lines.append(f"  teacher joining at 95-99 in {len(miss)} games; of them not 4 Hunters at 99: {len(lost)}")
    s = "\n".join(lines)
    print(s)
    if args.save:
        Path(args.save).write_text(s + "\n")


if __name__ == "__main__":
    main()
