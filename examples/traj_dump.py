"""Dump per-decision trajectories (positions, class, held item) + kill events of ``--ours`` vs ``--opponent``
for the first ``--cap-s`` seconds, one .npz per chunk, for behaviour analysis (examples/traj_analysis.py).

    PYTHONPATH=. .venv/bin/python examples/traj_dump.py --ours 'V32.1;net:checkpoints/rl/abl_beta01/final.pt' \\
        --opponent V32.1 --seeds 32 --workers 8 --out reports/heur_econ/traj
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))


def tag(name: str) -> str:
    return name.split("/")[-2] if name.startswith("net:") else name.replace(":", "_").replace(",", "_").replace("=", "")


def play_chunk(job):
    import torch

    torch.set_num_threads(1)
    from v27_eval import _policy

    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState

    ours_name, opp_name, seeds, sides, cap_s, out = job
    F32 = np.float32
    n = len(seeds)
    sides = np.asarray(sides, np.int64)
    ours, opp = _policy(ours_name), _policy(opp_name)
    ours.reset(n)
    opp.reset(n)
    bs = BatchSim([SimState.from_seed(int(s)) for s in seeds])
    bufs = (np.zeros((n, 2, N_CH, H, W), F32), np.zeros((n, 2, 4), F32), np.zeros((n, 2, 10, N_AGENT_COLS), F32))
    moves = np.zeros((n, 10, 2), F32)
    T = cap_s * 25
    pos = np.full((n, T, 10, 2), np.nan, np.float32)
    cls = np.full((n, T, 10), -1, np.int8)
    held = np.full((n, T, 10), -1, np.int8)
    score = np.zeros((n, T, 2), np.int16)
    kills = []  # (b, t, killer, victim, vcls, kcls)
    for t in range(T):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        bs.render(live, *bufs)
        sub = tuple(x[: len(live)] for x in bufs)
        for team in (0, 1):
            mine = sides[live] == team
            for pol, sel in ((ours, mine), (opp, ~mine)):
                if not sel.any():
                    continue
                s_in = sub if sel.all() else tuple(x[sel] for x in sub)
                moves[live[sel], 5 * team:5 * team + 5] = pol.moves(bs, live[sel], (team,), s_in)[team]
        pos[live, t] = bs.uf[live, :, :2]
        cls[live, t] = bs.ui[live, :, core.U_CLS]
        h = bs.ui[live, :, core.U_HELD]
        held[live, t] = np.where(h >= 0, bs.it[live[:, None], np.maximum(h, 0), core.I_TYPE], -1)
        bs.step(moves, live)
        score[live, t] = bs.scores[live]
        for b in live:
            ne = int(bs.g[b, core.G_NEV])
            for e in bs.ev[b, :ne]:
                if int(e[1]) == core.EV_KILL:
                    kills.append((b, t, int(e[2]), int(e[3]), int(e[4]), int(e[5])))
    fn = Path(out) / f"{tag(ours_name)}__{tag(opp_name)}__{seeds[0]}_{sides[0]}.npz"
    np.savez_compressed(fn, seeds=np.asarray(seeds), sides=sides, pos=pos, cls=cls, held=held, score=score,
                        kills=np.asarray(kills, np.int64).reshape(-1, 6))
    return str(fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed0", type=int, default=5_100_000)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--cap-s", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = []
    for opp in a.opponent.split(";"):
        for ours in a.ours.split(";"):
            todo = [(s, sd) for s in range(a.seed0, a.seed0 + a.seeds) for sd in (0, 1)]
            for c in range(0, len(todo), a.chunk):
                ch = todo[c:c + a.chunk]
                jobs.append((ours, opp, [s for s, _ in ch], [sd for _, sd in ch], a.cap_s, a.out))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for fn in pool.imap_unordered(play_chunk, jobs):
            print(fn, flush=True)


if __name__ == "__main__":
    main()
