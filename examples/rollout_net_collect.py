"""Simulator DAgger data for the small rollout net: the net drives, strategic_v23 labels.

Batched sim games from seeds: team A is driven by the rollout net (or, per game with probability
``--teacher-frac``, by strategic_v23 itself), team B by a pool (rollout net / V23 / V22 / any-angle
V21 / V17).  Every ``--every``-th decision both teams' canonical views are labelled with
strategic_v23's own move on that observation (act-teacher labels exactly as the P11 collection:
the raw heading in the bins' canonical frame + the arc label nearest bin +- 1 at K = 32; a
standing move = no label), and written in examples/rollout_net_train.py's packed format, so
``train --extra`` can mix them with the Unity shards.  V23 is stateless (fresh == persistent),
so one instance per game and team is the act teacher.

    PYTHONPATH=. .venv/bin/python examples/rollout_net_collect.py --net checkpoints/rollout_net/r4.pt \
        --games 512 --workers 4 --out checkpoints/rollout_net/simdata
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
POOL = {"small": 0.3, "strategic_v23": 0.3, "strategic_v22": 0.15, "strategic_v21_anyangle": 0.1, "strategic_v17": 0.15}
K = 32


def label_rows(moves: np.ndarray, team_b: bool):
    """[5, 2] world moves -> (labels uint64 [5], angles f32 [5]) in the canonical frame."""
    from blackout_env.model.direction_bins import vector_to_bin

    mv = np.clip(np.asarray(moves, np.float32), -1, 1)
    if team_b:  # the y = x reflection: angle t -> 90 deg - t
        mv = mv[:, ::-1].copy()
    labels = np.zeros(5, np.uint64)
    angles = np.full(5, np.nan, np.float32)
    for u in range(5):
        if float(np.linalg.norm(mv[u])) < 1e-5:
            continue
        c = int(vector_to_bin(mv[u], K))
        for s in (-1, 0, 1):
            labels[u] |= np.uint64(1) << np.uint64((c + s) % K)
        angles[u] = np.float32(math.atan2(float(mv[u, 1]), float(mv[u, 0])))
    return labels, angles


def pack_inputs(graphic: np.ndarray) -> np.ndarray:
    """float [n, 13, H, W] -> uint16 [n, H, W] (rollout_net_train.pack_graphic's layout)."""
    base = graphic[:, :8].argmax(1).astype(np.uint16)
    bat = np.rint(graphic[:, 8] * 15).astype(np.uint16)
    items = graphic[:, 9:13]
    item = np.where(items.max(1) > 0, items.argmax(1) + 1, 0).astype(np.uint16)
    return base | (bat << 3) | (item << 7)


def worker(job):
    import torch

    torch.set_num_threads(1)
    import numba

    numba.set_num_threads(1)
    from blackout_env.sim import SimState, obs_from_raw
    from blackout_env.sim import core
    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.distill.opponents import make_opponent

    rng = np.random.default_rng(job["seed"])
    seeds = [job["seed0"] + i for i in range(job["games"])]
    net = NetPolicy(RolloutNet.from_checkpoint(job["net"]), device=job["device"])
    sims = [SimState.from_seed(s) for s in seeds]
    for s, k in zip(sims, seeds):
        s.reseed(k)
    bs = BatchSim(sims)
    n = bs.n
    pool = job.get("pool") or POOL
    names = list(pool)
    opp = rng.choice(names, size=n, p=np.array(list(pool.values())) / sum(pool.values()))
    driver_v23 = rng.random(n) < job["teacher_frac"]
    teachers = [(make_opponent("strategic_v23", 0), make_opponent("strategic_v23", 0)) for _ in range(n)]  # one per team: an instance binds to its first team
    opps = [None if o in ("small", "strategic_v23") else make_opponent(o, int(seeds[b])) for b, o in enumerate(opp)]
    bufs = (np.zeros((n, 2, N_CH, 24, 24), np.float32), np.zeros((n, 2, 4), np.float32),
            np.zeros((n, 2, 10, N_AGENT_COLS), np.float32))
    moves = np.zeros((n, 10, 2), np.float32)
    out = {k: [] for k in ("map", "team_state", "agent_states", "labels", "angles", "tick")}
    a_names = [f"unit_{i}" for i in range(5)]
    b_names = [f"unit_{i}" for i in range(5, 10)]
    for t in range(job["cap"]):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        bs.render(live, *bufs)
        small = net.moves(bs, live, (0, 1), bufs)
        record = t % job["every"] == 0
        rows_map = pack_inputs(bufs[0][: len(live)].reshape(-1, N_CH, 24, 24)).reshape(len(live), 2, 24, 24) if record else None
        for j, b in enumerate(live):
            obs = obs_from_raw(*bs.raw(int(b)))
            ta = teachers[b][0].act({k: obs[k] for k in a_names})
            tb = teachers[b][1].act({k: obs[k] for k in b_names})
            va = np.array([ta[k] for k in a_names], np.float32)
            vb = np.array([tb[k] for k in b_names], np.float32)
            moves[b, :5] = np.clip(va, -1, 1) if driver_v23[b] else small[0][j]
            if opp[b] == "small":
                moves[b, 5:] = small[1][j]
            elif opp[b] == "strategic_v23":
                moves[b, 5:] = np.clip(vb, -1, 1)
            else:
                ob = opps[b].act({k: obs[k] for k in b_names})
                moves[b, 5:] = np.clip(np.array([ob[k] for k in b_names], np.float32), -1, 1)
            if record:
                for team, v in ((0, va), (1, vb)):
                    lab, ang = label_rows(v, team == 1)
                    out["map"].append(rows_map[j, team])
                    out["team_state"].append(bufs[1][j, team].copy())
                    out["agent_states"].append(bufs[2][j, team].copy())
                    out["labels"].append(lab)
                    out["angles"].append(ang)
                    out["tick"].append(t)
        bs.step(moves, live)
    data = {k: np.stack(v) for k, v in out.items()}
    data["tick"] = data["tick"].astype(np.int32)
    np.savez(job["path"], **data)
    return job["path"], len(data["tick"]), float(np.mean(bs.over))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--per-job", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed0", type=int, default=1_100_000)
    p.add_argument("--teacher-frac", type=float, default=0.2)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--cap-s", type=int, default=150)
    p.add_argument("--device", default="cpu")
    p.add_argument("--p11", action="store_true", help="add P11 targets (examples/rollout_net_train.p11_targets, MPS)")
    p.add_argument("--pool", default="", help='team-B pool as JSON, e.g. \'{"small": 1.0}\' (self-play: every row is the net\'s own state)')
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"net": args.net, "games": args.per_job, "seed0": args.seed0 + i * args.per_job, "seed": i,
             "teacher_frac": args.teacher_frac, "every": args.every, "cap": args.cap_s * 25, "device": args.device,
             "pool": json.loads(args.pool) if args.pool else None,
             "path": str(out / f"part_{i:03d}.npz")} for i in range(args.games // args.per_job)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for path, rows, over in ex.map(worker, jobs):
            print(path, rows, f"over {over:.2f}", f"{time.time() - t0:.0f}s", flush=True)
    parts = [dict(np.load(j["path"])) for j in jobs]
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    if args.p11:
        sys.path.insert(0, str(REPO / "examples"))
        from rollout_net_train import p11_targets

        data.update(p11_targets(data))
    np.savez(out / "train.npz", **data)
    print("rows", len(data["tick"]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
