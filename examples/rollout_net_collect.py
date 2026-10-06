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

Net teacher (docs §31 Q2: a MyModel student such as P14b labels on the GPU instead of V23 on the CPU):
``--teacher net:PATH`` -- every ``--every``-th decision the teacher net labels both teams' views in one
batched forward (arc label = its arg max bin +- 1, angle = its executed heading, both canonical) and
its logits / offsets are stored as the ``--p11-kl`` targets (``p11_q`` / ``p11_off``).  Each job plays
one opponent from ``--pool`` (weights -> job counts): ``self`` (the net), ``teacher``, ``net:PATH`` /
``small:PATH`` (batched nets), ``heur:<descriptor>`` (CPU, per game).  ``--teacher-frac``: games whose
team A the teacher drives.

    PYTHONPATH=. .venv/bin/python examples/rollout_net_collect.py --net S1.pt --teacher net:P14B.pt \
        --pool '{"self": 0.4, "teacher": 0.3, "heur:strategic_v25": 0.15, "net:P11.pt": 0.15}' \
        --games 1536 --per-job 128 --workers 3 --device cuda --out simdata_p14
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


def _net_policy(spec: str, device: str):
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    if spec.startswith("small:"):
        return NetPolicy(RolloutNet.from_checkpoint(spec[6:]), device=device)
    if spec.startswith("net:"):
        return NetPolicy.from_checkpoint(spec[4:], device=device)
    raise ValueError(spec)


def teacher_labels(q: np.ndarray, off: np.ndarray):
    """[m, 5, K] canonical logits + offsets of the teacher -> (labels uint64 [m, 5]: arg max bin +- 1,
    angles f32 [m, 5]: the executed heading 2 pi j / K + off_j, canonical frame) -- what
    ``label_rows`` gives for the teacher's own move (decode_moves = the same heading, mirrored for B).
    A STAND teacher (odd width K + 1, P18b §43 / P20): a STAND arg max is a standing move = no label, NaN angle."""
    k = q.shape[-1] - q.shape[-1] % 2
    j = q.argmax(-1)
    stand = j == k
    j = np.where(stand, 0, j)
    o = np.take_along_axis(off[..., :k], j[..., None], -1)[..., 0].astype(np.float64)
    ang = 2 * math.pi * j / k + o
    ang = np.where(stand, np.nan, np.arctan2(np.sin(ang), np.cos(ang))).astype(np.float32)
    one = np.uint64(1)
    labels = np.zeros(j.shape, np.uint64)
    for s in (-1, 0, 1):
        labels |= one << ((j + s) % k).astype(np.uint64)
    labels[stand] = 0
    return labels, ang


def worker_net(job):
    """One job of the net-teacher collection: ``job["games"]`` seeds against one opponent spec."""
    import torch

    torch.set_num_threads(1)
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import N_AGENT_COLS, BatchSim, HeuristicPolicy, decode_moves

    dev = job["device"]
    rng = np.random.default_rng(job["seed"])
    seeds = [job["seed0"] + i for i in range(job["games"])]
    teacher = _net_policy(job["teacher"], dev)
    student = _net_policy("small:" + job["net"], dev) if job["net"] else None
    opp = job["opponent"]
    opp_pol = (HeuristicPolicy(opp[5:], seed=job["seed"]) if opp.startswith("heur:") else
               _net_policy(opp, dev) if opp.startswith(("net:", "small:")) else None)
    if opp_pol is not None:
        opp_pol.reset(len(seeds))
    sims = [SimState.from_seed(s) for s in seeds]
    for s, k in zip(sims, seeds):
        s.reseed(k)
    bs = BatchSim(sims)
    n = bs.n
    drive_t = (rng.random(n) < job["teacher_frac"]) if student is not None else np.ones(n, bool)
    views = np.zeros((n, 2, 24, 24), np.int16)
    team = np.zeros((n, 2, 4), np.float32)
    agents = np.zeros((n, 2, 10, N_AGENT_COLS), np.float32)
    moves = np.zeros((n, 10, 2), np.float32)
    out = {k: [] for k in ("map", "team_state", "agent_states", "labels", "angles", "tick", "p11_q", "p11_off")}
    t_net = 0.0
    width = ow = 32
    for t in range(job["cap"]):
        live = np.nonzero(~bs.over)[0]
        if not len(live):
            break
        m = len(live)
        bs.render_packed(live, views, team, agents)
        record = t % job["every"] == 0
        # teacher views: both teams when recording; else only where it moves (team A it drives, team B as opponent)
        need_a = drive_t[live] | record
        need_b = np.full(m, record or opp == "teacher")
        rows_t = np.concatenate([np.nonzero(need_a)[0] * 2, np.nonzero(need_b)[0] * 2 + 1])
        rows_t.sort()
        c0 = time.perf_counter()
        g2, t2, a2 = views[:m].reshape(2 * m, 24, 24), team[:m].reshape(2 * m, 4), agents[:m].reshape(2 * m, 10, N_AGENT_COLS)
        qt = np.zeros((2 * m, 5, width), np.float32)  # width 33 = a STAND teacher (P20); the KL targets keep the 32 bins
        ot = np.zeros((2 * m, 5, ow), np.float32)
        if len(rows_t):
            qa, oa = teacher.forward(g2[rows_t], t2[rows_t], a2[rows_t])
            if qa.shape[-1] != width or oa.shape[-1] != ow:
                width, ow = qa.shape[-1], oa.shape[-1]
                qt = np.zeros((2 * m, 5, width), np.float32)
                ot = np.zeros((2 * m, 5, ow), np.float32)
            qt[rows_t], ot[rows_t] = qa, oa
        team_b = np.tile(np.array([False, True]), m)
        mv_t = decode_moves(qt, ot, team_b).reshape(m, 2, 5, 2)
        # the student: team A where it drives, team B in self-play
        rows_s = [2 * i for i in np.nonzero(~drive_t[live])[0]] + ([2 * i + 1 for i in range(m)] if opp == "self" else [])
        mv_s = np.zeros((m, 2, 5, 2), np.float32)
        if rows_s and student is not None:
            rows_s = np.sort(np.asarray(rows_s))
            qs, os_ = student.forward(g2[rows_s], t2[rows_s], a2[rows_s])
            mv_s.reshape(2 * m, 5, 2)[rows_s] = decode_moves(qs, os_, team_b[rows_s])
        t_net += time.perf_counter() - c0
        moves[live, :5] = np.where(drive_t[live][:, None, None], mv_t[:, 0], mv_s[:, 0])
        if opp == "self":
            moves[live, 5:] = mv_s[:, 1]
        elif opp == "teacher":
            moves[live, 5:] = mv_t[:, 1]
        else:
            moves[live, 5:] = opp_pol.moves(bs, live, (1,), None if not opp_pol.needs_inputs else
                                            (views, team, agents))[1]
        if record:
            lab, ang = teacher_labels(qt, ot)
            out["map"].append(g2.view(np.uint16).copy())
            out["team_state"].append(t2.copy())
            out["agent_states"].append(a2.copy())
            out["labels"].append(lab)
            out["angles"].append(ang)
            out["tick"].append(np.full(2 * m, t, np.int32))
            out["p11_q"].append(qt[..., :32].astype(np.float16))
            out["p11_off"].append(ot[..., :32].astype(np.float16))
        bs.step(moves, live)
    data = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(job["path"], **data)
    return job["path"], len(data["tick"]), float(np.mean(bs.over)), t_net


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
    p.add_argument("--teacher", default="strategic_v23",
                   help="net:PATH (a MyModel student, batched on --device: the worker_net path) or the V23 act teacher")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.teacher.startswith(("net:", "small:")):
        return main_net(args)
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
        import torch

        from rollout_net_train import p11_targets

        data.update(p11_targets(data, device="cuda" if torch.cuda.is_available() else "mps"))
    np.savez(out / "train.npz", **data)
    print("rows", len(data["tick"]), f"{time.time() - t0:.0f}s")


def main_net(args):
    import multiprocessing as mp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads(args.pool) if args.pool else {"self": 0.5, "teacher": 0.5}
    n_jobs = args.games // args.per_job
    w = np.array(list(pool.values()), float)
    counts = np.maximum(1, np.floor(w / w.sum() * n_jobs).astype(int))  # every listed opponent gets >= 1 job (§39)
    counts[np.argmax(w)] += n_jobs - counts.sum()
    opps = [o for o, c in zip(pool, counts) for _ in range(c)]
    jobs = [{"net": args.net if args.net != "none" else "", "teacher": args.teacher, "opponent": o, "games": args.per_job,
             "seed0": args.seed0 + i * args.per_job, "seed": i, "teacher_frac": args.teacher_frac, "every": args.every,
             "cap": args.cap_s * 25, "device": args.device, "path": str(out / f"part_{i:03d}.npz")}
            for i, o in enumerate(opps)]
    # slow (heuristic) jobs first
    jobs.sort(key=lambda j: not j["opponent"].startswith("heur:"))
    t0 = time.time()
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(jobs)} jobs ({len(todo)} to do):", {o: int(c) for o, c in zip(pool, counts)}, flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for path, rows, over, t_net in ex.map(worker_net, todo):
            print(path, rows, f"over {over:.2f} net {t_net:.0f}s", f"{time.time() - t0:.0f}s", flush=True)
    parts = [dict(np.load(j["path"])) for j in sorted(jobs, key=lambda j: j["path"])]
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    np.savez(out / "train.npz", **data)
    print("rows", len(data["tick"]), f"{time.time() - t0:.0f}s", flush=True)
    for j in jobs:
        Path(j["path"]).unlink()


if __name__ == "__main__":
    main()
