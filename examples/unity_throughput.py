"""Unity eval throughput on a server: matches/s vs the number of Unity workers (docs §21).

For each ``--workers`` count a fresh pool of that many (Unity player + Python policy) workers plays
``--per-worker`` matches each of ``--pair A,B`` (team A vs team B, opponents.py descriptors or the
shorthands below), full length with ``stop_when_exhausted`` (the Unity evals' end rule).  Timing
starts once every worker has connected and finished a warm-up match, so it is steady-state
throughput: matches/s, decisions/s, mean wall per match, and the machine's load average.

Shorthands: ``v23`` ``v19`` (heuristics), ``p11`` (P11 on the worker's CPU, 1 thread), ``p11cuda``
(P11 on a GPU, workers dealt round-robin over ``--gpus``), ``r5``.

    PYTHONPATH=. python examples/unity_throughput.py --pair v23,v19 --workers 8 16 24 --out reports/unity_throughput
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P11 = "checkpoints/distill/frozen/p11_v22.pt"
R5 = "checkpoints/rollout_net/r5.pt"
SHORT = {"v23": "strategic_v23", "v19": "strategic_v19", "p11": f"ckpt:{P11}", "r5": f"ckpt:{R5}"}

_ENV = None
_GPU = None


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(2.0)
    try:
        _ENV.close()
    finally:
        os._exit(0)


def _init(build, time_scale, gpus):
    global _ENV, _GPU
    _GPU = gpus.get() if gpus is not None else None
    if _GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU)
    import warnings
    from multiprocessing import util

    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from blackout_env import BlackOutEnv

    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                       additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    util.Finalize(_ENV, _ENV.close, exitpriority=10)
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()


def _policy(name: str, seed: int):
    from blackout_env.train.distill.opponents import make_opponent

    if name == "p11cuda":
        from blackout_env.model.my_policy import MyPolicy
        from blackout_env.train.distill.inference import _FastNet, load_checkpoint_net

        fast = _FastNet(load_checkpoint_net(str(REPO / P11)).to("cuda"))
        fast._traced = fast.own  # no jit trace on CUDA: it bakes a shape-derived clamp bound as a CPU tensor
        return MyPolicy(_OnGPU(fast), device="cpu")
    return make_opponent(SHORT.get(name, name), seed)


class _OnGPU:
    """MyPolicy's net on a GPU: inputs moved over, outputs back to the CPU (MyPolicy decodes on CPU)."""

    def __init__(self, net):
        self.net, self.is_temporal = net, False

    def __call__(self, *args, **kw):
        out = self.net(*(a.to("cuda") if hasattr(a, "to") else a for a in args), **kw)
        return tuple(o.cpu() if hasattr(o, "cpu") else o for o in out)

    def __getattr__(self, name):
        return getattr(self.net, name)


_POL = {}


def play(task: dict) -> dict:
    from blackout_env.env.constants import team_a_agents
    from blackout_env.train.dead_segments import batteries_in_play

    env = _ENV
    key = (task["a"], task["b"])
    if key not in _POL:  # one policy object per team, kept across matches (P11 loads once)
        _POL[key] = (_policy(task["a"], 1), _policy(task["b"], 2))
    pa, pb = _POL[key]
    for p in (pa, pb):
        if hasattr(p, "reset"):
            p.reset()
    names_a = set(team_a_agents())
    t0 = time.time()
    obs, _ = env.reset(seed=task["seed"])
    steps, last_abs, info = 0, None, {}
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            continue
        oa = {n: o for n, o in obs.items() if n in names_a}
        ob = {n: o for n, o in obs.items() if n not in names_a}
        act = {}
        if oa:
            act.update(pa.act(oa))
        if ob:
            act.update(pb.act(ob))
        obs, _, terms, _, infos = env.step(act)
        steps += 1
        if infos:
            info = next(iter(infos.values()))
        if any(terms.values()) or not obs:
            continue
        v = next(iter(obs.values()))
        a = float(v["team_state"][3])
        fired = last_abs is not None and a > last_abs + 1e-6
        last_abs = a
        if fired and batteries_in_play(v["graphic"][None], v["agent_states"][None])[0] == 0:
            break
    return {"seed": task["seed"], "steps": steps, "t0": t0, "t1": time.time(), "pid": os.getpid(), "gpu": _GPU,
            "warm": task.get("warm", False)}


def run(args, workers: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor, wait

    a, b = args.pair.split(",")
    ctx = mp.get_context("spawn")
    gpus = None
    if "cuda" in args.pair:
        gpus = ctx.Manager().Queue()
        ids = [int(g) for g in args.gpus.split(",")]
        for w in range(workers):
            gpus.put(ids[w % len(ids)])
    ex = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(args.build, args.time_scale, gpus))
    try:
        # warm-up: one short match per worker (submitted together so that every worker gets one)
        warm = [ex.submit(play, {"a": a, "b": b, "seed": 1000 + w, "warm": True}) for w in range(workers)]
        wait(warm)
        tasks = [{"a": a, "b": b, "seed": args.seed0 + i} for i in range(workers * args.per_worker)]
        load0 = os.getloadavg()[0]
        t0 = time.time()
        res = [f.result() for f in [ex.submit(play, x) for x in tasks]]
        wall = time.time() - t0
        load1 = os.getloadavg()[0]
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    steps = np.array([r["steps"] for r in res])
    per = np.array([r["t1"] - r["t0"] for r in res])
    out = {"pair": args.pair, "workers": workers, "matches": len(res), "wall_s": wall,
           "matches_per_s": len(res) / wall, "matches_per_min": 60 * len(res) / wall,
           "decisions_per_s": float(steps.sum() / wall), "decisions_per_s_per_worker": float(steps.sum() / wall / workers),
           "mean_match_s": float(per.mean()), "mean_steps": float(steps.mean()),
           "per_worker_steps_per_s": float((steps / per).mean()), "load_before": load0, "load_after": load1,
           "pids": len({r["pid"] for r in res})}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", default="build/linux_server/BlackOut.x86_64")
    p.add_argument("--pair", default="v23,v19")
    p.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--per-worker", type=int, default=3)
    p.add_argument("--seed0", type=int, default=5_000_000)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--out", default="reports/unity_throughput")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for w in args.workers:
        r = run(args, w)
        print(json.dumps(r), flush=True)
        with (out / "throughput.jsonl").open("a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
