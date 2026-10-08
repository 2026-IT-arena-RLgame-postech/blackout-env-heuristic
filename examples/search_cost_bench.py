"""Search cost of two rollout nets, interleaved A/B (docs §24): one ``Searcher.search`` over R roots
(B8 branches, 25 + 125 decisions, V_sym leaf), for the server configuration (CUDA graphs, packed
maps) at small R and the simulator lockstep configuration (eager, float maps) at R = 128.  Runs
alternate between the nets every repetition, so a shared GPU's load drift hits both alike.

    PYTHONPATH=. python examples/search_cost_bench.py --out reports/s3_newnet/cost.json \\
        --pair r5=checkpoints/rollout_net/r5.pt,checkpoints/value/frozen/s2v_iter5.pt \\
        --pair new=checkpoints/rollout_net/arch/f3_x_d2.pt,checkpoints/value/s2v_f3xd2/iter2.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True, help="name=rollout_ckpt,value_ckpt")
    p.add_argument("--server-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--lockstep", type=int, default=128)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import numba
    import torch

    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches
    from blackout_env.sim.search_server import build_searcher

    torch.set_num_threads(args.threads)
    numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
    pairs = {}
    for spec in args.pair:
        name, rest = spec.split("=", 1)
        pairs[name] = rest.split(",")
    # roots: r5 x r5 games 10-40 s in (the same roots for every net)
    r5 = NetPolicy(RolloutNet.from_checkpoint("checkpoints/rollout_net/r5.pt"), device="cuda")
    n = max(args.lockstep, max(args.server_sizes))
    roots = BatchRollout(r5, r5).run([SimState.from_seed(4_100_000 + k) for k in range(n)], 500).states()
    roots = [s for s in roots if not s.over][:n]
    teams = [k % 2 for k in range(len(roots))]
    seeds = list(range(len(roots)))
    built = {}
    for name, (roll, val) in pairs.items():
        server = build_searcher(rollout=roll, value=val, max_roots=max(args.server_sizes))
        lock = Searcher(ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(roll), device="cuda")), ValueNetLeaf(val, "cuda"),
                        parse_branches("alt*,samp:1.0x2"), window=25, margin=3.0)
        for r in args.server_sizes:  # warm-up: graphs, numba
            server.search(roots[:r], teams[:r], seeds[:r])
        lock.search(roots[:8], teams[:8], seeds[:8])
        built[name] = (server, lock)
    res = {name: {f"server_{r}": [] for r in args.server_sizes} | {f"lockstep_{args.lockstep}": []} for name in pairs}
    for rep in range(args.reps):
        for name, (server, lock) in built.items():
            for r in args.server_sizes:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                server.search(roots[:r], teams[:r], seeds[:r])
                res[name][f"server_{r}"].append(time.perf_counter() - t0)
            r = min(args.lockstep, len(roots))
            t0 = time.perf_counter()
            lock.search(roots[:r], teams[:r], seeds[:r])
            res[name][f"lockstep_{args.lockstep}"].append(time.perf_counter() - t0)
        print(rep, json.dumps({k: {kk: round(vv[-1], 3) for kk, vv in v.items()} for k, v in res.items()}), flush=True)
    summary = {name: {k: {"median_s": float(np.median(v)), "per_root_ms": 1000 * float(np.median(v)) / int(k.split("_")[1])}
                      for k, v in d.items()} for name, d in res.items()}
    names = list(pairs)
    if len(names) == 2:
        a, b = names
        summary["ratio"] = {k: summary[b][k]["median_s"] / summary[a][k]["median_s"] for k in summary[a]}
    summary["roots"] = len(roots)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "raw": res}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
