"""Search server microbenchmarks and parity on real match roots (docs §23).

* ``latency``: one server-style Searcher (bucketed CUDA graphs), batches of R roots -> ms per batch,
  ms per root (R = 1 also with the per-worker setup: exact-size graphs, 1 thread).
* ``parity``: roots from r5 x r5 matches (seeds 4000000+, every 5 s of the first 60 s, both
  teams); each root searched alone in-process (the §21 worker path: exact graphs, batch 1 root)
  and through a real server subprocess, all roots sent concurrently from client threads (the
  batched path).  Reports chosen-branch agreement and leaf-value differences.

    PYTHONPATH=. python examples/search_server_bench.py latency --out reports/search_server/bench
    PYTHONPATH=. python examples/search_server_bench.py parity --roots 96 --out reports/search_server/parity
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"


def match_roots(n: int, seed0: int = 4_000_000, every_s: int = 5, upto_s: int = 60):
    """States of r5 x r5 matches at every ``every_s`` seconds (first ``upto_s``), teams alternating."""
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    pol = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device="cpu")
    roll = BatchRollout(pol, pol)
    per = upto_s // every_s
    seeds = list(range(seed0, seed0 + -(-n // per)))
    states = [SimState.from_seed(s) for s in seeds]
    out = []
    for k in range(per):
        res = roll.run(states, every_s * 25)
        states = res.states()
        out += [s for s in states if not s.over]
    out = out[:n]
    return out, [j % 2 for j in range(len(out))], [1000 + j for j in range(len(out))]


def cmd_latency(args):
    import numba
    import torch

    from blackout_env.sim import SimState
    from blackout_env.sim.search_server import build_searcher

    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    roots, teams, reseeds = match_roots(max(args.sizes))
    rows = []
    s = build_searcher(device=args.device, max_roots=max(args.sizes), repo=str(REPO))
    for r in args.sizes:
        s.search(roots[:r], teams[:r], reseeds[:r])  # warm (graphs for this size's buckets)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            s.search(roots[:r], teams[:r], reseeds[:r])
            ts.append(time.perf_counter() - t0)
        ms = np.asarray(ts) * 1e3
        row = {"setup": f"server(threads {args.threads})", "roots": r, "ms_batch": float(ms.mean()),
               "ms_p95": float(np.percentile(ms, 95)), "ms_root": float(ms.mean() / r)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    # the per-worker path: exact-size graphs, one torch / numba thread
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches

    torch.set_num_threads(1)
    numba.set_num_threads(1)
    w = Searcher(ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=args.device, cuda_graphs=True)),
                 ValueNetLeaf(str(REPO / VALUE), args.device), parse_branches("alt*,samp:1.0x2"), margin=3.0)
    w.search([SimState.from_seed(1)], [0], [1])
    ts = []
    for j in range(args.reps * 4):
        t0 = time.perf_counter()
        w.search([roots[j % len(roots)]], [teams[j % len(roots)]], [reseeds[j % len(roots)]])
        ts.append(time.perf_counter() - t0)
    ms = np.asarray(ts) * 1e3
    row = {"setup": "per-worker (1 thread, exact graphs)", "roots": 1, "ms_batch": float(ms.mean()),
           "ms_p95": float(np.percentile(ms, 95)), "ms_root": float(ms.mean())}
    rows.append(row)
    print(json.dumps(row), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "latency.jsonl").open("a") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "device": args.device, "load1": os.getloadavg()[0]}) + "\n")


def cmd_parity(args):
    import numba
    import torch

    from blackout_env.sim import SimState
    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, choose, parse_branches
    from blackout_env.sim.search_server import SearchClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots, teams, reseeds = match_roots(args.roots)
    path = f"/tmp/bo-parity-{os.getpid()}.sock"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    srv = subprocess.Popen([sys.executable, "-m", "blackout_env.sim.search_server", "serve", "--socket", path,
                            "--repo", str(REPO), "--device", args.device, "--max-roots", str(args.max_roots),
                            "--parent-pid", str(os.getpid()), "--max-wait-ms", "20"],
                           cwd=REPO, env=env, stdout=(out / "server.log").open("w"), stderr=subprocess.STDOUT)
    try:
        torch.set_num_threads(1)
        numba.set_num_threads(1)
        w = Searcher(ModNetPolicy(NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=args.device,
                                            cuda_graphs=args.device == "cuda")),
                     ValueNetLeaf(str(REPO / VALUE), args.device), parse_branches("alt*,samp:1.0x2"), margin=3.0)
        w.search([SimState.from_seed(1)], [0], [1])
        ref = [w.search([r.copy()], [t], [k]).values[0] for r, t, k in zip(roots, teams, reseeds)]
        got = [None] * len(roots)
        nthreads = args.clients

        def client(c):
            cli = SearchClient(path, expect=w.config(), connect_timeout=600)
            for j in range(c, len(roots), nthreads):
                rep = cli.search([roots[j]], [teams[j]], [reseeds[j]], margin=3.0)
                got[j] = (rep.values[0], int(rep.choice[0]), rep.batch_roots)
            cli.close()

        ts = [threading.Thread(target=client, args=(c,)) for c in range(nthreads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        srv.terminate()
        srv.wait(30)
    ref = np.stack(ref)
    gv = np.stack([g[0] for g in got])
    rc, gc = choose(ref, 3.0), np.asarray([g[1] for g in got])
    d = np.abs(gv - ref)
    rank_same = (ref.argmax(1) == gv.argmax(1)).mean()
    res = {"roots": len(roots), "choice_agree": float((rc == gc).mean()), "argmax_agree": float(rank_same),
           "bit_exact_roots": int((d.max(1) == 0).sum()), "abs_diff_mean": float(d.mean()),
           "abs_diff_p95": float(np.percentile(d, 95)), "abs_diff_max": float(d.max()),
           "batch_roots_mean": float(np.mean([g[2] for g in got])),
           "base_share_ref": float((rc == 0).mean()), "base_share_server": float((gc == 0).mean()),
           "corr": float(np.corrcoef(ref.ravel(), gv.ravel())[0, 1])}
    # how close to the margin were the disagreements?
    gain = ref.max(1) - ref[:, 0]
    res["disagree_ref_gain"] = [float(x) for x in gain[rc != gc]]
    print(json.dumps(res, indent=1))
    (out / "parity.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "parity_values.npz", ref=ref, server=gv)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("latency")
    q.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32])
    q.add_argument("--reps", type=int, default=5)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--device", default="cuda")
    q.add_argument("--out", required=True)
    q = sub.add_parser("parity")
    q.add_argument("--roots", type=int, default=96)
    q.add_argument("--clients", type=int, default=12)
    q.add_argument("--max-roots", type=int, default=32)
    q.add_argument("--device", default="cuda")
    q.add_argument("--out", required=True)
    args = p.parse_args()
    {"latency": cmd_latency, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()
