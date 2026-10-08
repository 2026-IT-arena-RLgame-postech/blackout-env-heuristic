"""R1 latency (docs §34): a server-style Searcher (P14b base, bucketed CUDA graphs) on real mid-match roots,
ms per batch of R roots for det / o4f with the f3_x_d2 or v25d256 rollout net.  Run on an idle GPU:
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python reports/blind_search/scripts/latency_bench.py > reports/blind_search/latency.json"""
import json
import sys
import time

import numpy as np
import torch

from blackout_env.sim import SimState
from blackout_env.sim.search_server import build_searcher

torch.set_num_threads(4)
P14B = "checkpoints/distill/frozen/p14_v25.pt"
V = "checkpoints/value/s2v_f3xd2/iter2.pt"
O4 = {"opp_branches": "samp:1.0x3", "opp_agg": "mean", "opp_left": 150}
NETS = {"f3": "checkpoints/rollout_net/arch/f3_x_d2.pt", "v25r": "checkpoints/rollout_net/v25/v25d256.pt"}
roots = []
for k in range(8):
    s = SimState.from_seed(4_000_000 + k)
    rng = np.random.default_rng(k)
    for _ in range(250 + 25 * k):
        s.step(rng.normal(size=(10, 2)).astype(np.float32).clip(-1, 1) * 0.5)
    roots.append(s)
out = {}
for net in ("f3", "v25r"):
    for name, deep in (("det", None), ("o4f", O4)):
        s = build_searcher(NETS[net], V, "alt*,samp:1.0x2", 25, 125, "cuda", True, 8, deep=deep, base=P14B)
        for r in (1, 4, 8):
            s.search([x.copy() for x in roots[:r]], [k % 2 for k in range(r)], list(range(r)))  # warm-up
            ts = []
            for rep in range(5):
                t0 = time.perf_counter()
                s.search([x.copy() for x in roots[:r]], [k % 2 for k in range(r)], [rep * 10 + k for k in range(r)])
                ts.append(time.perf_counter() - t0)
            out[f"{name}_{net}_R{r}"] = {"ms_median": 1e3 * float(np.median(ts)), "ms_per_root": 1e3 * float(np.median(ts)) / r}
            print(f"{name}_{net} R={r}: {out[f'{name}_{net}_R{r}']}", file=sys.stderr, flush=True)
        del s
        torch.cuda.empty_cache()
print(json.dumps(out, indent=1))
