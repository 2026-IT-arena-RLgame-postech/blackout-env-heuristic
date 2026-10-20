"""Time GRPOTrainer.update on a saved batch (examples/grpo_profile.py --save-batch), baseline vs variants.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/perf_update_bench.py /tmp/batch.npz
"""
import sys
import time

import numpy as np
import torch

from blackout_env.rl.grpo import GRPOConfig, GRPOTrainer

path = sys.argv[1]
d = dict(np.load(path))
F = "checkpoints/distill/frozen"
cfg = GRPOConfig(opponents=[f"net:{F}/p24c_v321.pt"], groups=2, k=2, heur_workers=0)
tr = GRPOTrainer(cfg)
print("rows", len(d["act"]))


def run(label, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        st = tr.update(d)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"{label:30s} {np.round(ts, 2)}  loss={st['loss']:.5f} kl={st['kl']:.5f}", flush=True)


run("baseline")
if len(sys.argv) > 2 and sys.argv[2] == "prof":
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tr.update(d); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))
