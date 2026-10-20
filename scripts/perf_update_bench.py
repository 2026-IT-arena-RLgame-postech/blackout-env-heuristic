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

# fp16 autocast variant: speed and log-prob difference vs fp32 on the same batch
import copy
from torch.nn import functional as F
n = 4096
g, t, a = (torch.from_numpy(d[k][:n]).cuda() for k in "gta")
act = torch.from_numpy(d["act"][:n]).cuda().long()
def lp(fp16):
    tr.cfg.update_fp16 = fp16
    with torch.no_grad():
        return F.log_softmax(tr._logits(tr.own, g[:1024], t[:1024], a[:1024]), -1)
x, y = lp(False), lp(True)
pa = x.exp()
print("fp16 vs fp32 update forward: max|dlogp|=%.4f  mean|dlogp|=%.5f  argmax agree=%.5f  KL(fp32||fp16)=%.2e" % (
    (x - y).abs().max(), (x - y).abs().mean(), (x.argmax(-1) == y.argmax(-1)).float().mean(), (pa * (x - y)).sum(-1).mean()))
tr.cfg.update_fp16 = True
run("update_fp16", 2)
