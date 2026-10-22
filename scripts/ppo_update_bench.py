"""Time PPOTrainer.update (critic part and policy part) on a saved batch (examples/ppo_profile.py --save-batch).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/ppo_update_bench.py /tmp/ppo_batch.npz [prof] [--set k=v ...]
"""
import sys
import time

import numpy as np
import torch

from blackout_env.rl.ppo import PPOConfig, PPOTrainer

args = sys.argv[1:]
path = args.pop(0)
prof = "prof" in args
sets = args[args.index("--set") + 1:] if "--set" in args else []
d = dict(np.load(path))
F = "checkpoints/distill/frozen"
cfg = PPOConfig(opponents=[f"net:{F}/p24c_v321.pt"], games=2, heur_workers=0, critic_warmup=0)
for it in sets:
    k, v = it.split("=", 1)
    cur = getattr(cfg, k)
    setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
tr = PPOTrainer(cfg)
print("rows", len(d["act"]), flush=True)
for i in range(3):
    torch.manual_seed(i)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    st = tr.update(d)
    torch.cuda.synchronize()
    print(f"update {time.perf_counter() - t0:.2f}s vf={st['vf_loss']:.5f} loss={st['loss']:.5f} kl={st['kl']:.5f}", flush=True)
if prof:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        tr.update(d); torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=50))
