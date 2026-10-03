"""torchrun --nproc_per_node N ddp_bench.py [compile] [totals] [prefetch]: per-part DDP step timing."""
import os, sys, time
sys.path.insert(0, "examples")
import numpy as np
import torch
import rollout_net_train as r
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.train.accel import Dist, Prefetcher, adamw, grad_scaler, pick_device, sync, to_device

comp, use_totals, pre = (int(x) for x in sys.argv[1:4])
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
device = pick_device("auto")
dist = Dist(device)
train = r.load_split("checkpoints/rollout_net/s3data_x1/train.npz")
pool = r.labelled_rows(train["labels"])
keys = ["map", "team_state", "agent_states", "labels", "angles", "tick"]


class A:
    opening_weight = 1.5; angle_weight = 1.0; p11_kl = 0.0; channels_last = True


torch.manual_seed(0)
net = RolloutNet.from_checkpoint("checkpoints/rollout_net/r5.pt").train().to(device).to(memory_format=torch.channels_last)
opt = adamw(net.parameters(), device, lr=1e-4, weight_decay=1e-4)
scaler = grad_scaler(device, "fp16")
step_fn = dist.wrap(r.train_step_module(torch)(net, A, "fp16", device))
if comp:
    step_fn = torch.compile(step_fn)
src = r.BatchSource(train, None, pool, None, batch // dist.world, 0.0, dist.rank, keys)
pf = Prefetcher(src, device) if pre else None
T = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
N, W = 200, 30
for i in range(W + N):
    t0 = time.perf_counter()
    b = to_device(pf.next() if pf else src(), device)
    ct, at = r.global_totals(b, A, dist, torch) if use_totals else (None, None)
    sync(device); t1 = time.perf_counter()
    loss, stats = step_fn(b["map"], b["team_state"], b["agent_states"], b["labels"], b["angles"], b["tick"], None, None, ct, at)
    sync(device); t2 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    sync(device); t3 = time.perf_counter()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sync(device); t4 = time.perf_counter()
    if i >= W:
        for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            T[k] += v
if dist.main:
    ms = {k: round(1000 * v / N, 2) for k, v in T.items()}
    tot = sum(ms.values())
    print(f"world {dist.world} compile {comp} totals {use_totals} prefetch {pre} batch {batch}: {ms} total {tot:.2f} ms "
          f"-> {batch * 1000 / tot:.0f} rows/s  OMP {os.environ.get('OMP_NUM_THREADS')}", flush=True)
if pf:
    pf.stop()
dist.close()
