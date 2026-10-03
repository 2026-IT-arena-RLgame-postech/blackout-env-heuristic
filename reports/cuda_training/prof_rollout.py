"""torch.profiler over the compiled fp16 RolloutNet training step (server scratch)."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, "examples")
import numpy as np
import torch
import rollout_net_train as r
from blackout_env.sim.rollout_net import RolloutNet
from blackout_env.train.accel import adamw, grad_scaler, pick_device, sync, to_device

amp = sys.argv[1] if len(sys.argv) > 1 else "fp16"
comp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
mode = sys.argv[3] if len(sys.argv) > 3 else "default"
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
cl = int(sys.argv[5]) if len(sys.argv) > 5 else 0
device = pick_device("auto")
train = r.load_split("checkpoints/rollout_net/s3data_x1/train.npz")
pool = r.labelled_rows(train["labels"])
keys = ["map", "team_state", "agent_states", "labels", "angles", "tick", "p11_q", "p11_off"]


class A:
    opening_weight = 1.5; angle_weight = 1.0; p11_kl = 0.5; channels_last = bool(cl)


net = RolloutNet.from_checkpoint("checkpoints/rollout_net/r5.pt").train().to(device)
if cl:
    net = net.to(memory_format=torch.channels_last)
opt = adamw(net.parameters(), device, lr=1e-4, weight_decay=1e-4)
scaler = grad_scaler(device, amp)
step_fn = r.train_step_module(torch)(net, A, amp, device)
if comp:
    step_fn = torch.compile(step_fn, mode=None if mode == "default" else mode)
src = r.BatchSource(train, None, pool, None, batch, 0.0, 0, keys)
bs = [to_device(src(), device) for _ in range(8)]


import os, contextlib
from torch.nn.attention import SDPBackend, sdpa_kernel
ctx = (lambda: sdpa_kernel(SDPBackend.MATH)) if os.environ.get("MATH") else contextlib.nullcontext


def step(b):
    with ctx():
        loss, stats = step_fn(b["map"], b["team_state"], b["agent_states"], b["labels"], b["angles"], b["tick"], b["p11_q"], b["p11_off"])
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()


for i in range(30):
    step(bs[i % 8])
sync(device)
t = time.perf_counter()
for i in range(100):
    step(bs[i % 8])
sync(device)
print(f"amp {amp} compile {comp} mode {mode} batch {batch} cl {cl}: {1000 * (time.perf_counter() - t) / 100:.2f} ms/step (data resident)")
if "--prof" in sys.argv:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for i in range(10):
            step(bs[i % 8])
        sync(device)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
