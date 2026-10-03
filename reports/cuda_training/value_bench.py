"""Value-net (MyModel trunk) training step on CUDA: precision x compile scope x batch (server scratch)."""
import sys, time, itertools
import numpy as np
import torch
from blackout_env.train.accel import adamw, autocast, grad_scaler, pick_device, sync, to_device, pinned
from blackout_env.value.data import open_cache
from blackout_env.value.model import TRUNK_MODULES, ValueNet, value_loss
from blackout_env.value.train import gather

device = pick_device("auto")
cache = open_cache("datasets/value_s1_sub4")
rows = np.flatnonzero(np.asarray(cache["split"]) == 0)
rng = np.random.default_rng(0)
init = torch.load("checkpoints/distill/frozen/p5a_v22.pt", map_location="cpu", weights_only=True)["policy_state"]
batches = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 else [256]
confs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["none:0", "fp16:0", "fp16:trunk", "fp16:all"]
steps = 60
for batch, conf in itertools.product(batches, confs):
    amp, comp = conf.split(":")
    torch.manual_seed(0)
    net = ValueNet().to(device)
    net.load_trunk(init)
    if comp == "trunk":
        for n in TRUNK_MODULES:
            getattr(net.trunk, n).compile()
    elif comp in ("all", "cg"):
        net.compile(mode="reduce-overhead" if comp == "cg" else None)
    opt = adamw([{"params": list(net.trunk_parameters())}, {"params": net.head_parameters()}], device, lr=1e-4, weight_decay=0.01)
    scaler = grad_scaler(device, amp)
    t = {"host": 0.0, "h2d": 0.0, "step": 0.0}
    for i in range(20 + steps):
        t0 = time.perf_counter()
        host = {k: pinned(v, device) for k, v in gather(cache, rows[rng.integers(0, len(rows), batch)]).items()}
        t1 = time.perf_counter()
        b = to_device(host, device); sync(device)
        t2 = time.perf_counter()
        with autocast(device, amp):
            out = net.forward_u8(b["graphic"], b["team_state"], b["agent_states"])
        loss, stats = value_loss(out, b, b["team_state"][:, 0] - b["team_state"][:, 1])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sync(device)
        t3 = time.perf_counter()
        if i >= 20:
            for k, v in zip(t, (t1 - t0, t2 - t1, t3 - t2)):
                t[k] += v
    ms = {k: 1000 * v / steps for k, v in t.items()}
    print(f"batch {batch:5d} {conf:>11s}: host {ms['host']:6.2f} h2d {ms['h2d']:5.2f} step {ms['step']:6.2f} ms -> "
          f"{batch * 1000 / ms['step']:7.0f} rows/s (device step only)", flush=True)
