"""Map-encoder A/B (docs/rl_finetune_plan_ko.md): parameters and MACs of each encoder, and the learner
step time (forward + backward + optimizer, fp16 autocast, compiled modules, as the distill recipe) at the
training batch size on synthetic batches of the real shapes.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python examples/vitab_bench.py --batch 256 --device cuda
"""
import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from blackout_env.model.derived_obs import N_DERIVED_MAP_CHANNELS
from blackout_env.train.distill.loss import set_valued_cross_entropy
from blackout_env.train.distill.run import Learner, parse_args

ARMS = {"A": "cnn", "B1": "patch4", "B2": "patch6"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--device", default="auto")
    cli = p.parse_args()
    for arm, kind in ARMS.items():
        args = parse_args(["--build", "unused", "--policy-head", "logits", "--action-bins", "32", "--stand-action", "--angle-head",
                           "--angle-arc-bins", "1", "--amp", "fp16", "--compile", "--map-encoder", kind, "--device", cli.device])
        learner = Learner(args)
        net, dev = learner.net, learner.device
        enc = net.graphic_encoder
        n_enc = sum(p.numel() for p in enc.parameters())
        n_all = sum(p.numel() for p in net.parameters())
        # MACs of one sample's encoder forward on CPU (uncompiled copy), MACs = FLOPs / 2
        cpu_enc = type(enc)(256, map_encoder=kind).eval()
        x = torch.zeros(1, 13 + N_DERIVED_MAP_CHANNELS, 24, 24)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            cpu_enc(x)
        macs = fc.get_total_flops() / 2
        g = torch.rand(cli.batch, 13, 24, 24, device=dev).round()
        ts = torch.rand(cli.batch, 4, device=dev)
        ag = torch.rand(cli.batch, 10, 12, device=dev) * 2 - 1
        target = torch.zeros(cli.batch, 5, 33, dtype=torch.bool, device=dev)
        target[:, :, 3] = True
        w = torch.ones(cli.batch, 5, device=dev)
        ts_ms = []
        torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
        for i in range(cli.warmup + cli.steps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.float16):
                q = net(g, ts, ag, q_units=5)[0][:, :5].float()
                loss, _ = set_valued_cross_entropy(q, target, w, 1e-4)
            learner.optimizer.zero_grad(set_to_none=True)
            learner.scaler.scale(loss).backward()
            learner.scaler.unscale_(learner.optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            learner.scaler.step(learner.optimizer)
            learner.scaler.update()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if i >= cli.warmup:
                ts_ms.append(1000 * (time.perf_counter() - t0))
        ts_ms.sort()
        mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
        print(f"{arm:3s} {kind:7s} enc_params {n_enc:9d} total_params {n_all:9d} enc_MMAC {macs / 1e6:7.2f} "
              f"step_ms median {ts_ms[len(ts_ms) // 2]:6.1f} min {ts_ms[0]:6.1f} (B={cli.batch}) peak_alloc {mem:.0f} MiB", flush=True)
        del learner, net


if __name__ == "__main__":
    main()
