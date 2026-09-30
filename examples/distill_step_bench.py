"""Where a distillation learner step spends its time, and what batch / precision / quantile
settings buy.

Loads the newest ``--rows`` rows of a run's shards into a DistillDataset and times, per
configuration, each part of ``Learner.step`` with device syncs between them: sampling from the
ring, host->device copy, the derived observation features, the rest of the forward, backward,
the optimizer, and the running statistics.  Reports ms per step and rows per second.

  python examples/distill_step_bench.py checkpoints/distill/d4/shards --batch-sizes 256 512 1024 \\
      --quantiles 32 8 --amp none fp16

Run it with the GPU otherwise idle: a training run sharing it distorts every number.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch

from blackout_env.model.derived_obs import derived_map_channels, local_wall_features
from blackout_env.train.distill.dataset import DistillDataset, new_shards
from blackout_env.train.distill.loss import set_valued_cross_entropy
from blackout_env.train.distill.run import Learner, class_weights, parse_args

PARTS = ("sample", "to_device", "derived", "forward", "backward", "optimizer", "stats")


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _load(root: Path, rows: int) -> DistillDataset:
    dataset = DistillDataset(root, rows)
    paths = sorted(root.glob("*.npz"))
    seen = {p.name for p in paths}
    kept, total = [], 0
    for path in reversed(paths):  # the newest shards, enough to fill the ring
        seen.discard(path.name)
        kept.append(path.name)
        total += int(np.load(path, mmap_mode="r")["labels"].shape[0])
        if total >= rows:
            break
    for shard_rows, meta in new_shards(root, seen):
        dataset.append(shard_rows, meta.get("student_version", 0))
    return dataset


def bench(learner: Learner, dataset: DistillDataset, batch: int, quantiles: int, amp: str, steps: int, warmup: int,
          q_units: int = 5) -> dict:
    device, args = learner.device, learner.args
    rng = np.random.default_rng(0)
    autocast = (torch.autocast(device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[amp])
                if amp != "none" else contextlib.nullcontext())
    times = dict.fromkeys(PARTS, 0.0)
    net = learner.net.train()
    for i in range(warmup + steps):
        marks = [time.perf_counter()]

        def mark():
            _sync(device)
            marks.append(time.perf_counter())

        rows = dataset.sample(batch, rng)
        mark()
        graphic, team_state, agent_states, masks, classes, kinds = learner._tensors(rows)
        mark()
        with autocast:
            # the same features MyModel.forward derives first, timed on their own
            local_wall_features(graphic, agent_states)
            derived_map_channels(graphic, agent_states)
            mark()
            q_values, *_ = net(graphic, team_state, agent_states, n_quantiles=quantiles, q_units=q_units)
            logits = q_values[:, :5].float()
            valid = masks.any(dim=-1).float()
            loss, stats = set_valued_cross_entropy(logits, masks, class_weights(classes, valid), args.z_loss)
        mark()
        learner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        mark()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        learner.optimizer.step()
        mark()
        with torch.no_grad():
            learner.accuracy.add(logits, masks, classes, kinds)
            log_p = torch.log_softmax(logits, dim=-1)
            (-(log_p.exp() * log_p).sum(-1) * valid).sum()
        mark()
        if i >= warmup:
            for part, (a, b) in zip(PARTS, zip(marks, marks[1:])):
                times[part] += b - a
    # the forward timed above includes the derived features again (MyModel.forward derives them)
    ms = {part: 1000 * t / steps for part, t in times.items()}
    ms["forward"] = max(0.0, ms["forward"] - ms["derived"])
    total = sum(ms.values())
    return {"batch": batch, "quantiles": quantiles, "amp": amp, "ms": ms, "total_ms": total, "rows_per_s": batch * 1000 / total}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("shards", type=Path)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--quantiles", type=int, nargs="+", default=[32])
    p.add_argument("--amp", nargs="+", choices=("none", "fp16", "bf16"), default=["none"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--q-units", type=int, nargs="+", default=[5], help="unit rows through the Q head (10 = before 2026-10)")
    p.add_argument("--device", default=None)
    cli = p.parse_args()

    args = parse_args(["--build", "unused"] + (["--device", cli.device] if cli.device else []))
    learner = Learner(args)
    t = time.time()
    dataset = _load(cli.shards, cli.rows)
    print(f"[bench] {dataset.size} rows loaded in {time.time() - t:.0f}s, device {learner.device}, torch {torch.__version__}")
    print(f"{'units':>5s} {'batch':>5s} {'quant':>5s} {'amp':>4s} " + " ".join(f"{part:>9s}" for part in PARTS) + f" {'total':>7s} {'rows/s':>7s} {'step/s':>6s}")
    for q_units in cli.q_units:
      for amp in cli.amp:
        for quantiles in cli.quantiles:
            for batch in cli.batch_sizes:
                try:
                    r = bench(learner, dataset, batch, quantiles, amp, cli.steps, cli.warmup, q_units)
                except RuntimeError as exc:  # e.g. an op without an fp16 MPS kernel
                    print(f"{q_units:5d} {batch:5d} {quantiles:5d} {amp:>4s} failed: {str(exc).splitlines()[0]}")
                    continue
                print(f"{q_units:5d} {batch:5d} {quantiles:5d} {amp:>4s} " + " ".join(f"{r['ms'][part]:9.1f}" for part in PARTS)
                      + f" {r['total_ms']:7.1f} {r['rows_per_s']:7.0f} {1000 / r['total_ms']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
