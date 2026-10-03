"""Offline A/B of the distillation learner (no Unity): train ``Learner.step`` for N steps on a run's
shards and score held-out shards, to check that a faster configuration (CUDA fp16 + whole-net
compile, docs/cuda_training_ko.md) learns the same as the reference (eager fp32).

The last ``--heldout-frac`` of the shards (by name = collection order) are held out; the ring holds
the newest ``--rows`` of the rest.  Batches come from a seeded RNG, so two configurations see the
same rows.  Every flag after ``--`` goes to ``train/distill/run.py``'s parser (the learner recipe).

  PYTHONPATH=. python examples/distill_offline_ab.py checkpoints/distill/p11/shards --steps 3000 \\
      --init checkpoints/distill/frozen/p10_v22.pt --out reports/cuda/distill_ab_fast.json -- \\
      --policy-head logits --action-bins 32 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 \\
      --opening-weight 1.5 --opening-seconds 10 --lr 1e-4 --amp fp16 --compile
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from blackout_env.train.distill.dataset import DistillDataset, new_shards
from blackout_env.train.distill.metrics import LabelAccuracy
from blackout_env.train.distill.run import Learner, load_weights, parse_args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    argv = sys.argv[1:]
    learner_argv = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("shards", type=Path)
    p.add_argument("--rows", type=int, default=600_000)
    p.add_argument("--heldout-frac", type=float, default=0.05)
    p.add_argument("--heldout-rows", type=int, default=40_000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    cli = p.parse_args(argv)

    args = parse_args(["--build", "unused", *learner_argv])
    torch.manual_seed(cli.seed)
    learner = Learner(args)
    if cli.init is not None:
        load_weights(learner.net, torch.load(cli.init, map_location="cpu", weights_only=True)["policy_state"])
        learner.mark_weights()
    device = learner.device

    names = sorted(q.name for q in cli.shards.glob("*.npz") if not q.name.endswith(".tmp.npz"))
    n_hold = max(1, int(len(names) * cli.heldout_frac))
    train_names, hold_names = names[:-n_hold], names[-n_hold:]
    t0 = time.time()
    ring = DistillDataset(cli.shards, cli.rows, args.action_bins)
    for rows, meta in new_shards(cli.shards, set(hold_names), args.action_bins):
        ring.append(rows, meta.get("student_version", 0))
    held: dict[str, list] = {}
    for rows, _ in new_shards(cli.shards, set(train_names), args.action_bins):
        for k, v in rows.items():
            held.setdefault(k, []).append(v)
    held = {k: np.concatenate(v) for k, v in held.items()}
    keep = np.sort(np.random.default_rng(1).choice(len(held["labels"]), min(cli.heldout_rows, len(held["labels"])),
                                                   replace=False))
    held = {k: v[keep] for k, v in held.items()}
    print(f"[ab] ring {ring.size} rows from {len(train_names)} shards, held-out {len(keep)} rows from {n_hold} shards "
          f"({time.time() - t0:.0f}s); device {device}, amp {args.amp}, compile {args.compile}, batch {args.batch_size}",
          flush=True)

    rng = np.random.default_rng(cli.seed)
    log = []
    t_win = time.time()
    for step in range(1, cli.steps + 1):
        learner.set_lr(step)
        learner.step(ring.sample(args.batch_size, rng))
        if step % cli.log_every == 0 or step == cli.steps:
            _sync(device)
            now = time.time()
            stats = learner.pop_stats()
            row = {"step": step, "ms_step": 1000 * (now - t_win) / cli.log_every,
                   **{k.split("/", 1)[1]: v for k, v in stats.items() if k.split("/", 1)[1] in
                      ("ce", "angle_loss", "accuracy", "grad_norm", "amp_skipped", "loss_scale")}}
            log.append(row)
            print(json.dumps(row), flush=True)
            t_win = time.time()
    acc = LabelAccuracy(device)
    learner.score(held, acc)
    result = {k: v for k, v in acc.pop("heldout").items() if "share" not in k}
    ms = sorted(r["ms_step"] for r in log[1:]) or [log[0]["ms_step"]]
    summary = {"device": str(device), "torch": torch.__version__, "amp": args.amp, "compile": args.compile,
               "batch": args.batch_size, "steps": cli.steps, "median_ms_step": ms[len(ms) // 2],
               "rows_per_s": args.batch_size * 1000 / ms[len(ms) // 2], "heldout": result, "log": log}
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=1), flush=True)
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
