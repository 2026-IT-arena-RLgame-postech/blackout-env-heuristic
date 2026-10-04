"""Offline A/B of the distillation learner (no Unity): train ``Learner.step`` for N steps on a run's
shards and score held-out shards, to check that a faster configuration (CUDA fp16 + whole-net
compile, docs/cuda_training_ko.md) learns the same as the reference (eager fp32).

The last ``--heldout-frac`` of the shards (by name = collection order) are held out; the ring holds
the newest ``--rows`` of the rest.  Batches come from a seeded RNG, so two configurations see the
same rows.  Every flag after ``--`` goes to ``train/distill/run.py``'s parser (the learner recipe).
The held-out score is reported overall and split by seconds of play (``--splits``: the first 10 s
= the race, after 30 s = late play); ``--save`` writes the trained net as a distill checkpoint
(``policy_state``), e.g. for the simulator strength runs (docs §25: P11 vs P11 + map tokens).
``--clean-heldout DIR`` also scores every shard in DIR (e.g. the Unity held-out eval's saved shards:
fresh seeds no run trained on -- a run's own last shards were in its ring); ``--steps 0`` only scores.

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
from blackout_env.train.distill.metrics import MATCH_SECONDS, LabelAccuracy
from blackout_env.train.distill.run import Learner, load_weights, parse_args

SPLITS = {"all": (0.0, 1e9), "t0_10s": (0.0, 10.0), "t10_30s": (10.0, 30.0), "t30s_plus": (30.0, 1e9)}


def score_splits(learner, held: dict, elapsed: np.ndarray) -> dict:
    out = {}
    for name, (lo, hi) in SPLITS.items():
        keep = (elapsed >= lo) & (elapsed < hi)
        if not keep.any():
            continue
        acc = LabelAccuracy(learner.device)
        learner.score({k: v[keep] for k, v in held.items()}, acc)
        out[name] = {k.split("/", 1)[1]: v for k, v in acc.pop("heldout").items() if "share" not in k}
        out[name]["rows"] = int(keep.sum())
    return out


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
    p.add_argument("--heldout-rows", type=int, default=40_000, help="<= 0: every held-out row")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--save", type=Path, default=None, help="write the trained net here (policy_state + distill meta)")
    p.add_argument("--eval-every", type=int, default=0, help="also score the held-out splits every N steps")
    p.add_argument("--clean-heldout", type=Path, default=None, help="also score every shard in this directory")
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
    n_rows = len(held["labels"]) if cli.heldout_rows <= 0 else min(cli.heldout_rows, len(held["labels"]))
    keep = np.sort(np.random.default_rng(1).choice(len(held["labels"]), n_rows, replace=False))
    held = {k: v[keep] for k, v in held.items()}
    elapsed = (1.0 - held["team_state"][:, 2].astype(np.float64)) * MATCH_SECONDS
    clean = None
    if cli.clean_heldout is not None:
        parts: dict[str, list] = {}
        for rows, _ in new_shards(cli.clean_heldout, set(), args.action_bins):
            for k, v in rows.items():
                parts.setdefault(k, []).append(v)
        clean = {k: np.concatenate(v) for k, v in parts.items()}
        clean_elapsed = (1.0 - clean["team_state"][:, 2].astype(np.float64)) * MATCH_SECONDS
        print(f"[ab] clean held-out {len(clean['labels'])} rows from {cli.clean_heldout}", flush=True)

    def evaluate(step):
        e = {"step": step, **score_splits(learner, held, elapsed)}
        if clean is not None:
            e["clean"] = score_splits(learner, clean, clean_elapsed)
        return e

    print(f"[ab] ring {ring.size} rows from {len(train_names)} shards, held-out {len(keep)} rows from {n_hold} shards "
          f"({time.time() - t0:.0f}s); device {device}, amp {args.amp}, compile {args.compile}, batch {args.batch_size}",
          flush=True)

    rng = np.random.default_rng(cli.seed)
    log = []
    evals = []
    if cli.eval_every:
        evals.append(evaluate(0))
        print(json.dumps({"eval": evals[-1]}), flush=True)
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
            if cli.eval_every and step % cli.eval_every == 0 and step != cli.steps:
                evals.append(evaluate(step))
                print(json.dumps({"eval": evals[-1]}), flush=True)
            t_win = time.time()
    final = evaluate(cli.steps)
    splits = {k: v for k, v in final.items() if k not in ("step", "clean")}
    result = {k: v for k, v in splits["all"].items() if k != "rows"}
    ms = sorted(r["ms_step"] for r in log[1:]) or [log[0]["ms_step"] if log else float("nan")]
    summary = {"device": str(device), "torch": torch.__version__, "amp": args.amp, "compile": args.compile,
               "batch": args.batch_size, "steps": cli.steps, "median_ms_step": ms[len(ms) // 2],
               "rows_per_s": args.batch_size * 1000 / ms[len(ms) // 2], "heldout": result, "heldout_splits": splits, "clean_splits": final.get("clean"),
               "params": sum(p.numel() for p in learner.net.parameters()), "map_xattn": args.map_xattn,
               "map_xattn_layers": args.map_xattn_layers, "init": str(cli.init), "seed": cli.seed,
               "learner_argv": learner_argv, "evals": evals, "log": log}
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=1), flush=True)
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(summary, indent=1))
    if cli.save:
        cli.save.parent.mkdir(parents=True, exist_ok=True)
        meta = {"offline_ab": True, "init": str(cli.init), "steps": cli.steps, "action_bins": args.action_bins,
                "policy_head": args.policy_head, "angle_head": args.angle_head, "angle_arc_bins": args.angle_arc_bins,
                "map_xattn": args.map_xattn, "map_xattn_layers": args.map_xattn_layers, "lr": args.lr,
                "stand_action": args.stand_action, "temporal": args.temporal}
        torch.save({"policy_state": learner.net.state_dict(), "distill": meta}, cli.save)


if __name__ == "__main__":
    main()
