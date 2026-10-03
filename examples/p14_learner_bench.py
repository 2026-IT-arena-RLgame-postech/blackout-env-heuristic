"""P14: ms per real ``Learner.step`` (P11 recipe: logits K=32 + angle head, opening weight) per
precision / compile / TF32 / batch setting, on a CUDA card (the RTX 3060 Laptop of ``wsl_gpu``).

Unlike ``distill_step_bench.py`` this times the learner's own ``step`` (angle loss, GradScaler,
statistics included), end to end with a device sync per step, and reports peak VRAM.  Each
configuration runs in a fresh process (compile and TF32 flags are global).

    PYTHONPATH=. .venv/bin/python examples/p14_learner_bench.py <shards dir> --configs \\
        none:0:0:256 fp16:1:0:256 bf16:1:0:256 none:1:1:256 bf16:1:1:512
(config = amp:compile:tf32:batch)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

P11_FLAGS = ["--policy-head", "logits", "--action-bins", "32", "--act-arc-bins", "1", "--angle-head",
             "--angle-arc-bins", "1", "--angle-loss-weight", "1.0", "--opening-weight", "1.5", "--opening-seconds", "10",
             "--lr", "1e-4"]


def run_one(shards: Path, config: str, steps: int, warmup: int, rows: int) -> dict:
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from distill_step_bench import _load

    from blackout_env.train.distill.run import Learner, parse_args

    amp, comp, tf32, batch = config.split(":")
    argv = P11_FLAGS + ["--amp", amp, "--batch-size", batch] + (["--compile"] if comp == "1" else []) + \
        (["--tf32"] if tf32 == "1" else [])
    args = parse_args(argv)
    learner = Learner(args)
    dataset = _load(shards, rows)
    rng = np.random.default_rng(0)
    batch = int(batch)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        rows_ = dataset.sample(batch, rng, labelled_only=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        learner.step(rows_)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.median(times))
    return {"config": config, "ms_median": ms, "ms_mean": 1000 * float(np.mean(times)), "rows_per_s": batch * 1000 / ms,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shards", type=Path)
    p.add_argument("--configs", nargs="+", default=["none:0:0:256", "none:0:1:256", "fp16:1:0:256", "bf16:1:0:256"])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rows", type=int, default=60_000)
    p.add_argument("--one", default="", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.one:
        print(json.dumps(run_one(args.shards, args.one, args.steps, args.warmup, args.rows)), flush=True)
        return
    results = []
    for c in args.configs:
        r = subprocess.run([sys.executable, __file__, str(args.shards), "--one", c, "--steps", str(args.steps),
                            "--warmup", str(args.warmup), "--rows", str(args.rows)], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if not line:
            print(c, "FAILED", r.stderr[-1500:], flush=True)
            continue
        d = json.loads(line[-1])
        results.append(d)
        print(f"{c:16s} {d['ms_median']:6.1f} ms/step (mean {d['ms_mean']:.1f})  {d['rows_per_s']:7.0f} rows/s  "
              f"peak {d['peak_vram_gb']:.2f} GB", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
