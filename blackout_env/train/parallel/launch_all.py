"""
Runs 2 independent launch_group.py groups side by side, across all 4 GPUs -- group 0 on GPUs
0-1, group 1 on GPUs 2-3 by default -- as 2 independent experiments (different heuristic seeds by
default; pass distinct extra args per group if you also want different hyperparameters). Same
"spawn one subprocess per independent worker, wait, report" convention as
collect_heuristic_dataset_parallel.py elsewhere in this codebase, just one process per GROUP
(itself a whole actor+inference+learner process tree) instead of one process per Unity worker.

Splits --num-actors and CPU budget evenly between the 2 groups automatically unless overridden.

Usage:
    python -m blackout_env.train.parallel.launch_all \\
        --build build/linux/BlackOut.x86_64 --steps 2000000

Smoke test (no Unity build, no GPUs required):
    python -m blackout_env.train.parallel.launch_all --smoke-test --steps 2000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default=None)
    parser.add_argument("--steps", type=int, default=1_000_000, help="Per-group env step budget (each group trains independently)")
    parser.add_argument("--num-actors", type=int, default=None, help="Per-group actor count; default splits (cpu_count - 8) evenly across the 2 groups")
    parser.add_argument("--devices", default="0,1,2,3", help="4 comma-separated CUDA device indices: group0=(devices[0] infer, devices[1] learn), group1=(devices[2] infer, devices[3] learn)")
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--smoke-test", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    if not args.smoke_test and not args.build:
        raise SystemExit("--build is required unless --smoke-test")

    if args.smoke_test:
        devices = ["cpu", "cpu", "cpu", "cpu"]
    else:
        devices = [f"cuda:{d.strip()}" for d in args.devices.split(",")]
        if len(devices) != 4:
            raise SystemExit(f"--devices must list exactly 4 indices, got {args.devices!r}")

    num_actors = args.num_actors or max(1, (mp.cpu_count() - 8) // 2)

    procs = []
    for group_id, (infer_dev, learn_dev, seed_offset) in enumerate([(devices[0], devices[1], 0), (devices[2], devices[3], 1000)]):
        cmd = [
            sys.executable, "-m", "blackout_env.train.parallel.launch_group",
            "--steps", str(args.steps),
            "--num-actors", str(num_actors),
            "--infer-device", infer_dev,
            "--learn-device", learn_dev,
            "--time-scale", str(args.time_scale),
            "--group-id", str(group_id),
            "--seed-offset", str(seed_offset),
        ]
        if args.build:
            cmd += ["--build", args.build]
        if args.smoke_test:
            cmd += ["--smoke-test"]
        cmd += passthrough
        print(f"[launch_all] group {group_id}: infer={infer_dev} learn={learn_dev} actors={num_actors}")
        procs.append(subprocess.Popen(cmd))

    try:
        exit_codes = [p.wait() for p in procs]
    except KeyboardInterrupt:
        print("\n[launch_all] KeyboardInterrupt -- terminating both groups")
        for p in procs:
            p.terminate()
        deadline = time.time() + 30
        for p in procs:
            try:
                p.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        return

    failures = [i for i, code in enumerate(exit_codes) if code != 0]
    if failures:
        raise SystemExit(f"[launch_all] group(s) {failures} exited non-zero: {exit_codes}")
    print("[launch_all] both groups finished")


if __name__ == "__main__":
    main()
