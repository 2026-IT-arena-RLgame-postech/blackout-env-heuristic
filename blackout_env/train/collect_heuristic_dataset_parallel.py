"""
Standalone entry point: run N collect_heuristic_dataset.py workers in parallel (each its own
Unity process -- BlackOutEnv's base_port=None default has every worker call find_free_port()
independently, so no manual port coordination is needed, see blackout_env.py), then merge their
shards into one combined dataset.

Collection is embarrassingly parallel (each worker's episodes are independent heuristic-vs-
heuristic rollouts), so wall-clock scales down close to linearly with --workers on a machine
with enough cores/memory for that many concurrent Unity instances. Workers get distinct
heuristic seeds (2*i, 2*i+1) so they don't all sample the identical HeuristicPolicyMixture
sequence -- see collect_heuristic_dataset.py's own --heuristic-seed-a/-b.

Usage:
    python -m blackout_env.train.collect_heuristic_dataset_parallel \\
        --build build/mac/BlackOut.app --steps 200000 --workers 8 --noise-frac 0.1 --out datasets/run1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from blackout_env.train.offline_dataset import merge_shards, write_collection_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="Path to the Unity build executable")
    parser.add_argument("--steps", type=int, required=True, help="Total env steps across all workers combined")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel Unity instances")
    parser.add_argument("--noise-frac", type=float, default=0.1, help="See collect_heuristic_dataset.py --noise-frac")
    parser.add_argument("--out", required=True, help="Output directory for the merged buffer_a.npz/buffer_b.npz")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale")
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep each worker's shard dir (<out>/_shards/worker_i) after merging instead of "
        "deleting it -- shards are plain float32 .npz dumps, easily hundreds of MB each times "
        "--workers, so the default is to reclaim that disk once the merge succeeds.",
    )
    parser.add_argument("--no-unity-shaping", action="store_true",
                        help="See collect_heuristic_dataset.py --no-unity-shaping")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    out_dir = Path(args.out)
    shard_root = out_dir / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    base_steps, remainder = divmod(args.steps, args.workers)
    shard_dirs = []
    procs: list[subprocess.Popen] = []
    log_files = []
    t0 = time.time()
    for i in range(args.workers):
        worker_steps = base_steps + (1 if i < remainder else 0)  # spread the remainder over the first few workers
        if worker_steps <= 0:
            continue
        shard_dir = shard_root / f"worker_{i}"
        shard_dirs.append(shard_dir)
        log_path = shard_root / f"worker_{i}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)
        cmd = [
            sys.executable, "-m", "blackout_env.train.collect_heuristic_dataset",
            "--build", args.build,
            "--steps", str(worker_steps),
            "--noise-frac", str(args.noise_frac),
            "--time-scale", str(args.time_scale),
            "--out", str(shard_dir),
            "--heuristic-seed-a", str(2 * i),
            "--heuristic-seed-b", str(2 * i + 1),
        ] + (["--no-unity-shaping"] if args.no_unity_shaping else [])
        print(f"[parallel] launching worker {i}: {worker_steps} steps, seeds ({2*i},{2*i+1}), log -> {log_path}")
        procs.append(subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT))

    failures = []
    for i, p in enumerate(procs):
        code = p.wait()
        log_files[i].close()
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"[parallel] worker {i} {status}")
        if code != 0:
            failures.append(i)

    if failures:
        raise SystemExit(
            f"[parallel] {len(failures)}/{len(procs)} worker(s) failed: {failures} -- see "
            f"{shard_root}/worker_<i>.log for each one's output. Not merging a partial dataset."
        )

    elapsed = time.time() - t0
    print(f"[parallel] all {len(procs)} workers finished in {elapsed:.0f}s, merging shards ...")

    n_a = merge_shards([d / "buffer_a.npz" for d in shard_dirs], out_dir / "buffer_a.npz")
    n_b = merge_shards([d / "buffer_b.npz" for d in shard_dirs], out_dir / "buffer_b.npz")
    write_collection_info(out_dir, unity_shaping=not args.no_unity_shaping)
    print(f"[parallel] merged {n_a} (stream a) / {n_b} (stream b) transitions -> {out_dir}")

    if not args.keep_shards:
        shutil.rmtree(shard_root)
        print(f"[parallel] removed shard dir {shard_root}")
    else:
        print(f"[parallel] kept shard dir {shard_root} (--keep-shards)")

    print(
        f"[parallel] next: python -m blackout_env.train.offline_pretrain "
        f"--dataset-dir {out_dir} --steps <gradient steps> --device mps"
    )


if __name__ == "__main__":
    main()
