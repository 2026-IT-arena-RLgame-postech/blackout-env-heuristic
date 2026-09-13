"""
Launches ONE group: 1 inference-server process + 1 learner process + N actor processes (see
package docstring). A "group" uses 2 GPUs (one for inference, one for learning) and however many
CPU cores/Unity instances you give it via --num-actors.

For the target 4x RTX2080Ti / 28-core box, run two of these side by side (see launch_all.py) --
e.g. group 0 on GPUs 0-1, group 1 on GPUs 2-3, each with --num-actors ~10-12 -- to use all 4 GPUs
as two independent experiments (different seeds/hyperparameters) while still getting each
individual run's actor-learner throughput win.

Usage:
    python -m blackout_env.train.parallel.launch_group \\
        --build build/linux/BlackOut.x86_64 --steps 2000000 --num-actors 10 \\
        --infer-device cuda:0 --learn-device cuda:1

Smoke test (no Unity build, no GPU required -- validates the multiprocessing wiring only):
    python -m blackout_env.train.parallel.launch_group \\
        --smoke-test --steps 2000 --num-actors 2 --infer-device cpu --learn-device cpu
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from blackout_env.train.qmix_trainer import QMIXConfig
from blackout_env.train.tb_logger import default_run_dir

from .actor import run_actor
from .inference_server import run_inference_server
from .learner import run_learner
from .shared import SharedState


def _default_num_actors() -> int:
    """Leaves headroom for the learner + inference-server processes (each mostly GPU-bound but
    still needs a real core for sampling/data-loading/Python overhead) plus OS/other-group usage
    when running two groups side by side -- see launch_all.py, which divides this by 2."""
    cores = mp.cpu_count()
    return max(1, cores - 4)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default=None, help="Path to the Unity build executable (required unless --smoke-test)")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total env steps (summed across all actors) to train for")
    parser.add_argument("--num-actors", type=int, default=None, help="Parallel Unity instances; default leaves headroom on the box (cpu_count - 4)")
    parser.add_argument("--infer-device", default="cuda:0", help="torch device for the inference server (e.g. cuda:0)")
    parser.add_argument("--learn-device", default="cuda:1", help="torch device for the learner (e.g. cuda:1)")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity Time.timeScale per actor")
    parser.add_argument("--max-infer-batch", type=int, default=64, help="Max requests the inference server batches into one forward pass")
    parser.add_argument("--weight-sync-interval", type=int, default=50, help="Learner gradient steps between weight pushes to the inference server")
    parser.add_argument("--group-id", default="0", help="Used only to namespace default checkpoint/log/weights paths when running multiple groups")
    parser.add_argument("--checkpoint-dir", default=None, help="Default: fresh timestamped dir under checkpoints/ (see default_run_dir)")
    parser.add_argument("--tb-log-dir", default=None, help="Default: fresh timestamped dir under runs/; pass '' to disable. Each actor logs to <this>/actor_<id>")
    parser.add_argument("--weights-path", default=None, help="Where the learner writes net/ema_net for the inference server; default lives under --checkpoint-dir")
    parser.add_argument("--seed-offset", type=int, default=0, help="Added to every actor's heuristic seeds -- vary this across groups/experiments for diversity")
    parser.add_argument("--resume", default=None, help="Checkpoint path for the learner to resume from")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Force heuristic_fill_frac=0 -- see qmix_trainer.py's --skip-bootstrap")
    parser.add_argument("--heuristic-opponent-frac", type=float, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the learner's net/target_net/ema_net")
    parser.add_argument("--reset-interval", type=int, default=None)
    parser.add_argument("--unity-log-dir", default=None, help="Directory for each actor's Unity player log; default under unity_logs/")
    parser.add_argument("--smoke-test", action="store_true", help="Use an in-process fake env instead of Unity -- validates the pipeline wiring only, does not train a real policy")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.smoke_test and not args.build:
        raise SystemExit("--build is required unless --smoke-test is passed")

    num_actors = args.num_actors or _default_num_actors()
    checkpoint_dir = args.checkpoint_dir or default_run_dir(base=f"checkpoints/group_{args.group_id}")
    tb_log_dir = args.tb_log_dir if args.tb_log_dir is not None else default_run_dir(base=f"runs/group_{args.group_id}")
    weights_path = args.weights_path or str(Path(checkpoint_dir) / "_infer_weights.pt")
    unity_log_base = args.unity_log_dir or default_run_dir(base=f"unity_logs/group_{args.group_id}")
    if not args.smoke_test:
        # Unity's -logFile does not create missing parent directories itself (see
        # qmix_trainer.py's main(), which has the same requirement).
        Path(unity_log_base).mkdir(parents=True, exist_ok=True)

    config_kwargs = dict(
        device=args.learn_device,
        compile=args.compile,
        checkpoint_dir=checkpoint_dir,
        tb_log_dir=(tb_log_dir or None),
        heuristic_seed_a=args.seed_offset,
        heuristic_seed_b=args.seed_offset + 1,
    )
    if args.heuristic_opponent_frac is not None:
        config_kwargs["heuristic_opponent_frac"] = args.heuristic_opponent_frac
    if args.reset_interval is not None:
        config_kwargs["reset_interval"] = args.reset_interval
    if args.skip_bootstrap:
        config_kwargs["heuristic_fill_frac"] = 0.0
    cfg = QMIXConfig(**config_kwargs)

    if args.resume:
        # Loaded by the learner itself right after construction -- but run_learner doesn't take
        # --resume today (QMIXTrainer.load needs a live trainer instance to call it on). Fail
        # fast with a clear message rather than silently ignoring --resume, until/unless this is
        # worth threading through run_learner.
        raise SystemExit(
            "--resume is not yet wired into run_learner (see launch_group.py) -- for now, resume "
            "with the single-process qmix_trainer.py, or extend run_learner to call trainer.load()."
        )

    print(
        f"[launch_group {args.group_id}] {num_actors} actors | infer={args.infer_device} learn={args.learn_device} | "
        f"checkpoint_dir={checkpoint_dir} tb_log_dir={tb_log_dir or '(disabled)'}"
    )

    ctx = mp.get_context("spawn")
    shared = SharedState(ctx)
    # Bounded (not just relying on put_until_stop's timeout retries) so a stalled learner or
    # inference server makes actors visibly back-pressure/block instead of actors' memory usage
    # growing without limit -- see shared.py's put_until_stop for why put() needing a timeout at
    # all is unavoidable regardless of maxsize.
    request_queue = ctx.Queue(maxsize=max(4, num_actors * 4))
    transition_queue = ctx.Queue(maxsize=20_000)
    response_queues = {i: ctx.Queue(maxsize=4) for i in range(num_actors)}

    procs: list[mp.process.BaseProcess] = []

    learner_proc = ctx.Process(
        target=run_learner,
        args=(cfg, transition_queue, shared, args.steps, weights_path, args.weight_sync_interval),
        name="learner",
    )
    procs.append(learner_proc)

    inference_proc = ctx.Process(
        target=run_inference_server,
        args=(cfg, args.infer_device, request_queue, response_queues, shared.stop, weights_path, args.max_infer_batch),
        name="inference_server",
    )
    procs.append(inference_proc)

    actor_procs = []
    for i in range(num_actors):
        unity_log_file = None if args.smoke_test else f"{unity_log_base}/actor_{i}.log"
        p = ctx.Process(
            target=run_actor,
            args=(
                i,
                args.build,
                cfg,
                shared,
                request_queue,
                response_queues[i],
                transition_queue,
                args.time_scale,
                unity_log_file,
                tb_log_dir or None,
                args.smoke_test,
            ),
            name=f"actor_{i}",
        )
        actor_procs.append(p)
    procs.extend(actor_procs)

    for p in procs:
        p.start()

    try:
        learner_proc.join()
    except KeyboardInterrupt:
        print(f"\n[launch_group {args.group_id}] KeyboardInterrupt -- shutting down")
    finally:
        shared.stop.set()
        deadline = time.time() + 30
        for p in procs:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
        for p in procs:
            if p.is_alive():
                print(f"[launch_group {args.group_id}] force-terminating {p.name} (didn't exit within timeout)")
                p.terminate()
                p.join(timeout=5)

    print(f"[launch_group {args.group_id}] done")


if __name__ == "__main__":
    main()
