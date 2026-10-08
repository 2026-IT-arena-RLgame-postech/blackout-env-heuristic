"""Where a collection worker's tick goes: teacher labels, student, opponent, Unity, recording.

Plays ``--matches`` labelled matches in one headless Unity (as a CollectorPool worker would)
with every part of ``play_labelled_match`` timed, then prints ms per decision tick for each part
and, with ``--cprofile``, the hottest functions under the teacher and the opponent.

  python examples/distill_tick_profile.py --student checkpoints/distill/d4/latest.pt \\
      --teacher v21-teach --opponent strategic_v17 --matches 2 --cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict

import numpy as np

import blackout_env.train.distill.collect as collect
from blackout_env.train.distill.collect import MatchTask, play_labelled_match
from blackout_env.train.distill.metrics import MatchRecorder

TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)


def _timed(name: str, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMES[name] += time.perf_counter() - start
            CALLS[name] += 1
    return wrapper


class _TimedPolicy:
    """Times ``act`` / ``teach`` of the wrapped policy under ``name``."""

    def __init__(self, policy, name: str):
        self._policy, self._name = policy, name

    def __getattr__(self, attr):
        value = getattr(self._policy, attr)
        if attr in ("act", "teach"):
            return _timed(f"{self._name}.{attr}", value)
        return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--student", default=None, help="checkpoint; none = the teacher drives (beta 1)")
    p.add_argument("--inference", default="iqn")
    p.add_argument("--teacher", default="v21-teach")
    p.add_argument("--opponent", default="strategic_v17")
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--seed", type=int, default=990_000)
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--cprofile", action="store_true")
    args = p.parse_args()

    import torch

    torch.set_num_threads(1)  # as in a pool worker
    from blackout_env import BlackOutEnv

    env = BlackOutEnv(args.build, time_scale=args.time_scale, no_graphics=True,
                      additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    env.step = _timed("unity.step", env.step)
    env.reset = _timed("unity.reset", env.reset)
    teacher_cls = collect.TEACHERS[args.teacher]
    collect.TEACHERS[args.teacher] = lambda: _TimedPolicy(teacher_cls(), "teacher")
    make_opponent = collect.make_opponent
    collect.make_opponent = lambda d, s: _TimedPolicy(make_opponent(d, s), "opponent")
    collect._labelled_row = _timed("record.row", collect._labelled_row)
    MatchRecorder.before_step = _timed("record.before_step", MatchRecorder.before_step)
    MatchRecorder.after_step = _timed("record.after_step", MatchRecorder.after_step)
    student, _ = collect._load_student(args.student, args.inference)
    if student is not None:
        student = _TimedPolicy(student, "student")

    profiler = cProfile.Profile() if args.cprofile else None
    game_s, wall = 0.0, 0.0
    try:
        for i in range(args.matches):
            task = MatchTask(seed=args.seed + i, swap=bool(i % 2), opponent=args.opponent,
                             beta=1.0 if student is None else 0.0, teacher=args.teacher,
                             student=args.student, shard=None, tag="profile", inference=args.inference)
            start = time.perf_counter()
            if profiler:
                profiler.enable()
            rows, summary = play_labelled_match(env, task, student)
            if profiler:
                profiler.disable()
            wall += time.perf_counter() - start
            game_s += summary["elapsed_s"]
            print(f"[profile] match {i}: {summary['elapsed_s']:.0f} game s in {time.perf_counter() - start:.1f} s wall", flush=True)
    finally:
        env.close()

    ticks = max(1, CALLS["teacher.teach"])
    accounted = sum(t for name, t in TIMES.items() if name != "unity.reset")
    print(f"\n{ticks} decision ticks, {game_s:.0f} game s in {wall:.1f} s wall = {game_s / wall:.1f}x real time "
          f"(time_scale {args.time_scale:g})")
    print(f"{'part':22s} {'ms/tick':>8s} {'share':>6s} {'calls':>6s}")
    for name, t in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        print(f"{name:22s} {1000 * t / ticks:8.2f} {t / wall:6.1%} {CALLS[name]:6d}")
    other = wall - accounted - TIMES.get("unity.reset", 0.0)
    print(f"{'(loop, other)':22s} {1000 * other / ticks:8.2f} {other / wall:6.1%}")

    if profiler:
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out)
        stats.sort_stats("tottime").print_stats(30)
        print(out.getvalue())


if __name__ == "__main__":
    main()
