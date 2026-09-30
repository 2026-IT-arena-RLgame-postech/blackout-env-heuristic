"""Held-out evaluation of distilled students: fresh seeds, both sides, several opponents.

The in-training eval replays the same 3 seeds every version, and its states are trained on, so it
tracks progress but cannot judge the result.  This plays seeds no run has used (``--seed-base``)
from both sides against every ``--opponents`` entry, for each checkpoint x ``--inference`` mode,
and the teacher itself on the same seeds as the baseline.  Every match is labelled by the teacher,
so agreement, composition, economy and movement counts come out as in training (metrics.py).

  python examples/distill_heldout_eval.py checkpoints/distill/d1/v_0030.pt checkpoints/distill/d1/latest.pt \\
      --inference iqn fixed_tau tie_mean --out reports/distill_heldout_d1

``--opponents`` takes any opponents.py descriptor -- ``ckpt:<path>``, a heuristic with fixed
arguments (``strategic_v19:hunter_quota=4,hunter_mode=camp``), ``random``, ``idle``,
``teacher:<name>``, ``self`` (mirror match) -- and ``ood``, the out-of-distribution set of
docs/distill_plan_ko.md §6 (``OOD_OPPONENTS``):

  python examples/distill_heldout_eval.py checkpoints/distill/d4/latest.pt --teacher v21-teach \
      --opponents ood --save-shards --out reports/distill_ood_d4

Results append to ``<out>/matches.jsonl`` (a rerun skips matches already there).  The table --
plus each checkpoint's margin minus the teacher's on the same seed and side, the paired gap -- is
printed and written to ``<out>/summary.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.train.distill.collect import TEACHERS, CollectorPool, MatchTask
from blackout_env.train.distill.inference import INFERENCE
from blackout_env.train.distill.metrics import summarize_matches
from blackout_env.train.distill.opponents import SELF, TEACHER_PREFIX, check_opponent, opponent_name

TEACHER = "teacher"  # rows are labelled f"teacher:{--teacher}"; bare "teacher" rows predate that (v17-teach)
IN_DISTRIBUTION = ["strategic_v4", "strategic_v17", "strategic_v19"]
RUN16 = "checkpoints/offline/20260919-010456_42178"  # kept out of D4's opponent pool
OOD_OPPONENTS = [
    # learned play never met in training
    f"ckpt:{RUN16}/final.pt", f"ckpt:{RUN16}/step_100000.pt", f"ckpt:{RUN16}/step_50000.pt",
    # heuristics outside the mixture's default weights
    "strategic_v20", "strategic_v21",
    # outside the mixture's perturbation ranges (Hunters 2-4 mixed/hunt, steal 1.3-2.3, exposure 0.6-1.2)
    "strategic_v19:hunter_quota=4,hunter_mode=camp", "strategic_v17:steal_weight=3.0,exposure_cost=0.3",
    # nothing any heuristic plays like
    "random", "idle",
    # mirror matches: the learner's own states on both sides
    SELF, TEACHER_PREFIX + "{teacher}",
]


def _key(candidate: str, inference: str, opponent: str, seed: int, swap: bool) -> tuple:
    return candidate, inference, opponent, seed, swap


def _shard_path(args, candidate: str, inference: str, opponent: str, seed: int, swap: bool) -> str | None:
    if not args.save_shards:
        return None
    name = candidate.replace(":", "_") if candidate.startswith(TEACHER) else "_".join(Path(candidate).parts[-2:]).removesuffix(".pt")
    (args.out / "shards").mkdir(parents=True, exist_ok=True)
    return str(args.out / "shards" / f"{name}__{inference}__{opponent_name(opponent)}__{seed}_{'B' if swap else 'A'}.npz")


def _opponents(entries: list[str], teacher: str) -> list[str]:
    """``--opponents`` with ``ood`` expanded; every descriptor is built once here to fail early."""
    expanded = [o for entry in entries for o in (OOD_OPPONENTS if entry == "ood" else [entry])]
    opponents = list(dict.fromkeys(o.format(teacher=teacher) for o in expanded))
    for descriptor in opponents:
        check_opponent(descriptor)
    return opponents


def _paired_gaps(groups: dict[tuple, list[dict]], teacher_name: str) -> list[str]:
    """Each checkpoint's margin minus the teacher's on the same (seed, side), averaged per seed
    over its two sides; ± is the standard error over seeds.  ``self`` is left out: both sides of
    a mirror match are the same policy, so its margin is noise around zero (its agreement, deaths
    and traps are what it is for); ``teacher:<teacher>`` against the teacher baseline -- the
    teacher's own mirror -- makes that row the checkpoint's margin against its teacher."""
    margin = lambda d: d["learner_score"] - d["opponent_score"]
    lines = [f"{'paired gap (candidate - teacher)':42s} {'inference':9s} {'opponent':44s} {'seeds':>5s} {'gap':>7s} {'±se':>5s}"]
    for (candidate, inference, opponent), group in sorted(groups.items()):
        base = groups.get((teacher_name, "-", opponent))
        if candidate == teacher_name or opponent == SELF or not base:
            continue
        teacher_margin = {(d["seed"], d["swap"]): margin(d) for d in base}
        per_seed = defaultdict(list)
        for d in group:
            if (d["seed"], d["swap"]) in teacher_margin:
                per_seed[d["seed"]].append(margin(d) - teacher_margin[(d["seed"], d["swap"])])
        gaps = np.array([np.mean(v) for v in per_seed.values()])
        if not len(gaps):
            continue
        se = float(gaps.std(ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
        name = "/".join(Path(candidate).parts[-2:])
        lines.append(f"{name:42s} {inference:9s} {opponent_name(opponent):44s} {len(gaps):5d} {gaps.mean():+7.1f} {se:5.1f}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoints", nargs="*", type=Path, help="none = the teacher baseline alone")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-shards", action="store_true",
                   help="keep every match's per-tick rows (states, labels, moves) under <out>/shards for analysis")
    p.add_argument("--opponents", nargs="+", default=IN_DISTRIBUTION,
                   help="opponents.py descriptors; 'ood' = OOD_OPPONENTS")
    p.add_argument("--inference", nargs="+", choices=INFERENCE, default=["iqn"])
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=900_000, help="far from every training/eval seed (2026xxxx)")
    p.add_argument("--teacher", choices=sorted(TEACHERS), default="v17-teach")
    p.add_argument("--no-teacher-baseline", action="store_true")
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--time-scale", type=float, default=20.0)
    args = p.parse_args()
    args.opponents = _opponents(args.opponents, args.teacher)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "matches.jsonl"
    done = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    for d in done:
        if d["candidate"] == TEACHER:
            d["candidate"] = f"{TEACHER}:v17-teach"
    teacher_name = f"{TEACHER}:{args.teacher}"
    finished = {_key(d["candidate"], d["inference"], d["opponent"], d["seed"], d["swap"]) for d in done}

    candidates = [(str(path), inference) for path in args.checkpoints for inference in args.inference]
    if not args.no_teacher_baseline:
        candidates.insert(0, (teacher_name, "-"))
    tasks = []
    for candidate, inference in candidates:
        for opponent in args.opponents:
            for i in range(args.seeds):
                for swap in (False, True):
                    seed = args.seed_base + i
                    if _key(candidate, inference, opponent, seed, swap) in finished:
                        continue
                    is_teacher = candidate == teacher_name
                    tasks.append(MatchTask(
                        seed=seed, swap=swap, opponent=opponent, beta=1.0 if is_teacher else 0.0,
                        teacher=args.teacher, student=None if is_teacher else candidate,
                        shard=_shard_path(args, candidate, inference, opponent, seed, swap), tag="heldout",
                        inference="iqn" if inference == "-" else inference))
    print(f"[heldout] {len(tasks)} matches to play ({len(finished)} already done), {args.workers} workers", flush=True)

    with CollectorPool(args.build, args.workers, args.time_scale) as pool, open(log_path, "a") as log:
        for task in tasks:
            pool.submit(task)
        remaining = len(tasks)
        while remaining:
            for summary in pool.poll(timeout=5.0):
                candidate = f"{TEACHER}:{summary['teacher']}" if summary["student"] is None else summary["student"]
                inference = "-" if summary["student"] is None else summary["inference"]
                record = {"candidate": candidate, **{k: v for k, v in summary.items() if k != "inference"}, "inference": inference}
                log.write(json.dumps(record) + "\n")
                log.flush()
                done.append(record)
                remaining -= 1
                if remaining % 20 == 0:
                    print(f"[heldout] {remaining} left", flush=True)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in done:
        groups[(d["candidate"], d["inference"], d["opponent"])].append(d)
    lines = [f"{'candidate':42s} {'inference':9s} {'opponent':44s} {'n':>3s} {'margin':>7s} {'±se':>5s} {'win':>5s} "
             f"{'H<30s':>5s} {'H<60s':>5s} {'H<120':>5s} {'Ca<60':>5s} {'end_s':>5s} {'agree':>6s} {'deliv/1k':>8s} {'rev/1k':>7s} "
             f"{'die/1k':>6s} {'@base':>5s} {'camped':>6s}"]
    for (candidate, inference, opponent), group in sorted(groups.items()):
        s = summarize_matches(group)
        rev = s.get("reversals_per_1000/student", s.get("reversals_per_1000/teacher", float("nan")))
        name = candidate if candidate.startswith(TEACHER) else "/".join(Path(candidate).parts[-2:])
        lines.append(f"{name:42s} {inference:9s} {opponent_name(opponent):44s} {s['matches']:3d} {s['mean_margin']:+7.1f} {s['margin_se']:5.1f} "
                     f"{s['win_rate']:5.2f} {s.get('units_hunter_0_30s', float('nan')):5.1f} {s.get('units_hunter_30_60s', float('nan')):5.1f} "
                     f"{s.get('units_hunter_60_120s', float('nan')):5.1f} {s.get('units_carrier_30_60s', float('nan')):5.1f} {s['match_seconds']:5.0f} "
                     f"{s.get('agree/all', float('nan')):6.2f} {s['learner/deliveries_per_1000']:8.1f} {rev:7.0f} "
                     f"{s.get('deaths_per_1000', float('nan')):6.2f} {s.get('deaths_near_base_frac', float('nan')):5.2f} "
                     f"{s.get('camped_frac', float('nan')):6.2f}")
    table = "\n".join(lines + [""] + _paired_gaps(groups, teacher_name))
    print(table)
    (args.out / "summary.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
