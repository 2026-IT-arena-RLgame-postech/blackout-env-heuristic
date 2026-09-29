"""Held-out evaluation of distilled students: fresh seeds, both sides, several opponents.

The in-training eval replays the same 3 seeds every version, and its states are trained on, so it
tracks progress but cannot judge the result.  This plays seeds no run has used (``--seed-base``)
from both sides against every ``--opponents`` entry, for each checkpoint x ``--inference`` mode,
and the teacher itself on the same seeds as the baseline.  Every match is labelled by the teacher,
so agreement, composition, economy and movement counts come out as in training (metrics.py).

  python examples/distill_heldout_eval.py checkpoints/distill/d1/v_0030.pt checkpoints/distill/d1/latest.pt \\
      --inference iqn fixed_tau tie_mean --out reports/distill_heldout_d1

Results append to ``<out>/matches.jsonl`` (a rerun skips matches already there) and the table is
printed and written to ``<out>/summary.txt``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from blackout_env.train.distill.collect import TEACHERS, CollectorPool, MatchTask
from blackout_env.train.distill.inference import INFERENCE
from blackout_env.train.distill.metrics import summarize_matches

TEACHER = "teacher"  # rows are labelled f"teacher:{--teacher}"; bare "teacher" rows predate that (v17-teach)


def _key(candidate: str, inference: str, opponent: str, seed: int, swap: bool) -> tuple:
    return candidate, inference, opponent, seed, swap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoints", nargs="*", type=Path, help="none = the teacher baseline alone")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--opponents", nargs="+", default=["strategic_v4", "strategic_v17", "strategic_v19"])
    p.add_argument("--inference", nargs="+", choices=INFERENCE, default=["iqn"])
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=900_000, help="far from every training/eval seed (2026xxxx)")
    p.add_argument("--teacher", choices=sorted(TEACHERS), default="v17-teach")
    p.add_argument("--no-teacher-baseline", action="store_true")
    p.add_argument("--build", default="build/mac/BlackOut.app")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--time-scale", type=float, default=20.0)
    args = p.parse_args()

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
                        shard=None, tag="heldout", inference="iqn" if inference == "-" else inference))
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
    lines = [f"{'candidate':42s} {'inference':9s} {'opponent':14s} {'n':>3s} {'margin':>7s} {'±se':>5s} {'win':>5s} "
             f"{'H<30s':>5s} {'H<60s':>5s} {'H<120':>5s} {'Ca<60':>5s} {'end_s':>5s} {'agree':>6s} {'deliv/1k':>8s} {'rev/1k':>7s} "
             f"{'die/1k':>6s} {'@base':>5s} {'camped':>6s}"]
    for (candidate, inference, opponent), group in sorted(groups.items()):
        s = summarize_matches(group)
        rev = s.get("reversals_per_1000/student", s.get("reversals_per_1000/teacher", float("nan")))
        name = candidate if candidate.startswith(TEACHER) else "/".join(Path(candidate).parts[-2:])
        lines.append(f"{name:42s} {inference:9s} {opponent:14s} {s['matches']:3d} {s['mean_margin']:+7.1f} {s['margin_se']:5.1f} "
                     f"{s['win_rate']:5.2f} {s.get('units_hunter_0_30s', float('nan')):5.1f} {s.get('units_hunter_30_60s', float('nan')):5.1f} "
                     f"{s.get('units_hunter_60_120s', float('nan')):5.1f} {s.get('units_carrier_30_60s', float('nan')):5.1f} {s['match_seconds']:5.0f} "
                     f"{s.get('agree/all', float('nan')):6.2f} {s['learner/deliveries_per_1000']:8.1f} {rev:7.0f} "
                     f"{s.get('deaths_per_1000', float('nan')):6.2f} {s.get('deaths_near_base_frac', float('nan')):5.2f} "
                     f"{s.get('camped_frac', float('nan')):6.2f}")
    table = "\n".join(lines)
    print(table)
    (args.out / "summary.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
