"""Tables for the decisions in docs/distill_plan_ko.md, from a distillation run's metrics.jsonl.

  python examples/distill_report.py checkpoints/distill/d1 [--every 5000]

1. evaluation per student version, with the teacher's baseline on the same seeds first;
2. learning curve: train vs fresh (never-trained-on) vs probe accuracy, by role, logit health,
   pipeline health;
3. the latest fresh accuracy by label kind (what the student still gets wrong);
4. direction reversals in the eval matches, split by whether the teacher had a label (from the
   shards, so it also covers runs logged before the split existed).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.model.direction_bins import bits_bins, unpack_bits
from blackout_env.train.distill.dataset import shard_meta

ROLES = ("collector", "hunter", "carrier")
TRAP_TICKS = 25


def _records(out_dir: Path) -> dict[str, list[dict]]:
    by_kind: dict[str, list[dict]] = {}
    for line in (out_dir / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_kind.setdefault(record["kind"], []).append(record)
    return by_kind


def _fmt(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def _table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def evaluation(records: dict[str, list[dict]]) -> None:
    print("\n== eval (student alone vs the eval opponent; teacher = baseline on the same seeds) ==")
    header = ["who", "step", "margin", "±se", "win", "H", "Ca", "C", "H<30s", "H<60s", "end_s", "t(H3)", "agree", "agreeH", "agreeC",
              "deliv/1k", "lost/1k", "kill/1k", "blocked/1k", "rev/1k"]
    rows = []
    for r in records.get("eval_teacher", []) + records.get("eval", []):
        who = "teacher" if r["kind"] == "eval_teacher" else f"v{r['version']}"
        rows.append([who, str(r["step"]), _fmt(r["mean_margin"], "+.1f"), _fmt(r["margin_se"], ".1f"), _fmt(r["win_rate"]),
                     _fmt(r["units_hunter"], ".1f"), _fmt(r["units_carrier"], ".1f"), _fmt(r["units_collector"], ".1f"),
                     _fmt(r.get("units_hunter_0_30s"), ".1f"), _fmt(r.get("units_hunter_30_60s"), ".1f"), _fmt(r.get("match_seconds"), ".0f"),
                     _fmt(r.get("first_hunter3_s"), ".0f"), _fmt(r.get("agree/all")), _fmt(r.get("agree/role_hunter")),
                     _fmt(r.get("agree/role_collector")), _fmt(r["learner/deliveries_per_1000"], ".1f"),
                     _fmt(r["learner/cargo_lost_per_1000"], ".1f"), _fmt(r["opponent/cargo_lost_per_1000"], ".1f"),
                     _fmt(r["movement/blocked_per_1000"], ".1f"),
                     _fmt(r.get("reversals_per_1000/student", r.get("reversals_per_1000/teacher")), ".0f")])
    _table(header, rows)


def learning(records: dict[str, list[dict]], every: int) -> None:
    print("\n== learning (accuracy = argmax inside the teacher's set) ==")
    probes = records.get("probe", [])
    header = ["step", "v", "beta", "ce", "train", "fresh", "freshEval", "probe",
              *[f"fresh_{r[:4]}" for r in ROLES], "z", "entropy", "|logit|", "step/s", "busy", "throttled", "ringAge"]
    rows, last = [], None
    for r in records.get("train", []):
        if last is not None and r["step"] - last < every:
            continue
        last = r["step"]
        probe = next((p for p in reversed(probes) if p["step"] <= r["step"]), {})
        rows.append([str(r["step"]), str(r["version"]), _fmt(r["beta"]), _fmt(r["train/ce"], ".3f"),
                     _fmt(r.get("train/accuracy")), _fmt(r.get("fresh_collect/accuracy")), _fmt(r.get("fresh_eval/accuracy")),
                     _fmt(probe.get("probe/accuracy")),
                     *[_fmt(r.get(f"fresh_collect/accuracy_role_{role}")) for role in ROLES],
                     _fmt(r["train/z"], ".2f"), _fmt(r["train/entropy"]), _fmt(r["train/logit_abs_max"], ".1f"),
                     _fmt(r["pipeline/steps_per_s"], ".1f"), _fmt(r.get("pipeline/worker_utilization")),
                     _fmt(r["pipeline/throttled_sync_frac"]), _fmt(r.get("ring/mean_version_age"), ".1f")])
    _table(header, rows)


def by_kind(records: dict[str, list[dict]]) -> None:
    latest = next((r for r in reversed(records.get("train", [])) if "fresh_collect/accuracy" in r), None)
    if latest is None:
        return
    print(f"\n== fresh accuracy by label kind (step {latest['step']}) ==")
    kinds = sorted({k.split("accuracy_kind_")[1] for k in latest if "/accuracy_kind_" in k})
    rows = [[kind, _fmt(latest.get(f"fresh_collect/accuracy_kind_{kind}")), _fmt(latest.get(f"fresh_collect/share_kind_{kind}")),
             _fmt(latest.get(f"train/accuracy_kind_{kind}"))] for kind in kinds]
    _table(["kind", "fresh", "share", "train"], rows)


def reversals(out_dir: Path) -> None:
    """A reversal is a move turning more than 135 degrees from the previous tick's move (both
    moving; MatchRecorder's definition).  Where the teacher has no label (interacting, holding a
    camp slot) it stands still; the student has no "stay" action and dithers, so those reversals
    are its way of standing.  Where it has a label, a reversal away from the label set is a real
    error; one into the set is a correction."""
    print("\n== eval reversals per 1000 unit-ticks, by whether the teacher had a label ==")
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # eval shards live in eval_shards/ (runs before 2026-09-29 kept them in shards/)
    for path in sorted([*out_dir.glob("eval_shards/*_eval*.npz"), *out_dir.glob("shards/*_eval*.npz")]):
        tag, version = re.search(r"_(eval(?:_teacher)?)_v(\d+)", path.name).groups()
        who = "teacher" if tag == "eval_teacher" else f"v{int(version)}"
        with np.load(path) as shard:
            moves, labels, states = shard["actions"].astype(np.int16), shard["labels"], shard["agent_states"]
            time_left = shard["team_state"][:, 2]
            bins = bits_bins(labels, shard_meta(shard).get("action_bins"))  # 8, or K for a --action-bins run
        if len(moves) < 2:
            continue
        step = -np.diff(time_left)
        consecutive = (step > 0) & (step < 1.5 * np.median(step))  # dropped (unlabelled) ticks break a pair
        prev, cur, label = moves[:-1], moves[1:], labels[1:]
        moving = (prev < bins) & (cur < bins) & consecutive[:, None]
        turn = (cur - prev) % bins  # in bins: a turn of 135 degrees or more = 3K/8 .. 5K/8
        reversal = moving & (turn >= 3 * bins // 8) & (turn <= 5 * bins // 8)
        labelled = label > 0
        in_set = np.take_along_axis(unpack_bits(label, bins), np.minimum(cur, bins - 1)[..., None], axis=-1)[..., 0]
        hunter = states[1:, :5, 10] > 0.5
        # A trap is an unbroken run of reversals: the unit steps back and forth in place.
        trapped = np.zeros_like(reversal)
        for u in range(reversal.shape[1]):
            run_start = None
            for i, flag in enumerate(np.r_[reversal[:, u], False]):
                if flag and run_start is None:
                    run_start = i
                elif not flag and run_start is not None:
                    if i - run_start >= TRAP_TICKS:
                        trapped[run_start:i, u] = True
                        totals[who]["traps"] += 1
                        totals[who]["trap_longest"] = max(totals[who]["trap_longest"], i - run_start)
                    run_start = None
        t = totals[who]
        t["matches"] += 1
        t["trapped_ticks"] += (trapped & labelled).sum()
        t["rev_labelled_out_free"] += (reversal & labelled & ~in_set & ~trapped).sum()
        t["labelled_ticks"] += (labelled & consecutive[:, None]).sum()
        t["unlabelled_ticks"] += (~labelled & consecutive[:, None]).sum()
        t["rev_labelled_out"] += (reversal & labelled & ~in_set).sum()
        t["rev_labelled_in"] += (reversal & labelled & in_set).sum()
        t["rev_unlabelled"] += (reversal & ~labelled).sum()
        t["rev_unlabelled_hunter"] += (reversal & ~labelled & hunter).sum()
    order = sorted(totals, key=lambda w: (w != "teacher", int(w[1:]) if w != "teacher" else 0))
    rows = []
    for who in order:
        t = totals[who]
        per = lambda n, d: _fmt(1000 * t[n] / max(1, t[d]), ".0f")
        rows.append([who, _fmt(t["labelled_ticks"] / max(1, t["labelled_ticks"] + t["unlabelled_ticks"])),
                     per("rev_labelled_out", "labelled_ticks"), per("rev_labelled_out_free", "labelled_ticks"),
                     per("rev_labelled_in", "labelled_ticks"), per("rev_unlabelled", "unlabelled_ticks"),
                     _fmt(t["rev_unlabelled_hunter"] / max(1, t["rev_unlabelled"])),
                     _fmt(t["traps"] / max(1, t["matches"]), ".1f"), str(int(t["trap_longest"])),
                     _fmt(t["trapped_ticks"] / max(1, t["labelled_ticks"]), ".1%")])
    _table(["who", "labelled", "away", "away:outside-traps", "into", "no-label", "no-label:hunter",
            "traps/match", "longest", "labelled-trapped"], rows)
    print(f"  away/into = reversal leaving/entering the label set; a trap = >= {TRAP_TICKS} reversals in a row "
          f"(~1 s back and forth); rates per 1000 ticks of that labelled/unlabelled kind")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--every", type=int, default=5000, help="steps between learning-curve rows")
    args = parser.parse_args()
    info = json.loads((args.out_dir / "run_info.json").read_text())
    for start in info:
        print(f"start {start['started']}  commit {start['git_commit'][:8]}  dirty {start['git_dirty'] or 'no'}")
    records = _records(args.out_dir)
    evaluation(records)
    learning(records, args.every)
    by_kind(records)
    reversals(args.out_dir)


if __name__ == "__main__":
    main()
