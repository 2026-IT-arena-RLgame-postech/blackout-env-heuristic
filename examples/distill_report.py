"""Tables for the decisions in docs/distill_plan_ko.md, from a distillation run's metrics.jsonl.

  python examples/distill_report.py checkpoints/distill/d1 [--every 5000]

1. evaluation per student version, with the teacher's baseline on the same seeds first;
2. learning curve: train vs fresh (never-trained-on) vs probe accuracy, by role, logit health,
   pipeline health;
3. the latest fresh accuracy by label kind (what the student still gets wrong).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLES = ("collector", "hunter", "carrier")


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
    header = ["who", "step", "margin", "±se", "win", "H", "Ca", "C", "t(H3)", "agree", "agreeH", "agreeC",
              "deliv/1k", "lost/1k", "kill/1k", "blocked/1k", "rev/1k"]
    rows = []
    for r in records.get("eval_teacher", []) + records.get("eval", []):
        who = "teacher" if r["kind"] == "eval_teacher" else f"v{r['version']}"
        rows.append([who, str(r["step"]), _fmt(r["mean_margin"], "+.1f"), _fmt(r["margin_se"], ".1f"), _fmt(r["win_rate"]),
                     _fmt(r["units_hunter"], ".1f"), _fmt(r["units_carrier"], ".1f"), _fmt(r["units_collector"], ".1f"),
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
                     _fmt(r["pipeline/steps_per_s"], ".1f"), _fmt(r["pipeline/busy_worker_frac"]),
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


if __name__ == "__main__":
    main()
