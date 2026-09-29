"""What a distillation run records, so that its decisions never need a rerun for evidence.

Two layers:

* ``MatchRecorder`` (worker side) turns one labelled match into raw *counts* -- student/teacher
  agreement by role and label kind, direction reversals by controller, the scoring pipeline
  (train/objective_monitor.py) for both sides, movement failures and stall reasons for the
  learner, when the composition was reached, and the margin at fixed times.  Counts, not rates,
  so ``rates`` can pool any set of matches without averaging averages.  The shard keeps the
  per-tick actions next to the labels for anything not foreseen here.
* ``LabelAccuracy`` (learner side) accumulates label accuracy by role and kind on-device; the
  trainer uses it for its batches, for fresh shards before they enter the ring, and for a fixed
  probe set.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from blackout_env.env.constants import unit_index
from blackout_env.model.my_policy import direction_vector_to_idx
from blackout_env.train.distill.dataset import KINDS, N_TEAM
from blackout_env.train.movement_monitor import MovementMonitor
from blackout_env.train.objective_monitor import ObjectiveMonitor
from blackout_env.train.stall_monitor import StallCounts, StallMonitor

ROLES = ("collector", "hunter", "carrier")
CLASS_COLS = slice(9, 12)  # agent_states class one-hot, ROLES order
MATCH_SECONDS = 420.0
MARGIN_AT = (30, 60, 120)  # seconds of play at which the learner's margin is recorded
# Composition by time of play: matches end when the field is exhausted, anywhere from 20 s to
# 420 s, so a whole-match average mostly measures how long the match lasted (D1 v35 read 1.2
# Hunters against the teacher's 2.8 with the same 0-60 s build-up, in matches a fifth as long).
TIME_BUCKETS = ((0, 30), (30, 60), (60, 120), (120, 10_000))
REVERSAL_COS = math.cos(math.radians(135))  # consecutive moves turning more than this = reversal
STAND = 255  # action index for "no movement" (a teacher without a label stands still)


def _role(state: np.ndarray) -> str:
    return ROLES[int(np.argmax(state[CLASS_COLS]))]


class MatchRecorder:
    """Counts for one match, from the learner's side.  Call ``before_step`` with the decision
    and ``after_step`` with the next agent_states."""

    def __init__(self, learner_names: list[str], opponent_names: list[str]):
        self.learner_names, self.opponent_names = learner_names, opponent_names
        self.counts: Counter = Counter()
        self.first_reached: dict[str, float] = {}
        self.margins: dict[str, float] = {}
        self._objectives = (ObjectiveMonitor(), ObjectiveMonitor())
        self._movement = MovementMonitor()
        self._stalls = StallMonitor()
        self._last_move: dict[str, tuple[np.ndarray, str]] = {}
        self._pending = None

    def before_step(self, learner_obs, opponent_obs, labels, actions, student_actions, teacher_drove) -> None:
        view = next(iter(learner_obs.values()))
        states, elapsed = view["agent_states"], (1.0 - float(view["team_state"][2])) * MATCH_SECONDS
        learner_rows = [unit_index(n) for n in self.learner_names]
        self._objectives[0].observe(view["graphic"], states, learner_rows)
        if opponent_obs:
            other = next(iter(opponent_obs.values()))
            self._objectives[1].observe(other["graphic"], other["agent_states"], [unit_index(n) for n in self.opponent_names])

        c = self.counts
        roles = {name: _role(states[unit_index(name)]) for name in learner_obs}
        for name in learner_obs:
            label, role = labels[name], roles[name]
            controller = "teacher" if teacher_drove[name] else "student"
            c[f"ticks/{controller}"] += 1
            c[f"ticks/role_{role}"] += 1
            c["labels/valid"] += label.valid
            c["labels/set_size"] += int(label.mask.sum())
            if label.valid and name in student_actions:  # does the student agree where it was asked?
                hit = bool(label.mask[int(direction_vector_to_idx(student_actions[name]))])
                for key in ("all", f"role_{role}", f"kind_{label.kind}", f"{controller}_driving"):
                    c[f"agree_hit/{key}"] += hit
                    c[f"agree_n/{key}"] += 1
            self._count_reversal(name, np.asarray(actions[name], np.float32), controller)

        bucket = next(f"{lo}_{hi}s" if hi < 10_000 else f"{lo}s_plus" for lo, hi in TIME_BUCKETS if lo <= elapsed < hi)
        c[f"bucket_ticks/{bucket}"] += 1
        for role in roles.values():
            c[f"bucket_units/{role}/{bucket}"] += 1
        present = Counter(roles.values())
        for role, needed in (("hunter", 1), ("hunter", 2), ("hunter", 3), ("carrier", 1)):
            key = f"first_{role}{needed}_s"
            if key not in self.first_reached and present[role] >= needed:
                self.first_reached[key] = round(elapsed, 2)
        for at in MARGIN_AT:
            if f"margin_{at}s" not in self.margins and elapsed >= at:
                self.margins[f"margin_{at}s"] = round((float(view["team_state"][0]) - float(view["team_state"][1])) * 100, 1)
        self._pending = (view["graphic"], states.copy(), {n: actions[n] for n in learner_obs})

    def _count_reversal(self, name: str, move: np.ndarray, controller: str) -> None:
        norm = float(np.linalg.norm(move))
        previous = self._last_move.get(name)
        if norm > 1e-6:
            if previous is not None and float(move @ previous[0]) / norm < REVERSAL_COS:
                self.counts[f"reversals/{controller}"] += 1
            self._last_move[name] = (move / norm, controller)

    def after_step(self, next_states: np.ndarray | None) -> None:
        if self._pending is None or next_states is None:
            return
        graphic, before, actions = self._pending
        self._movement.observe(self.learner_names, before, next_states, actions)
        self._stalls.observe(graphic, before, next_states, actions, {n: unit_index(n) for n in actions})
        self._pending = None

    def summary(self) -> dict:
        self._movement.finish()
        counts = dict(self.counts)
        for side, monitor in zip(("learner", "opponent"), self._objectives):
            o = monitor.counts
            counts.update({
                f"{side}/unit_ticks": o.unit_ticks, f"{side}/carry_ticks": o.carry_ticks,
                f"{side}/pickups": o.pickups, f"{side}/deliveries": o.deliveries, f"{side}/cargo_lost": o.cargo_lost,
                f"{side}/approach_battery_sum": float(np.sum(o.battery_steps)), f"{side}/approach_battery_n": len(o.battery_steps),
                f"{side}/approach_storage_sum": float(np.sum(o.storage_steps)), f"{side}/approach_storage_n": len(o.storage_steps),
            })
        for kind, runs in (("idle", self._movement.result.idle), ("blocked", self._movement.result.blocked)):
            threshold = 300 if kind == "idle" else 12  # MovementMonitor.summary's incident lengths
            counts[f"movement/{kind}_incidents"] = sum(length >= threshold for length in runs)
            counts[f"movement/{kind}_ticks"] = sum(runs)
        counts["movement/unit_ticks"] = self._movement.result.unit_ticks
        counts.update({f"stall/{k}": v for k, v in asdict(self._stalls.counts).items()})
        return {"counts": counts, "first_reached": self.first_reached, "margins": self.margins}


def rates(counts: dict[str, float]) -> dict[str, float]:
    """Rates from (pooled) ``MatchRecorder`` counts."""
    c = Counter(counts)
    out = {}
    for key in [k for k in c if k.startswith("agree_n/")]:
        name = key.split("/", 1)[1]
        out[f"agree/{name}"] = c[f"agree_hit/{name}"] / max(1, c[key])
        out[f"agree_share/{name}"] = c[key] / max(1, c["agree_n/all"])
    ticks = c["ticks/teacher"] + c["ticks/student"]
    out["teacher_frac"] = c["ticks/teacher"] / max(1, ticks)
    for role in ROLES:
        out[f"units_{role}"] = N_TEAM * c[f"ticks/role_{role}"] / max(1, ticks)
        for key in [k for k in c if k.startswith("bucket_ticks/")]:
            bucket = key.split("/", 1)[1]
            out[f"units_{role}_{bucket}"] = c[f"bucket_units/{role}/{bucket}"] / c[key]
    out["label_valid_frac"] = c["labels/valid"] / max(1, ticks)
    out["label_set_size"] = c["labels/set_size"] / max(1, c["labels/valid"])
    for controller in ("student", "teacher"):
        if c[f"ticks/{controller}"]:
            out[f"reversals_per_1000/{controller}"] = 1000 * c[f"reversals/{controller}"] / c[f"ticks/{controller}"]
    for side in ("learner", "opponent"):
        unit_ticks = max(1, c[f"{side}/unit_ticks"])
        out[f"{side}/carry_frac"] = c[f"{side}/carry_ticks"] / unit_ticks
        for event in ("pickups", "deliveries", "cargo_lost"):
            out[f"{side}/{event}_per_1000"] = 1000 * c[f"{side}/{event}"] / unit_ticks
        out[f"{side}/delivery_rate"] = c[f"{side}/deliveries"] / max(1, c[f"{side}/pickups"])
        for target in ("battery", "storage"):
            out[f"{side}/approach_{target}"] = c[f"{side}/approach_{target}_sum"] / max(1, c[f"{side}/approach_{target}_n"])
    movement_ticks = max(1, c["movement/unit_ticks"])
    for kind in ("idle", "blocked"):
        out[f"movement/{kind}_per_1000"] = 1000 * c[f"movement/{kind}_incidents"] / movement_ticks
        out[f"movement/{kind}_tick_frac"] = c[f"movement/{kind}_ticks"] / movement_ticks
    stall = StallCounts(**{f.name: int(c[f"stall/{f.name}"]) for f in fields(StallCounts)})
    out.update(stall.summary("stall/"))
    return out


def summarize_matches(summaries: list[dict]) -> dict[str, float]:
    """Outcome, pooled rates, and per-match means of the timing fields."""
    margins = [s["learner_score"] - s["opponent_score"] for s in summaries]
    pooled = Counter()
    for s in summaries:
        pooled.update(s["counts"])
    out = {
        "matches": len(summaries),
        "mean_margin": float(np.mean(margins)),
        "margin_se": float(np.std(margins) / math.sqrt(len(margins))) if len(margins) > 1 else 0.0,
        "win_rate": float(np.mean([s["result"] == 1 for s in summaries])),
        "draw_rate": float(np.mean([s["result"] == 0 for s in summaries])),
        "rows": int(sum(s["rows"] for s in summaries)),
        "match_seconds": float(np.mean([s["elapsed_s"] for s in summaries])),
        "student_version_lag": float(np.mean([s["version_lag"] for s in summaries if "version_lag" in s] or [0.0])),
        **rates(pooled),
    }
    for field in ("first_reached", "margins"):
        keys = {k for s in summaries for k in s[field]}
        for key in sorted(keys):
            values = [s[field][key] for s in summaries if key in s[field]]
            out[key] = float(np.mean(values))
            if field == "first_reached":
                out[key.replace("_s", "_reached_frac")] = len(values) / len(summaries)
    return out


class LabelAccuracy:
    """Label accuracy by role and label kind, summed on-device (no host sync until ``pop``)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.hits = torch.zeros(len(ROLES), len(KINDS), device=self.device)
        self.labels = torch.zeros(len(ROLES), len(KINDS), device=self.device)

    @torch.no_grad()
    def add(self, logits: torch.Tensor, masks: torch.Tensor, classes: torch.Tensor, kinds: torch.Tensor) -> None:
        """logits [B, 5, 8], masks bool [B, 5, 8], classes [B, 5, 3] one-hot, kinds [B, 5]."""
        valid = masks.any(dim=-1).float()
        hit = masks.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1).float() * valid
        kind_onehot = torch.nn.functional.one_hot(kinds.long(), len(KINDS)).float()  # [B, 5, K]
        role_kind = classes.unsqueeze(-1) * kind_onehot.unsqueeze(-2)  # [B, 5, 3, K]
        self.hits += (hit[..., None, None] * role_kind).sum(dim=(0, 1))
        self.labels += (valid[..., None, None] * role_kind).sum(dim=(0, 1))

    def pop(self, prefix: str) -> dict[str, float]:
        hits, labels = self.hits.cpu().numpy(), self.labels.cpu().numpy()
        self.reset()
        total = labels.sum()
        if not total:
            return {}
        out = {f"{prefix}/accuracy": float(hits.sum() / total), f"{prefix}/labels": float(total)}
        for axis, names, tag in ((1, ROLES, "role"), (0, KINDS, "kind")):
            h, n = hits.sum(axis=axis), labels.sum(axis=axis)
            for name, hi, ni in zip(names, h, n):
                if ni and name:
                    out[f"{prefix}/accuracy_{tag}_{name}"] = float(hi / ni)
                    out[f"{prefix}/share_{tag}_{name}"] = float(ni / total)
        return out


class MetricsLog:
    """Every record goes to TensorBoard and to ``metrics.jsonl`` (one JSON object per line:
    ``{"kind", "step", "time", ...}``), which examples/distill_report.py reads."""

    def __init__(self, out_dir: Path):
        from blackout_env.train.tb_logger import TBLogger

        self.tb = TBLogger(out_dir / "tb")
        self._file = open(out_dir / "metrics.jsonl", "a")

    def write(self, kind: str, step: int, stats: dict[str, float]) -> None:
        self.tb.scalars(kind, {k: v for k, v in stats.items() if isinstance(v, (int, float))}, step)
        self._file.write(json.dumps({"kind": kind, "step": step, "time": round(time.time(), 1), **stats}) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
