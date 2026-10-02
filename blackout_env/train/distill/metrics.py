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
from blackout_env.model.direction_bins import bins_of_width, move_angle, vector_to_bin, wrap_angle
from blackout_env.model.my_policy import N_DIRECTIONS, direction_vector_to_idx
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
CAMP_TILES = 3.0  # an enemy Hunter this close to our base (outside its own) is camping our exit
NEAR_BASE_TILES = 4.0  # a death this close to our base = killed on the way out
RESPAWN_JUMP = 0.15  # normalised position jump in one tick = died and respawned (MovementMonitor's rule)
SPAWN_ALLY_CHANNEL = 4
# P11: the opening race to the central sanctuary is settled by single 0.04 s ticks, and only the
# first ~10 s are precision-critical (docs/experiments_index_ko.md, P10 analysis); 2 degrees is
# the angle error the simulator showed still keeps tied races
OPENING_SECONDS = 10.0
ANGLE_TOL_DEG = 2.0
ANGLE_SPLITS = ("all", "opening")


def _move_bin(move: np.ndarray, bins: int) -> int:
    """The label index a continuous move falls in: compass direction, or K-bin heading."""
    return int(direction_vector_to_idx(move)) if bins == N_DIRECTIONS else int(vector_to_bin(move, bins))


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
        self._base_cells: np.ndarray | None = None  # our base (4x4 corner around our spawn), tiles
        self._camped = False

    def before_step(self, learner_obs, opponent_obs, labels, actions, student_actions, teacher_drove, compass=None,
                    teacher_moves=None) -> None:
        """``labels`` may be K-bin (P4): ``agree`` then asks whether the student's move falls in
        the K-bin set.  ``compass`` -- the teacher's own 8-direction labels, given when ``labels``
        were re-expressed over K bins -- adds ``agree8``: the student's move snapped to the
        compass, in the 8-direction set (the K = 8 meaning of ``agree``, comparable across K).
        ``teacher_moves`` -- an act-teacher's raw moves -- adds the student's heading error against
        them (P11: ``angle_err_deg``, ``angle_within2``, overall and in the first OPENING_SECONDS;
        a student standing where the teacher moves counts 180 degrees)."""
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
                hit = bool(label.mask[_move_bin(student_actions[name], len(label.mask))])
                hit8 = bool(compass[name].mask[int(direction_vector_to_idx(student_actions[name]))]) if compass is not None else None
                for key in ("all", f"role_{role}", f"kind_{label.kind}", f"{controller}_driving"):
                    c[f"agree_hit/{key}"] += hit
                    c[f"agree_n/{key}"] += 1
                    if hit8 is not None:
                        c[f"agree8_hit/{key}"] += hit8
                        c[f"agree8_n/{key}"] += 1
                if teacher_moves is not None and label.kind == "act" and name in teacher_moves:
                    self._count_angle(student_actions[name], teacher_moves[name], elapsed)
            elif name in student_actions:  # the teacher stands here: does the student?
                c["agree_hit/stand"] += not np.any(student_actions[name])
                c["agree_n/stand"] += 1
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
        self._observe_camping(view["graphic"], states)
        self._pending = (view["graphic"], states.copy(), {n: actions[n] for n in learner_obs})

    def _count_angle(self, student_move, teacher_move, elapsed: float) -> None:
        teacher = move_angle(np.clip(np.asarray(teacher_move, np.float32), -1.0, 1.0))
        if math.isnan(teacher):
            return
        student = move_angle(student_move)
        err = 180.0 if math.isnan(student) else abs(math.degrees(wrap_angle(student - teacher)))
        for split in ANGLE_SPLITS[: 1 + (elapsed < OPENING_SECONDS)]:
            self.counts[f"angle_err_sum/{split}"] += err
            self.counts[f"angle_within2_hit/{split}"] += err <= ANGLE_TOL_DEG
            self.counts[f"angle_n/{split}"] += 1

    def _tiles_to_base(self, graphic: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Straight-line tiles from each position to our base."""
        if self._base_cells is None:
            from blackout_env.heuristics.v17_planner import StrategicHeuristicV17

            spawn = np.argwhere(graphic[..., SPAWN_ALLY_CHANNEL] > 0.5)
            origin = tuple(int(v) for v in spawn[0]) if len(spawn) else None
            self._base_cells = np.argwhere(StrategicHeuristicV17._base_mask(origin, graphic.shape[:2])).astype(np.float64)
        if not len(self._base_cells):
            return np.full(len(positions), np.inf)
        h, w = graphic.shape[:2]
        cells = np.stack([(1.0 - positions[:, 1]) * 0.5 * h - 0.5, (positions[:, 0] + 1.0) * 0.5 * w - 0.5], axis=1)
        return np.sqrt(((cells[:, None, :] - self._base_cells[None]) ** 2).sum(-1)).min(1)

    def _observe_camping(self, graphic: np.ndarray, states: np.ndarray) -> None:
        hunters = [i for i in range(len(states)) if states[i, 2] < 0 and _role(states[i]) == "hunter"]
        self._camped = bool(hunters) and bool((self._tiles_to_base(graphic, states[hunters, :2]) <= CAMP_TILES).any())
        self.counts["camp/ticks"] += self._camped
        self.counts["camp/observed_ticks"] += 1

    def _count_reversal(self, name: str, move: np.ndarray, controller: str) -> None:
        # np.linalg.norm, not math.hypot: an exact 135-degree turn between compass moves sits on
        # REVERSAL_COS, so float32 rounding decides it -- keep the rounding earlier runs counted with
        norm = float(np.linalg.norm(move))
        previous = self._last_move.get(name)
        if norm > 1e-6:
            if previous is not None:
                cos = float(move @ previous[0]) / norm
                if cos < REVERSAL_COS:
                    self.counts[f"reversals/{controller}"] += 1
                # heading change between consecutive moves: K-bin jitter shows up here, not as reversals
                self.counts[f"turn_deg/{controller}"] += math.degrees(math.acos(min(1.0, max(-1.0, cos))))
                self.counts[f"turn_n/{controller}"] += 1
            self._last_move[name] = (move / norm, controller)

    def after_step(self, next_states: np.ndarray | None) -> None:
        if self._pending is None or next_states is None:
            return
        graphic, before, actions = self._pending
        rows = [unit_index(n) for n in actions]
        died = [r for r in rows if math.hypot(float(next_states[r, 0] - before[r, 0]), float(next_states[r, 1] - before[r, 1])) > RESPAWN_JUMP]
        if died:  # respawned at spawn: where did it die, and was our exit camped?
            near = self._tiles_to_base(graphic, before[died, :2]) <= NEAR_BASE_TILES
            for r, at_base in zip(died, near):
                role = _role(before[r])
                self.counts[f"deaths/{role}"] += 1
                self.counts[f"deaths_near_base/{role}"] += bool(at_base)
                self.counts["deaths_while_camped"] += self._camped
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
    for key in [k for k in c if k.startswith("agree8_n/")]:  # K-bin labels: the 8-way agreement too
        name = key.split("/", 1)[1]
        out[f"agree8/{name}"] = c[f"agree8_hit/{name}"] / max(1, c[key])
    for split in ANGLE_SPLITS:  # P11: the student's heading vs the act-teacher's raw one
        if c[f"angle_n/{split}"]:
            out[f"angle_err_deg/{split}"] = c[f"angle_err_sum/{split}"] / c[f"angle_n/{split}"]
            out[f"angle_within2/{split}"] = c[f"angle_within2_hit/{split}"] / c[f"angle_n/{split}"]
    ticks = c["ticks/teacher"] + c["ticks/student"]
    out["teacher_frac"] = c["ticks/teacher"] / max(1, ticks)
    for role in ROLES:
        out[f"units_{role}"] = N_TEAM * c[f"ticks/role_{role}"] / max(1, ticks)
        for key in [k for k in c if k.startswith("bucket_ticks/")]:
            bucket = key.split("/", 1)[1]
            out[f"units_{role}_{bucket}"] = c[f"bucket_units/{role}/{bucket}"] / c[key]
    deaths = sum(c[f"deaths/{role}"] for role in ROLES)
    out["deaths_per_1000"] = 1000 * deaths / max(1, ticks)
    out["deaths_near_base_frac"] = sum(c[f"deaths_near_base/{role}"] for role in ROLES) / max(1, deaths)
    out["deaths_while_camped_frac"] = c["deaths_while_camped"] / max(1, deaths)
    for role in ROLES:
        out[f"deaths_per_1000/{role}"] = 1000 * c[f"deaths/{role}"] / max(1, c[f"ticks/role_{role}"])
    out["camped_frac"] = c["camp/ticks"] / max(1, c["camp/observed_ticks"])
    out["label_valid_frac"] = c["labels/valid"] / max(1, ticks)
    out["label_set_size"] = c["labels/set_size"] / max(1, c["labels/valid"])
    for controller in ("student", "teacher"):
        if c[f"ticks/{controller}"]:
            out[f"reversals_per_1000/{controller}"] = 1000 * c[f"reversals/{controller}"] / c[f"ticks/{controller}"]
        if c[f"turn_n/{controller}"]:
            out[f"mean_turn_deg/{controller}"] = c[f"turn_deg/{controller}"] / c[f"turn_n/{controller}"]
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
    """Label accuracy by role and label kind, summed on-device (no host sync until ``pop``).

    Given the teacher's raw ``angles`` (P11; NaN = none), also the heading error of what the
    student would execute -- its arg max bin's centre plus that bin's offset (``offsets``; none =
    the bin centres, so a bin-only student is measured the same way) -- overall and in the first
    OPENING_SECONDS (``elapsed`` seconds of play per row): ``angle_err_deg`` (mean absolute),
    ``angle_within2`` (share within ANGLE_TOL_DEG), ``angle_bin_hit`` (arg max = the bin holding
    the angle) and ``angle_res_err_deg`` (the offset of that bin against the angle's residual:
    the angle head alone, whatever bin won)."""

    ANGLE_STATS = ("n", "err", "within2", "bin_hit", "res_err")

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.hits = torch.zeros(len(ROLES), len(KINDS), device=self.device)
        self.labels = torch.zeros(len(ROLES), len(KINDS), device=self.device)
        self.angle = torch.zeros(len(ANGLE_SPLITS), len(self.ANGLE_STATS), device=self.device)

    @torch.no_grad()
    def add(self, logits: torch.Tensor, masks: torch.Tensor, classes: torch.Tensor, kinds: torch.Tensor,
            angles: torch.Tensor | None = None, offsets: torch.Tensor | None = None,
            elapsed: torch.Tensor | None = None) -> None:
        """logits [B, 5, A], masks bool [B, 5, A] (A = 8, 9, or K bins [+ STAND]), classes [B, 5, 3]
        one-hot, kinds [B, 5]; angles [B, 5] radians (NaN = none), offsets [B, 5, K] radians,
        elapsed [B] seconds."""
        if angles is not None:
            self._add_angles(logits, angles, offsets, elapsed)
        valid = masks.any(dim=-1).float()
        hit = masks.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1).float() * valid
        kind_onehot = torch.nn.functional.one_hot(kinds.long(), len(KINDS)).float()  # [B, 5, K]
        role_kind = classes.unsqueeze(-1) * kind_onehot.unsqueeze(-2)  # [B, 5, 3, K]
        self.hits += (hit[..., None, None] * role_kind).sum(dim=(0, 1))
        self.labels += (valid[..., None, None] * role_kind).sum(dim=(0, 1))

    def _add_angles(self, logits, angles, offsets, elapsed) -> None:
        k, _ = bins_of_width(logits.shape[-1])
        width = 2 * math.pi / k
        valid = torch.isfinite(angles)
        angle = torch.where(valid, angles.float(), torch.zeros_like(angles, dtype=torch.float32))
        best = logits.argmax(-1)
        stands = best >= k
        target = torch.round(angle / width).long() % k
        if offsets is None:
            offsets = torch.zeros(*logits.shape[:-1], k, device=logits.device)
        offsets = offsets.float()
        chosen = best.clamp_max(k - 1)
        pred = chosen.float() * width + offsets.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        err = torch.where(stands, torch.full_like(pred, 180.0), torch.rad2deg(wrap_angle(pred - angle).abs()))
        residual = wrap_angle(angle - target.float() * width)
        res_err = torch.rad2deg(wrap_angle(offsets.gather(-1, target.unsqueeze(-1)).squeeze(-1) - residual).abs())
        stats = torch.stack([valid.float(), err, (err <= ANGLE_TOL_DEG).float(), (best == target).float(), res_err], -1)
        stats = stats * valid.unsqueeze(-1)  # [B, 5, S]
        self.angle[0] += stats.sum(dim=(0, 1))
        if elapsed is not None:
            opening = (elapsed < OPENING_SECONDS).float()[:, None, None]
            self.angle[1] += (stats * opening).sum(dim=(0, 1))

    def pop(self, prefix: str) -> dict[str, float]:
        hits, labels, angle = self.hits.cpu().numpy(), self.labels.cpu().numpy(), self.angle.cpu().numpy()
        self.reset()
        out = {}
        for split, row in zip(ANGLE_SPLITS, angle):
            if row[0]:
                for name, value in zip(self.ANGLE_STATS[1:], row[1:]):
                    key = f"angle_{name}_deg" if name.endswith("err") else f"angle_{name}"
                    out[f"{prefix}/{key}/{split}"] = float(value / row[0])
                out[f"{prefix}/angle_n/{split}"] = float(row[0])
        total = labels.sum()
        if not total:
            return out
        out.update({f"{prefix}/accuracy": float(hits.sum() / total), f"{prefix}/labels": float(total)})
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
