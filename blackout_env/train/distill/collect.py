"""Labelled matches for distillation, played in the background by headless Unity workers.

In a match the learner's team is driven unit by unit: each tick every unit follows the teacher
with probability ``beta`` and the student net otherwise (DAgger's mixture policy).  ``beta = 1``
is the teacher alone, ``beta = 0`` the student alone.  Whatever drove, the teacher labels the
state; only the learner's team is recorded.  The opponent is a fixed heuristic or the heuristic
mixture.

Teachers (``--teacher``) are the relabelling heuristics in ``TEACHERS`` (memoryless: labels are a
function of the observation) or ``act:<opponent descriptor>`` (act_teacher.py): any policy, possibly
stateful, labelled with its own moves.  The teacher is stepped exactly once per learner decision
tick -- partial windows included -- so a stateful teacher's memory follows the played trajectory;
where it drives, an act-teacher's raw move is executed (beta = 1 reproduces the policy exactly).

Every full-window tick is recorded, labelled or not, in order, with ``tick`` = the index of the
learner decision step within the match (partial windows are counted but not recorded, so a jump
in ``tick`` marks them): recurrent students train on windows of consecutive rows.

K direction bins (``MatchTask.action_bins``, P4; model/direction_bins.py): the teachers still
label in 8 compass directions, and ``bin_labels`` re-expresses each label over the K bins before
it is recorded -- a set teacher's directions widen to the bins within ``label_width_deg`` of them,
an act-teacher's label becomes its *raw* heading's nearest bin +- ``act_arc_bins``.  Executed
moves are unchanged (a relabeller's ``label_action``, an act-teacher's raw move, the student's
bin heading); shards store bin indices for ``actions``/``student`` and K-bit label sets (uint64
for K > 8).  The recorder's ``agree`` is then "the student's heading falls in the K-bin set";
``agree8`` keeps the 8-way meaning (its move snapped to the compass, in the teacher's own
8-direction label) so runs at different K compare.  K = 8 is the old path, unchanged.

Continuous angles (P11): every row also keeps, per own unit, the act-teacher's raw heading
(``angles``, radians, canonical frame -- team B's mirrored ``t -> pi/2 - t`` exactly like its bin
label, ``direction_bins.mirror_angle``) and the student's executed heading (``student_angles``);
NaN where there is none (a relabelling teacher, an invalid label / standing, a student not asked).
The recorder compares the two (``angle_err_deg``, ``angle_within2``).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.env.team_frame import canonical_obs, is_team_b_view, mirror_direction_bits, mirror_direction_idx
from blackout_env.heuristics import (
    StrategicHeuristicV4Teach, StrategicHeuristicV17Teach, StrategicHeuristicV20Teach, StrategicHeuristicV21Teach,
)
from blackout_env.heuristics.teach import TeachLabel, label_action
from blackout_env.model.direction_bins import (
    SET_HALF_WIDTH_DEG, arc_mask, label_dtype, mirror_angle, mirror_bin, mirror_bits, move_angle, pack_bits,
    vector_to_bin, widen_mask,
)
from blackout_env.model.my_policy import N_DIRECTIONS, direction_vector_to_idx
from blackout_env.train.dead_segments import batteries_in_play
from blackout_env.train.distill.act_teacher import ACT_KIND, ZERO_MOVE, ActTeacher, check_act_teacher, is_act_teacher
from blackout_env.train.distill.dataset import EXTRA_FIELDS, FIELDS, KIND_INDEX, N_TEAM, encode_graphic, save_shard
from blackout_env.train.distill.metrics import MATCH_SECONDS, STAND, MatchRecorder
from blackout_env.train.distill.opponents import SELF, make_opponent, opponent_label

class _Teachers(dict):
    """name -> teacher factory.  Besides the listed relabelling teachers, any ``act:<descriptor>``
    key resolves to an ActTeacher factory (act_teacher.py) without being listed."""

    def __missing__(self, key):
        if is_act_teacher(key):
            return functools.partial(ActTeacher, key[len("act:"):])
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        return super().__contains__(key) or is_act_teacher(key)


TEACHERS = _Teachers({
    "v17-teach": StrategicHeuristicV17Teach,
    "v20-teach": StrategicHeuristicV20Teach,  # + spawn-camp counterplay (heuristics/v20_anticamp.py)
    "v21-teach": StrategicHeuristicV21Teach,  # + Hunter transforms only by routes clear of enemy Hunters
    "v4-teach": functools.partial(StrategicHeuristicV4Teach, target_hunters=2, target_carriers=1),
})


def teacher_arg(name: str) -> str:
    """argparse ``type`` for ``--teacher``: a TEACHERS name or ``act:<opponent descriptor>``."""
    import argparse

    try:
        if is_act_teacher(name):
            check_act_teacher(name)
        elif name not in TEACHERS:
            raise ValueError(f"choose from {', '.join(sorted(dict.keys(TEACHERS)))} or act:<opponent descriptor>")
    except (ValueError, KeyError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid teacher {name!r}: {exc}") from exc
    return name
ABSORPTION_IDX = 3  # team_state: time to the next absorption
MAX_EMPTY_STEPS = 200


@dataclass(frozen=True)
class MatchTask:
    seed: int
    swap: bool  # the learner plays team B
    opponent: str  # descriptor: heuristic id, "mixture", "ckpt:<path>", "self" ... (opponents.py)
    beta: float
    teacher: str
    student: str | None  # checkpoint path; None = teacher only (beta must be 1)
    shard: str | None  # where to write the labelled rows; None = play only
    stop_when_exhausted: bool = True
    tag: str = "collect"  # "collect" (DAgger data) or "eval" (student alone vs a fixed opponent)
    version: int = 0  # student version current when the task was queued (eval groups by it)
    inference: str = "iqn"  # how the student turns logits into a move (inference.py)
    action_bins: int = N_DIRECTIONS  # labels and recorded moves over K heading bins (bin_labels)
    label_width_deg: float = SET_HALF_WIDTH_DEG  # a set teacher's direction -> bins within this
    act_arc_bins: int = 0  # an act-teacher's heading -> its nearest bin and this many either side

    @property
    def bins_are_compass(self) -> bool:
        """K = 8 and labels as the teacher gives them: nothing to re-express."""
        return self.action_bins == N_DIRECTIONS and self.act_arc_bins == 0 and self.label_width_deg < 45.0


def bin_labels(labels: dict[str, TeachLabel], raw_moves: dict[str, np.ndarray] | None, task: MatchTask) -> dict[str, TeachLabel]:
    """The teacher's 8-direction labels over ``task.action_bins`` bins (see the module docstring).
    An act label uses the raw move (``raw_moves``, the env's [-1, 1] clip applied), not its
    8-way snap; an invalid label stays invalid (all False)."""
    k = task.action_bins
    out = {}
    for name, label in labels.items():
        if not label.valid:
            mask = np.zeros(k, bool)
        elif label.kind == ACT_KIND and raw_moves is not None:
            mask = arc_mask(np.clip(np.asarray(raw_moves[name], np.float32), -1.0, 1.0), k, task.act_arc_bins)
        else:
            mask = widen_mask(label.mask, k, task.label_width_deg)
        out[name] = dataclasses.replace(label, mask=mask)
    return out


def play_labelled_match(env, task: MatchTask, student) -> tuple[dict[str, np.ndarray], dict]:
    """Plays ``task``; returns the learner's labelled rows (dataset.py schema) and a summary
    (the task, outcome, and ``MatchRecorder`` counts)."""
    if student is None and task.beta < 1.0:
        raise ValueError("a match with beta < 1 needs a student")
    teacher = TEACHERS[task.teacher]()
    if task.opponent == SELF:  # a mirror match: the learner's own policy on the other side too
        opponent = student if student is not None else TEACHERS[task.teacher]()
    else:
        opponent = make_opponent(task.opponent, task.seed)
    learner_names = team_b_agents() if task.swap else team_a_agents()
    opponent_names = team_a_agents() if task.swap else team_b_agents()
    recorder = MatchRecorder(learner_names, opponent_names)
    rng = np.random.default_rng(task.seed)
    rows: dict[str, list] = {field: [] for field in FIELDS + EXTRA_FIELDS}
    tick = -1  # learner decision steps so far (partial windows included)

    obs, _ = env.reset(seed=task.seed)
    for policy in (teacher, opponent, student):
        if hasattr(policy, "reset"):
            policy.reset()
    final_info: dict = {}
    last_absorption = None
    empty_steps = 0
    elapsed = 0.0
    while env.agents:
        if not obs:
            obs, _, _, _, infos = env.step({})
            final_info = next(iter(infos.values()), final_info) if infos else final_info
            empty_steps += 1
            if empty_steps > MAX_EMPTY_STEPS:
                raise RuntimeError(f"empty observations for {MAX_EMPTY_STEPS}+ steps")
            continue
        empty_steps = 0
        learner_obs = {a: obs[a] for a in learner_names if a in obs}
        opponent_obs = {a: obs[a] for a in opponent_names if a in obs}
        actions = opponent.act(opponent_obs) if opponent_obs else {}

        full_window = len(learner_obs) == len(learner_names)  # partial decision windows are not recorded
        if learner_obs:
            tick += 1
            compass = teacher.teach(learner_obs)  # every tick: a stateful teacher must see them all
            # an act-teacher executes its own raw move; a relabeller follows its label
            teacher_moves = getattr(teacher, "last_actions", None)
            labels = compass if task.bins_are_compass else bin_labels(compass, teacher_moves, task)
            # a temporal student is stepped every tick, even where the teacher drives, so its memory
            # follows the whole match (the same ticks its training windows see)
            asked = student is not None and (task.beta < 1.0 or getattr(student, "is_temporal", False))
            student_actions = student.act(learner_obs) if asked else {}
            teacher_drove = {name: name not in student_actions or rng.random() < task.beta for name in learner_obs}
            for name in learner_obs:
                if not teacher_drove[name]:
                    actions[name] = student_actions[name]
                else:
                    actions[name] = teacher_moves[name] if teacher_moves is not None else label_action(compass[name])
            if full_window:
                recorder.before_step(learner_obs, opponent_obs, labels, actions, student_actions, teacher_drove,
                                     compass=None if task.bins_are_compass else compass, teacher_moves=teacher_moves)
                row = _labelled_row(learner_obs, labels, actions, student_actions, teacher_drove, task.action_bins,
                                    teacher_moves=teacher_moves)
                row["tick"] = np.int32(tick)
                for field, value in row.items():  # unlabelled ticks too: context / STAND targets
                    rows[field].append(value)
            elapsed = (1.0 - float(next(iter(learner_obs.values()))["team_state"][2])) * MATCH_SECONDS

        obs, _, terminations, _, infos = env.step(actions)
        final_info = next(iter(infos.values()), final_info) if infos else final_info
        learner_next = next((obs[a] for a in learner_names if a in obs), None) if obs else None
        recorder.after_step(learner_next["agent_states"] if learner_next is not None and full_window else None)
        if any(terminations.values()) or not obs:
            continue
        view = next(iter(obs.values()))
        absorption = float(view["team_state"][ABSORPTION_IDX])
        fired = last_absorption is not None and absorption > last_absorption + 1e-6
        last_absorption = absorption
        if task.stop_when_exhausted and fired and batteries_in_play(view["graphic"][None], view["agent_states"][None])[0] == 0:
            final_info = _scores_from(view, team_b=is_team_b_view(view["agent_states"]))
            break  # nothing left to score: the rest of the match cannot change the result

    learner_team = 1 if task.swap else 0
    scores = tuple(round(float(final_info.get(key, 0.0)) * 100, 1) for key in ("score_0", "score_1"))
    winner = final_info.get("winner")
    summary = {
        **asdict(task),
        "rows": len(rows["labels"]),
        "learner_score": scores[learner_team],
        "opponent_score": scores[1 - learner_team],
        "result": 0 if winner in (None, -1) else (1 if int(winner) == learner_team else -1),
        "elapsed_s": round(elapsed, 1),
        "opponent_played": opponent_label(task.opponent, opponent),
        **recorder.summary(),
    }
    arrays = {field: np.stack(values) if values else np.zeros((0,)) for field, values in rows.items()}
    return arrays, summary


def teacher_angle(label: TeachLabel, raw_move) -> float:
    """The act-teacher's raw heading behind ``label`` (radians, world frame, the env's [-1, 1]
    clip applied), NaN unless it is a valid act label with its raw move."""
    if raw_move is None or not label.valid or label.kind != ACT_KIND:
        return float("nan")
    return move_angle(np.clip(np.asarray(raw_move, np.float32), -1.0, 1.0))


def _labelled_row(learner_obs, labels, actions, student_actions, teacher_drove, bins: int = N_DIRECTIONS,
                  teacher_moves: dict | None = None) -> dict[str, np.ndarray]:
    """One tick in the canonical frame: own units in rows 0-4, labels and moves mirrored for
    team B (dataset.py schema).  ``bins`` > 8: labels are K-bit sets (uint64), moves bin indices.
    ``teacher_moves``: an act-teacher's raw moves, kept as ``angles`` (NaN without)."""
    view = next(iter(learner_obs.values()))
    team_b = is_team_b_view(view["agent_states"])
    per_unit = {
        "labels": np.zeros(2 * N_TEAM, dtype=label_dtype(bins)), "kinds": np.zeros(2 * N_TEAM, dtype=np.uint8),
        "actions": np.full(2 * N_TEAM, STAND, dtype=np.uint8), "student": np.full(2 * N_TEAM, STAND, dtype=np.uint8),
        "teacher_drove": np.zeros(2 * N_TEAM, dtype=bool),
        "angles": np.full(2 * N_TEAM, np.nan, dtype=np.float32),
        "student_angles": np.full(2 * N_TEAM, np.nan, dtype=np.float32),
    }
    for name, label in labels.items():
        i = unit_index(name)
        if bins == N_DIRECTIONS:
            per_unit["labels"][i] = np.packbits(label.mask, bitorder="little")[0]
        else:
            per_unit["labels"][i] = pack_bits(label.mask)
        per_unit["kinds"][i] = KIND_INDEX.get(label.kind, 0) if label.valid else 0
        per_unit["actions"][i] = _direction(actions[name], bins)
        if name in student_actions:
            per_unit["student"][i] = _direction(student_actions[name], bins)
            per_unit["student_angles"][i] = move_angle(student_actions[name])
        per_unit["angles"][i] = teacher_angle(label, None if teacher_moves is None else teacher_moves.get(name))
        per_unit["teacher_drove"][i] = teacher_drove[name]
    if team_b:
        if bins == N_DIRECTIONS:
            per_unit["labels"] = mirror_direction_bits(per_unit["labels"])
        else:
            per_unit["labels"] = mirror_bits(np.roll(per_unit["labels"], -N_TEAM), bins)
        for field in ("actions", "student"):
            moves = per_unit[field]
            if bins == N_DIRECTIONS:
                mirrored = mirror_direction_idx(np.minimum(moves, 7))
            else:
                mirrored = mirror_bin(np.minimum(moves, bins - 1), bins)
            per_unit[field] = np.where(moves == STAND, STAND, mirrored).astype(np.uint8)
        for field in ("angles", "student_angles"):  # the same reflection as the bins (NaN stays NaN)
            per_unit[field] = mirror_angle(per_unit[field]).astype(np.float32)
        per_unit = {field: np.roll(values, -N_TEAM) if field != "labels" else values for field, values in per_unit.items()}
    canonical = canonical_obs(view)
    return {
        "graphic": encode_graphic(canonical["graphic"]),
        "team_state": canonical["team_state"].astype(np.float32),
        "agent_states": canonical["agent_states"].astype(np.float32),
        **{field: values[:N_TEAM] for field, values in per_unit.items()},
    }


def _direction(move: np.ndarray, bins: int = N_DIRECTIONS) -> int:
    """Compass (or K-bin) index of a continuous move, STAND for no movement."""
    move = np.asarray(move, np.float32)
    if float(np.linalg.norm(move)) < ZERO_MOVE:
        return STAND
    return int(direction_vector_to_idx(move)) if bins == N_DIRECTIONS else int(vector_to_bin(move, bins))


def _scores_from(view, team_b: bool) -> dict:
    own, other = float(view["team_state"][0]), float(view["team_state"][1])
    score_0, score_1 = (other, own) if team_b else (own, other)
    winner = -1 if abs(score_0 - score_1) < 1e-6 else (0 if score_0 > score_1 else 1)
    return {"winner": winner, "score_0": score_0, "score_1": score_1}


# ------------------------------------------------------------------------------ worker pool

_ENV = None
_STUDENT: tuple[tuple, object, int] | None = None  # ((path, mtime, inference), policy, version)


def _init_worker(build: str, time_scale: float) -> None:
    global _ENV
    import torch
    from multiprocessing import util

    from blackout_env import BlackOutEnv

    torch.set_num_threads(1)  # one Unity process + one CPU forward per worker
    _ENV = BlackOutEnv(build, time_scale=time_scale, no_graphics=True,
                       additional_args=["-logFile", "/dev/null"], unity_shaping=False)
    # see examples/gauntlet_heuristics.py: pool workers exit through os._exit(), so close here
    util.Finalize(_ENV, _ENV.close, exitpriority=10)
    threading.Thread(target=_exit_with_parent, args=(os.getppid(),), daemon=True).start()


def _exit_with_parent(parent: int) -> None:
    """A worker whose trainer was killed is not told (a pool only shuts workers down when it
    closes), and it kept its Unity process alive: 8 such pairs sat idle for 6 hours next to D1.
    Poll the parent and close both when it is gone."""
    while os.getppid() == parent:
        time.sleep(5.0)
    try:
        _ENV.close()
    finally:
        os._exit(0)


def _load_student(path: str | None, inference: str = "iqn") -> tuple[object | None, int]:
    """The student at ``path`` and its published version, reloaded only when the file changed
    (the trainer replaces it atomically, so a match always plays one consistent version)."""
    global _STUDENT
    if path is None:
        return None, 0
    import torch

    from blackout_env.train.distill.inference import load_net, make_student

    key = (path, Path(path).stat().st_mtime, inference)
    if _STUDENT is None or _STUDENT[0] != key:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        _STUDENT = (key, make_student(load_net(ckpt["policy_state"]), inference), int(ckpt.get("distill", {}).get("version", 0)))
    return _STUDENT[1], _STUDENT[2]


def _run_task(task: MatchTask) -> dict:
    started = time.time()
    student, version = _load_student(task.student, task.inference)
    rows, summary = play_labelled_match(_ENV, task, student)
    summary.update(student_version=version, wall_s=round(time.time() - started, 1), worker_pid=os.getpid())
    if task.shard is not None and summary["rows"]:
        save_shard(Path(task.shard), rows, meta=summary)
    return summary


class CollectorPool:
    """``workers`` long-lived Unity processes playing submitted tasks in the background.

    ``submit`` never blocks; ``poll`` returns the summaries of matches finished so far (waiting
    up to ``timeout`` seconds for at least one)."""

    def __init__(self, build: str, workers: int, time_scale: float = 20.0):
        self.workers = workers
        self._executor = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(build, time_scale))
        self._pending: dict[Future, MatchTask] = {}

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    def submit(self, task: MatchTask) -> None:
        self._pending[self._executor.submit(_run_task, task)] = task

    def poll(self, timeout: float = 0.0) -> list[dict]:
        if not self._pending:
            return []
        done, _ = wait(self._pending, timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            del self._pending[future]
        return [future.result() for future in done]

    def drain(self, keep=lambda task: False) -> list[dict]:
        """Cancel queued tasks except those ``keep`` accepts, wait for the rest (running or kept),
        and return their summaries."""
        remaining = [f for f, task in self._pending.items() if keep(task) or not f.cancel()]
        self._pending = {}
        return [future.result() for future in wait(remaining).done]

    def close(self) -> None:
        self._executor.shutdown(cancel_futures=True)

    def __enter__(self) -> "CollectorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
