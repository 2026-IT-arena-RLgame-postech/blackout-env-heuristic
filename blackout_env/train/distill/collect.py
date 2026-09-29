"""Labelled matches for distillation, played in the background by headless Unity workers.

In a match the learner's team is driven unit by unit: each tick every unit follows the teacher
with probability ``beta`` and the student net otherwise (DAgger's mixture policy; both are
memoryless, so switching per tick is well defined).  ``beta = 1`` is the teacher alone,
``beta = 0`` the student alone.  Whatever drove, the teacher labels the state; only the
learner's team is recorded.  The opponent is a fixed heuristic or the heuristic mixture.
"""

from __future__ import annotations

import functools
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from blackout_env.env.constants import team_a_agents, team_b_agents, unit_index
from blackout_env.env.team_frame import canonical_obs, is_team_b_view, mirror_direction_bits
from blackout_env.heuristics import (
    HeuristicPolicyMixture, StrategicHeuristicV4Teach, StrategicHeuristicV17Teach, make_heuristic,
)
from blackout_env.heuristics.teach import label_action
from blackout_env.train.dead_segments import batteries_in_play
from blackout_env.train.distill.dataset import KIND_INDEX, N_TEAM, encode_graphic, save_shard

TEACHERS = {
    "v17-teach": StrategicHeuristicV17Teach,
    "v4-teach": functools.partial(StrategicHeuristicV4Teach, target_hunters=2, target_carriers=1),
}
ABSORPTION_IDX = 3  # team_state: time to the next absorption
CLASS_COLS = slice(9, 12)  # agent_states class one-hot: Collector, Hunter, Carrier
MAX_EMPTY_STEPS = 200


@dataclass(frozen=True)
class MatchTask:
    seed: int
    swap: bool  # the learner plays team B
    opponent: str  # heuristic id, or "mixture"
    beta: float
    teacher: str
    student: str | None  # checkpoint path; None = teacher only (beta must be 1)
    shard: str | None  # where to write the labelled rows; None = play only
    stop_when_exhausted: bool = True
    tag: str = "collect"  # "collect" (DAgger data) or "eval" (student alone vs a fixed opponent)
    version: int = 0  # student version current when the task was queued (eval groups by it)


def make_opponent(opponent: str, seed: int):
    return HeuristicPolicyMixture(seed=seed) if opponent == "mixture" else make_heuristic(opponent)


def play_labelled_match(env, task: MatchTask, student) -> tuple[dict[str, np.ndarray], dict]:
    """Plays ``task``; returns the learner's labelled rows (dataset.py schema) and a summary."""
    if student is None and task.beta < 1.0:
        raise ValueError("a match with beta < 1 needs a student")
    teacher = TEACHERS[task.teacher]()
    opponent = make_opponent(task.opponent, task.seed)
    learner_names = team_b_agents() if task.swap else team_a_agents()
    opponent_names = team_a_agents() if task.swap else team_b_agents()
    rng = np.random.default_rng(task.seed)

    rows: dict[str, list] = {"graphic": [], "team_state": [], "agent_states": [], "labels": [], "kinds": []}
    teacher_ticks = unit_ticks = recorded = labelled = 0
    class_ticks = np.zeros(3, dtype=np.int64)

    obs, _ = env.reset(seed=task.seed)
    for policy in (teacher, opponent, student):
        if hasattr(policy, "reset"):
            policy.reset()
    final_info: dict = {}
    last_absorption = None
    empty_steps = 0
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

        if learner_obs:
            labels = teacher.teach(learner_obs)
            student_actions = student.act(learner_obs) if student is not None and task.beta < 1.0 else {}
            for name in learner_obs:
                if name not in student_actions or rng.random() < task.beta:
                    actions[name] = label_action(labels[name])
                    teacher_ticks += 1
                else:
                    actions[name] = student_actions[name]
            unit_ticks += len(learner_obs)
            if len(learner_obs) == len(learner_names):  # partial decision windows are not recorded
                row = _labelled_row(learner_obs, labels)
                recorded += 1
                class_ticks += row["agent_states"][:N_TEAM, CLASS_COLS].sum(axis=0).round().astype(np.int64)
                if row["labels"].any():
                    labelled += 1
                    for field, value in row.items():
                        rows[field].append(value)

        obs, _, terminations, _, infos = env.step(actions)
        final_info = next(iter(infos.values()), final_info) if infos else final_info
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
        "rows": labelled,
        "learner_score": scores[learner_team],
        "opponent_score": scores[1 - learner_team],
        "result": 0 if winner in (None, -1) else (1 if int(winner) == learner_team else -1),
        "teacher_frac": teacher_ticks / max(1, unit_ticks),
        # mean units per class on the learner's team: the role ratio it actually played
        "collectors": class_ticks[0] / max(1, recorded), "hunters": class_ticks[1] / max(1, recorded),
        "carriers": class_ticks[2] / max(1, recorded),
    }
    arrays = {field: np.stack(values) if values else np.zeros((0,)) for field, values in rows.items()}
    return arrays, summary


def _labelled_row(learner_obs, labels) -> dict[str, np.ndarray]:
    """One tick in the canonical frame: own units in rows 0-4, labels mirrored for team B."""
    view = next(iter(learner_obs.values()))
    team_b = is_team_b_view(view["agent_states"])
    bits = np.zeros(2 * N_TEAM, dtype=np.uint8)
    kinds = np.zeros(2 * N_TEAM, dtype=np.uint8)
    for name, label in labels.items():
        bits[unit_index(name)] = np.packbits(label.mask, bitorder="little")[0]
        kinds[unit_index(name)] = KIND_INDEX.get(label.kind, 0) if label.valid else 0
    if team_b:
        bits = mirror_direction_bits(bits)
        kinds = np.roll(kinds, -N_TEAM)
    canonical = canonical_obs(view)
    return {
        "graphic": encode_graphic(canonical["graphic"]),
        "team_state": canonical["team_state"].astype(np.float32),
        "agent_states": canonical["agent_states"].astype(np.float32),
        "labels": bits[:N_TEAM],
        "kinds": kinds[:N_TEAM],
    }


def _scores_from(view, team_b: bool) -> dict:
    own, other = float(view["team_state"][0]), float(view["team_state"][1])
    score_0, score_1 = (other, own) if team_b else (own, other)
    winner = -1 if abs(score_0 - score_1) < 1e-6 else (0 if score_0 > score_1 else 1)
    return {"winner": winner, "score_0": score_0, "score_1": score_1}


# ------------------------------------------------------------------------------ worker pool

_ENV = None
_STUDENT: tuple[str, float, object, int] | None = None  # (path, mtime, policy, version)


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


def _load_student(path: str | None) -> tuple[object | None, int]:
    """The student at ``path`` and its published version, reloaded only when the file changed
    (the trainer replaces it atomically, so a match always plays one consistent version)."""
    global _STUDENT
    if path is None:
        return None, 0
    import torch

    from blackout_env.model.my_model import MyModel
    from blackout_env.model.my_policy import MyPolicy

    mtime = Path(path).stat().st_mtime
    if _STUDENT is None or _STUDENT[:2] != (path, mtime):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net = MyModel()
        net.load_state_dict(ckpt["policy_state"])
        _STUDENT = (path, mtime, MyPolicy(net.eval()), int(ckpt.get("distill", {}).get("version", 0)))
    return _STUDENT[2], _STUDENT[3]


def _run_task(task: MatchTask) -> dict:
    student, version = _load_student(task.student)
    rows, summary = play_labelled_match(_ENV, task, student)
    if task.shard is not None and summary["rows"]:
        save_shard(Path(task.shard), rows)
    return {**summary, "student_version": version}


class CollectorPool:
    """``workers`` long-lived Unity processes playing submitted tasks in the background.

    ``submit`` never blocks; ``poll`` returns the summaries of matches finished so far (waiting
    up to ``timeout`` seconds for at least one)."""

    def __init__(self, build: str, workers: int, time_scale: float = 20.0):
        self.workers = workers
        self._executor = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(build, time_scale))
        self._pending: set[Future] = set()

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    def submit(self, task: MatchTask) -> None:
        self._pending.add(self._executor.submit(_run_task, task))

    def poll(self, timeout: float = 0.0) -> list[dict]:
        if not self._pending:
            return []
        done, self._pending = wait(self._pending, timeout=timeout, return_when=FIRST_COMPLETED)
        return [future.result() for future in done]

    def drain(self) -> list[dict]:
        """Drop queued tasks, wait for the running ones, and return their summaries."""
        running = {future for future in self._pending if not future.cancel()}
        self._pending = set()
        return [future.result() for future in wait(running).done]

    def close(self) -> None:
        self._executor.shutdown(cancel_futures=True)

    def __enter__(self) -> "CollectorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
