"""Simulator collector for distillation: CollectorPool's interface, matches played by the bit-exact sim.

The Unity collector (collect.py) spends its CPU on one student forward per decision and per match.  Here every
worker job plays ``group`` matches in lockstep in a ``BatchSim``: the student (and ``self`` / ``ckpt:``
opponents) run as one batched forward per decision over all games (sim/batch.py ``NetPolicy``), the act-teacher
and heuristic opponents per game on ``obs_from_raw`` dicts (exactly the env's observations).  Rows, labels and
summaries are built by collect.py's own helpers (``_labelled_row``, ``bin_labels``, ``MatchRecorder``), so the
shards are the Unity collector's format and run.py's learner is unchanged (``run.py --sim``).

Differences from Unity: no partial decision windows (the sim decides all units every tick), the student is the
batched ``NetPolicy`` greedy decode (same argmax bin + angle offset as ``inference.py``'s logits path, batched
GEMM rounding), matches are stopped by the same "no battery after an absorption" rule.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np

from blackout_env.env.constants import team_a_agents, team_b_agents
from blackout_env.heuristics.teach import label_action
from blackout_env.train.distill.collect import TEACHERS, MatchTask, _labelled_row, bin_labels
from blackout_env.train.distill.dataset import EXTRA_FIELDS, FIELDS, save_shard
from blackout_env.train.distill.metrics import MatchRecorder
from blackout_env.train.distill.opponents import SELF, make_opponent, opponent_label

CKPT = "ckpt:"
_NETS: dict = {}  # (path, mtime) -> NetPolicy (worker-local cache)


def _net(path: str):
    from blackout_env.sim.batch import NetPolicy

    key = (path, Path(path).stat().st_mtime)
    pol = _NETS.get(key)
    if pol is None:
        for k in [k for k in _NETS if k[0] == path]:
            del _NETS[k]
        device = os.environ.get("BLACKOUT_SIM_NET_DEVICE", "cpu")
        pol = _NETS[key] = NetPolicy.from_checkpoint(path, device=device)
    return pol


def _version(path: str | None) -> int:
    if path is None:
        return 0
    from blackout_env.train.distill.inference import checkpoint_meta

    return int(checkpoint_meta(path).get("version", 0))


def _ckpt_path(descriptor: str) -> str | None:
    if not descriptor.startswith(CKPT):
        return None
    path = descriptor[len(CKPT):]
    return path if path.endswith(".pt") else None


def play_group(tasks: list[MatchTask]) -> list[tuple[dict, dict]]:
    """Plays ``tasks`` in lockstep; returns [(rows, summary)] in task order (collect.play_labelled_match's output)."""
    import torch

    from blackout_env.sim.batch import N_AGENT_COLS, N_CH, BatchSim
    from blackout_env.sim.batch import batteries_in_play as sim_batteries
    from blackout_env.sim.mapdata import H, W
    from blackout_env.sim.state import SimState, obs_from_raw

    torch.set_num_threads(1)
    n = len(tasks)
    bs = BatchSim([SimState.from_seed(t.seed) for t in tasks])
    learner_team = np.array([1 if t.swap else 0 for t in tasks])
    names = [team_b_agents() if t.swap else team_a_agents() for t in tasks]
    opp_names = [team_a_agents() if t.swap else team_b_agents() for t in tasks]
    student_path = next((t.student for t in tasks if t.student is not None), None)
    student = _net(student_path) if student_path is not None else None
    version = _version(student_path)
    teachers = [TEACHERS[t.teacher]() for t in tasks]
    for t in teachers:
        if hasattr(t, "reset"):
            t.reset()
    # opponents: batched nets (self / ckpt:<path>.pt) or one heuristic / policy instance per game
    opp_net, opp_obj = [None] * n, [None] * n
    for j, t in enumerate(tasks):
        path = _ckpt_path(t.opponent)
        if t.opponent == SELF:
            if student is not None:
                opp_net[j] = student_path
            else:
                opp_obj[j] = TEACHERS[t.teacher]()
        elif path is not None and Path(path).exists():
            opp_net[j] = path
        else:
            opp_obj[j] = make_opponent(t.opponent, t.seed)
        if opp_obj[j] is not None and hasattr(opp_obj[j], "reset"):
            opp_obj[j].reset()
    recorders = [MatchRecorder(names[j], opp_names[j]) for j in range(n)]
    rngs = [np.random.default_rng(t.seed) for t in tasks]
    rows = [{f: [] for f in FIELDS + EXTRA_FIELDS} for _ in range(n)]
    done = np.zeros(n, bool)
    last_abs = np.full(n, np.nan)
    elapsed = np.zeros(n)
    tick = 0
    F = np.float32
    graphic = np.zeros((n, 2, N_CH, H, W), F)
    team = np.zeros((n, 2, 4), F)
    agents = np.zeros((n, 2, 10, N_AGENT_COLS), F)
    moves = np.zeros((n, 10, 2), F)
    need_student = student is not None and any(t.beta < 1.0 for t in tasks)
    while True:
        live = np.nonzero(~done & ~bs.over)[0]
        if not len(live):
            break
        bs.render(live, graphic, team, agents)
        inputs = (graphic, team, agents)
        moves[:] = 0.0
        # batched student moves for the learner team of every live game
        stud = {}
        if need_student:
            for tm in (0, 1):
                sel = np.nonzero(learner_team[live] == tm)[0]
                if len(sel):
                    sub = tuple(x[sel] for x in inputs)
                    mv = student.moves(bs, live[sel], (tm,), sub)[tm]
                    for k, j in enumerate(live[sel]):
                        stud[int(j)] = mv[k]
        # batched net opponents
        net_mv = {}
        for path in {opp_net[j] for j in live if opp_net[j] is not None}:
            pol = _net(path)
            for tm in (0, 1):
                sel = np.nonzero([(opp_net[j] == path and 1 - learner_team[j] == tm) for j in live])[0]
                if len(sel):
                    sub = tuple(x[sel] for x in inputs)
                    mv = pol.moves(bs, live[sel], (tm,), sub)[tm]
                    for k, j in enumerate(live[sel]):
                        net_mv[int(j)] = mv[k]
        for k, j in enumerate(live):
            j = int(j)
            task = tasks[j]
            raw_state, raw_map = bs.raw(j)
            obs = obs_from_raw(raw_state, raw_map)
            lobs = {a: obs[a] for a in names[j]}
            oobs = {a: obs[a] for a in opp_names[j]}
            lt = int(learner_team[j])
            off = 5 * lt
            ooff = 5 * (1 - lt)
            if opp_obj[j] is not None:
                acts = opp_obj[j].act(oobs)
                for a, mv in acts.items():
                    moves[j, int(a.split("_")[1])] = np.clip(np.asarray(mv, F), -1, 1)
            else:
                moves[j, ooff:ooff + 5] = net_mv[j]
            teacher = teachers[j]
            compass = teacher.teach(lobs)
            tmoves = getattr(teacher, "last_actions", None)
            labels = compass if task.bins_are_compass else bin_labels(compass, tmoves, task)
            sacts = {}
            if j in stud and task.beta < 1.0:
                sacts = {a: stud[j][int(a.split("_")[1]) - off] for a in names[j]}
            drove = {a: a not in sacts or rngs[j].random() < task.beta for a in names[j]}
            actions = {}
            for a in names[j]:
                mv = (tmoves[a] if tmoves is not None else label_action(compass[a])) if drove[a] else sacts[a]
                actions[a] = np.asarray(mv, F)
                moves[j, int(a.split("_")[1])] = np.clip(actions[a], -1, 1)
            recorders[j].before_step(lobs, oobs, labels, actions, sacts, drove,
                                     compass=None if task.bins_are_compass else compass, teacher_moves=tmoves)
            row = _labelled_row(lobs, labels, actions, sacts, drove, task.action_bins, teacher_moves=tmoves,
                                focus=getattr(teacher, "focus", None))
            row["tick"] = np.int32(tick)
            for f, v in row.items():
                rows[j][f].append(v)
            elapsed[j] = (1.0 - float(team[k, lt, 2])) * 420.0
        bs.step(moves, live)
        tick += 1
        # recorder after_step + early stop (no battery left after an absorption), on the env's own observation
        for j in live:
            j = int(j)
            if bs.over[j]:
                recorders[j].after_step(None)
                continue
            view = obs_from_raw(*bs.raw(j), agents=[names[j][0]])[names[j][0]]
            recorders[j].after_step(view["agent_states"])
            ab = float(view["team_state"][3])
            if tasks[j].stop_when_exhausted and not np.isnan(last_abs[j]) and ab > last_abs[j] + 1e-6:
                if sim_batteries(bs, np.array([j]))[0] == 0:
                    done[j] = True
            last_abs[j] = ab
    out = []
    for j, task in enumerate(tasks):
        lt = int(learner_team[j])
        sc = bs.scores[j]
        ls, os_ = float(sc[lt]), float(sc[1 - lt])
        summary = {
            **asdict(task),
            "rows": len(rows[j]["labels"]),
            "learner_score": ls, "opponent_score": os_,
            "result": 0 if ls == os_ else (1 if ls > os_ else -1),
            "elapsed_s": round(float(elapsed[j]), 1),
            "opponent_played": opponent_label(task.opponent, opp_obj[j]) if opp_obj[j] is not None else task.opponent,
            "student_version": version,
            "sim": True,
            **recorders[j].summary(),
        }
        arrays = {f: np.stack(v) if v else np.zeros((0,)) for f, v in rows[j].items()}
        out.append((arrays, summary))
    return out


def _run_group(tasks: list[MatchTask]) -> list[dict]:
    started = time.time()
    results = play_group(tasks)
    wall = round(time.time() - started, 1)
    summaries = []
    for (rows, summary), task in zip(results, tasks):
        summary.update(wall_s=wall, worker_pid=os.getpid(), group=len(tasks))
        if task.shard is not None and summary["rows"]:
            save_shard(Path(task.shard), rows, meta=summary)
        summaries.append(summary)
    return summaries


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)
    threading.Thread(target=_exit_with_parent, args=(os.getppid(),), daemon=True).start()


def _exit_with_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(5.0)
    os._exit(0)


class SimCollectorPool:
    """CollectorPool's interface (submit / poll / drain / in_flight / workers) over sim worker processes; submitted
    tasks are batched into jobs of up to ``group`` matches (eval tasks in their own jobs)."""

    def __init__(self, workers: int, group: int = 32):
        self.workers = workers
        self.group = group
        import multiprocessing as mp

        self._executor = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        self._pending: dict[Future, list[MatchTask]] = {}
        self._buffer: list[MatchTask] = []

    @property
    def in_flight(self) -> int:
        return len(self._buffer) + sum(len(v) for v in self._pending.values())

    def _flush(self, force: bool = False) -> None:
        """Eval tasks go out at once (their own job); collect tasks in full groups, or whatever is buffered when no job
        is running at all (never idle with work queued)."""
        while self._buffer:
            evals = [t for t in self._buffer if t.tag != "collect"]
            if evals:
                batch = evals[:self.group]
            elif force or len(self._buffer) >= self.group or not self._pending:
                batch = self._buffer[:self.group]
            else:
                return
            ids = {id(t) for t in batch}
            self._buffer = [t for t in self._buffer if id(t) not in ids]
            self._pending[self._executor.submit(_run_group, batch)] = batch

    def _idle_workers(self) -> int:
        return self.workers - len(self._pending)

    def submit(self, task: MatchTask) -> None:
        self._buffer.append(task)
        if len(self._buffer) >= self.group:
            self._flush()

    def poll(self, timeout: float = 0.0) -> list[dict]:
        self._flush()
        if not self._pending:
            return []
        done, _ = wait(self._pending, timeout=timeout, return_when=FIRST_COMPLETED)
        out = []
        for f in done:
            del self._pending[f]
            out += f.result()
        self._flush()
        return out

    def drain(self, keep=lambda task: False) -> list[dict]:
        self._buffer = [t for t in self._buffer if keep(t)]
        self._flush(force=True)
        out = []
        for f in wait(list(self._pending)).done:
            out += f.result()
        self._pending = {}
        return out

    def close(self) -> None:
        self._executor.shutdown(cancel_futures=True)

    def __enter__(self) -> "SimCollectorPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
