"""
Cross-process state for the actor/inference/learner pipeline (see package docstring).

Kept intentionally tiny: only what actors need to know that only the learner's replay buffers
can determine (bootstrap phase, epsilon-anneal anchor) crosses process boundaries here. Every
other schedule (n_step/gamma/PER-beta annealing) is learner-internal only (used by
QMIXTrainer.train_step/_sample_batch on data already inside the learner process), so it stays
exactly as implemented in qmix_trainer.py -- nothing to duplicate.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue

from blackout_env.train.qmix_trainer import QMIXConfig


class SharedState:
    """Constructed once by the group launcher and passed (pickled, then shared via the
    multiprocessing.Value/Event machinery) to every actor/inference/learner process it spawns."""

    def __init__(self, ctx: mp.context.BaseContext) -> None:
        # 'l' = signed long -- total real env.step() calls across every actor in this group,
        # i.e. the distributed equivalent of QMIXTrainer.env_step_count. Incremented by each
        # actor once per env.step() (see actor.py), read by the learner every drain iteration.
        self.global_step = ctx.Value("l", 0)
        # Mirrors QMIXTrainer._bootstrapping: True while the learner's buffer_a/buffer_b are
        # still being filled purely from heuristic rollouts (phase 1). Actors read this before
        # every action-selection decision to decide whether to even contact inference_server.py.
        self.bootstrapping = ctx.Value("b", True)
        # Mirrors QMIXTrainer._phase2_epsilon_anchor_step: set exactly once, by the learner, the
        # moment `bootstrapping` flips True->False, so every actor's epsilon() (see
        # compute_epsilon below) starts exploring at eps_start instead of wherever global_step
        # already was.
        self.phase2_anchor_step = ctx.Value("l", 0)
        # Set by the launcher on shutdown (Ctrl+C, or the learner reaching total_env_steps) to
        # tell every actor/inference process to exit its main loop.
        self.stop = ctx.Event()

    def increment_step(self) -> int:
        with self.global_step.get_lock():
            self.global_step.value += 1
            return self.global_step.value


def put_until_stop(q, item, stop_event, timeout: float = 1.0) -> bool:
    """`q.put(item)`, but re-checks `stop_event` on a timeout instead of blocking forever.

    multiprocessing.Queue.put() is NOT the "append to an unbounded in-memory deque" operation
    it's easy to assume -- it waits on an internal counting semaphore (bounded even with no
    explicit maxsize: by SEM_VALUE_MAX, which on macOS/BSD can be as low as ~32k, far lower than
    Linux) that only advances as the other end calls get(). If that other end has already
    stopped draining (e.g. the learner exited after reaching total_env_steps, or crashed) and
    the queue's backlog hits that cap, a plain put() blocks inside the OS/semaphore call itself
    -- the calling actor's `while not stop_event.is_set(): ...` loop never gets back to the top
    to notice `stop_event` at all, and needs SIGTERM/SIGKILL to exit. Verified empirically while
    building this pipeline: an unthrottled producer reproduced exactly this hang on macOS in
    under a second. Returns False (caller should drop the item) only once stop_event has fired.
    """
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except _queue.Full:
            continue
    return False


def get_until_stop(q, stop_event, timeout: float = 1.0):
    """`q.get()`, but re-checks `stop_event` on a timeout instead of blocking forever if the
    other end (e.g. inference_server.py) has stopped responding -- see put_until_stop for why a
    plain blocking get() is unsafe here too. Returns None once stop_event has fired; callers
    must treat None as "shutting down", never as a valid payload."""
    while not stop_event.is_set():
        try:
            return q.get(timeout=timeout)
        except _queue.Empty:
            continue
    return None


def compute_epsilon(cfg: QMIXConfig, global_step: int, anchor_step: int) -> float:
    """Exact copy of QMIXTrainer.epsilon()'s formula (see that method's docstring for why the
    anchor offset exists), parameterized on the shared global step counter instead of
    self.env_step_count since action selection now happens on actor processes, not the learner."""
    frac = min(1.0, max(0.0, global_step - anchor_step) / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
