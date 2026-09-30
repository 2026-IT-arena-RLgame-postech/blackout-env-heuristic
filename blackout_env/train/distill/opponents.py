"""Who the learner plays in collection: a weighted pool of opponent kinds, one draw per match.

Kinds (``--opponents kind=weight ...``):

* ``mixture`` -- HeuristicPolicyMixture: one of the 20 heuristics per match with its parameter
  cloud redrawn every absorption, here including V17-V19's own strategy knobs
  (``perturb_top_tier``), which the QMIX collections leave at their defaults;
* ``checkpoints`` -- a past MyModel checkpoint (the offline QMIX runs), uniformly over
  ``--opponent-checkpoints``: learned play no heuristic produces, including its mistakes;
* any heuristic id (``strategic_v17`` ...) -- that policy, unperturbed.

A draw is a descriptor string carried by MatchTask.opponent and built inside the worker by
``make_opponent``.  Evaluation (examples/distill_heldout_eval.py) also takes the out-of-
distribution descriptors no collection draws:

* ``strategic_v19:hunter_quota=4,hunter_mode=camp`` -- a heuristic with fixed constructor
  arguments (ints, floats, true/false, otherwise strings);
* ``random`` -- every unit walks a random compass direction, redrawn with probability
  ``RANDOM_TURN`` a tick; ``idle`` -- every unit stands still;
* ``teacher:v21-teach`` -- a teacher following its own labels;
* ``self`` -- the learner's own policy on the other side (the student, or the teacher when the
  teacher plays alone), resolved by ``play_labelled_match``.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, HeuristicPolicyMixture, make_heuristic
from blackout_env.model.base import BaseModel

CHECKPOINT_PREFIX = "ckpt:"
TEACHER_PREFIX = "teacher:"
SELF = "self"
RANDOM_TURN = 0.1  # per tick; a random walker keeps its heading ~10 ticks
_CHECKPOINT_CACHE: dict[str, object] = {}  # worker-side: path -> policy


def parse_pool(entries: list[str]) -> dict[str, float]:
    """["mixture=0.75", "checkpoints=0.25"] -> normalised weights (a bare name weighs 1)."""
    pool = {}
    for entry in entries:
        name, _, weight = entry.partition("=")
        if name not in ("mixture", "checkpoints") and name not in POLICY_REGISTRY:
            raise ValueError(f"unknown opponent kind {name!r}")
        pool[name] = float(weight) if weight else 1.0
    total = sum(pool.values())
    if total <= 0:
        raise ValueError("opponent weights must sum to > 0")
    return {name: weight / total for name, weight in pool.items()}


def compatible_checkpoints(patterns: list[str]) -> tuple[list[str], list[str]]:
    """(loadable, rejected) checkpoint paths: those whose policy_state loads into the current
    MyModel at their own width (QMIX runs use hidden_size 128; runs before the 2026-09-17
    observation changes no longer fit)."""
    import torch

    from blackout_env.model.my_model import MyModel

    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    loadable, rejected = [], []
    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["policy_state"]
            MyModel(hidden_size=state["token_type_emb.weight"].shape[1]).load_state_dict(state)
            loadable.append(path)
        except (KeyError, RuntimeError):
            rejected.append(path)
    return loadable, rejected


def draw_opponent(rng: np.random.Generator, pool: dict[str, float], checkpoints: list[str]) -> str:
    names = list(pool)
    name = names[int(rng.choice(len(names), p=[pool[n] for n in names]))]
    if name == "checkpoints":
        if not checkpoints:
            raise ValueError("'checkpoints' opponents requested but no compatible checkpoint was found")
        return CHECKPOINT_PREFIX + checkpoints[int(rng.integers(len(checkpoints)))]
    return name


def make_opponent(descriptor: str, seed: int):
    if descriptor == "mixture":
        return HeuristicPolicyMixture(seed=seed, perturb_top_tier=True)
    if descriptor.startswith(CHECKPOINT_PREFIX):
        return _checkpoint_policy(descriptor[len(CHECKPOINT_PREFIX):])
    if descriptor.startswith(TEACHER_PREFIX):
        from blackout_env.train.distill.collect import TEACHERS  # collect imports this module

        return TEACHERS[descriptor[len(TEACHER_PREFIX):]]()
    if descriptor == "random":
        return RandomWalker(seed)
    if descriptor == "idle":
        return Idle()
    if descriptor == SELF:
        raise ValueError("'self' is the learner's own policy: play_labelled_match resolves it")
    policy_id, _, parameters = descriptor.partition(":")
    return make_heuristic(policy_id, **_parse_parameters(parameters))


def check_opponent(descriptor: str) -> None:
    """Raise now, not in a worker, on a descriptor ``make_opponent`` cannot build."""
    if descriptor == SELF:
        return
    if descriptor.startswith(CHECKPOINT_PREFIX):
        if not Path(descriptor[len(CHECKPOINT_PREFIX):]).is_file():
            raise ValueError(f"no checkpoint at {descriptor[len(CHECKPOINT_PREFIX):]!r}")
        return
    make_opponent(descriptor, 0)


def _parse_parameters(text: str) -> dict:
    parameters = {}
    for item in filter(None, text.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"heuristic parameter {item!r} is not key=value")
        parameters[key] = _parse_value(value)
    return parameters


def _parse_value(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


class Idle(BaseModel):
    """Every unit stands still."""

    def act(self, obs):
        return {name: np.zeros(2, dtype=np.float32) for name in obs}


class RandomWalker(BaseModel):
    """Every unit walks a random compass direction, turning with probability ``turn`` a tick."""

    def __init__(self, seed: int, turn: float = RANDOM_TURN):
        self.seed, self.turn = seed, turn
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._heading: dict[str, np.ndarray] = {}

    def act(self, obs):
        for name in obs:
            if name not in self._heading or self._rng.random() < self.turn:
                angle = self._rng.integers(8) * np.pi / 4
                self._heading[name] = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return {name: self._heading[name].copy() for name in obs}


def _checkpoint_policy(path: str):
    policy = _CHECKPOINT_CACHE.get(path)
    if policy is None:
        import torch

        from blackout_env.model.my_model import MyModel
        from blackout_env.train.distill.inference import make_student

        state = torch.load(path, map_location="cpu", weights_only=True)["policy_state"]
        net = MyModel(hidden_size=state["token_type_emb.weight"].shape[1])
        net.load_state_dict(state)
        if len(_CHECKPOINT_CACHE) >= 16:  # ~24 MB each; a worker meets a few dozen at most
            _CHECKPOINT_CACHE.pop(next(iter(_CHECKPOINT_CACHE)))
        policy = _CHECKPOINT_CACHE[path] = make_student(net)  # MyPolicy, own rows only, traced
    return policy


def opponent_label(descriptor: str, opponent) -> str:
    """What the learner actually played: the mixture's chosen policy, or the checkpoint's run."""
    if descriptor.startswith(CHECKPOINT_PREFIX):
        path = Path(descriptor[len(CHECKPOINT_PREFIX):])
        return f"ckpt:{path.parent.name}/{path.stem}"
    sample = getattr(opponent, "current_sample", None)
    return f"mixture:{sample.policy_id}" if sample is not None else descriptor


def opponent_name(descriptor: str) -> str:
    """A short, file-name-safe name for a descriptor (reports and shard files)."""
    if descriptor.startswith(CHECKPOINT_PREFIX):
        path = Path(descriptor[len(CHECKPOINT_PREFIX):])
        descriptor = f"ckpt:{path.parent.name.split('_')[0]}/{path.stem}"
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "-", descriptor)


def opponent_category(label: str) -> str:
    """checkpoint / top_tier (V17-V19) / heuristic -- how collection stats are split."""
    if label.startswith(CHECKPOINT_PREFIX):
        return "checkpoint"
    return "top_tier" if re.search(r"strategic_v1[789]$", label) else "heuristic"
