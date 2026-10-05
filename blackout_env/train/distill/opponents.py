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
  teacher plays alone), resolved by ``play_labelled_match``;
* ``switch:<seconds>:<A>|<B>`` -- A's moves before ``seconds`` of play (from team_state's time left), B's after;
  both are stepped every tick (P18b, docs §43: ``act:switch:6:<V29>|ckpt:<P17>`` = V29's race join, P17's play after);
* ``joinswitch:<t0>,<t1>,<seconds>:<A>|<B>`` -- A before ``seconds``; after it A only if the enemy had 4+ Hunters at
  once between ``t0`` and ``t1`` seconds (both teams' reserves joined the race, the (4,4) opening), otherwise B
  (P18b, docs §43: V29's play after a (4,4) opening beats P17's, P17's beats V29's after (4,3)/(3,3));
* ``perturb:<key=value,...>:<base>`` -- any other descriptor (not ``self``) executed with
  learned-policy errors: ``perturb:eps=0.2:strategic_v21``, ``perturb:sticky=4,lag=3:teacher:v21-teach``
  (keys eps, dither, sticky, lag, freeze; see perturb.py).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, HeuristicPolicyMixture, make_heuristic
from blackout_env.model.base import BaseModel
from blackout_env.train.distill.perturb import PREFIX as PERTURB_PREFIX
from blackout_env.train.distill.perturb import PerturbedPolicy, split_descriptor

CHECKPOINT_PREFIX = "ckpt:"
TEACHER_PREFIX = "teacher:"
SWITCH_PREFIX = "switch:"
JOINSWITCH_PREFIX = "joinswitch:"
MATCH_SECONDS = 420.0  # metrics.MATCH_SECONDS (team_state[2] = time left / MATCH_SECONDS)
SELF = "self"
RANDOM_TURN = 0.1  # per tick; a random walker keeps its heading ~10 ticks
_CHECKPOINT_CACHE: dict[str, object] = {}  # worker-side: path -> policy


def parse_pool(entries: list[str]) -> dict[str, float]:
    """["mixture=0.75", "checkpoints=0.25"] -> normalised weights (a bare name weighs 1)."""
    pool = {}
    for entry in entries:
        name, _, weight = entry.partition("=")
        if name not in ("mixture", "checkpoints", "extra") and name not in POLICY_REGISTRY:
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

    from blackout_env.train.distill.inference import load_net

    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    loadable, rejected = [], []
    for path in paths:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["policy_state"]
            load_net(state)
            loadable.append(path)
        except (KeyError, RuntimeError):
            rejected.append(path)
    return loadable, rejected


def draw_opponent(rng: np.random.Generator, pool: dict[str, float], checkpoints: list[str],
                  extra: list[str] = ()) -> str:
    """One opponent descriptor: ``checkpoints`` and ``extra`` (any descriptors, e.g. perturb: or
    parameterised heuristics, from --opponent-extra) draw uniformly from their lists."""
    names = list(pool)
    name = names[int(rng.choice(len(names), p=[pool[n] for n in names]))]
    if name == "checkpoints":
        if not checkpoints:
            raise ValueError("'checkpoints' opponents requested but no compatible checkpoint was found")
        return CHECKPOINT_PREFIX + checkpoints[int(rng.integers(len(checkpoints)))]
    if name == "extra":
        if not extra:
            raise ValueError("'extra' opponents requested but --opponent-extra is empty")
        return extra[int(rng.integers(len(extra)))]
    return name


def make_opponent(descriptor: str, seed: int):
    if descriptor.startswith(SWITCH_PREFIX):
        seconds, a, b = split_switch(descriptor)
        return SwitchPolicy(seconds, make_opponent(a, seed), make_opponent(b, seed + 1))
    if descriptor.startswith(JOINSWITCH_PREFIX):
        (t0, t1, seconds), a, b = split_joinswitch(descriptor)
        return JoinSwitchPolicy(t0, t1, seconds, make_opponent(a, seed), make_opponent(b, seed + 1))
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
    if descriptor.startswith(PERTURB_PREFIX):
        parameters, base = split_descriptor(descriptor)
        offset = parameters.pop("seed", 0)
        return PerturbedPolicy(make_opponent(base, seed), seed=seed + offset, **parameters)
    policy_id, _, parameters = descriptor.partition(":")
    return make_heuristic(policy_id, **_parse_parameters(parameters))


def check_opponent(descriptor: str) -> None:
    """Raise now, not in a worker, on a descriptor ``make_opponent`` cannot build."""
    if descriptor == SELF:
        return
    if descriptor.startswith((SWITCH_PREFIX, JOINSWITCH_PREFIX)):
        _, a, b = split_switch(descriptor) if descriptor.startswith(SWITCH_PREFIX) else split_joinswitch(descriptor)
        if SELF in (a, b):
            raise ValueError(f"{descriptor!r}: switch: halves cannot be 'self'")
        check_opponent(a)
        check_opponent(b)
        return
    if descriptor.startswith(PERTURB_PREFIX):
        parameters, base = split_descriptor(descriptor)
        if base.startswith(CHECKPOINT_PREFIX):  # the checkpoint itself is loaded in the worker
            check_opponent(base)
            PerturbedPolicy(Idle(), **parameters)
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


def split_switch(descriptor: str) -> tuple[float, str, str]:
    """``switch:<seconds>:<A>|<B>`` -> (seconds, A, B)."""
    seconds, sep, rest = descriptor[len(SWITCH_PREFIX):].partition(":")
    a, bar, b = rest.partition("|")
    if not sep or not bar or not a or not b:
        raise ValueError(f"{descriptor!r}: expected switch:<seconds>:<A>|<B>")
    return float(seconds), a, b


class SwitchPolicy(BaseModel):
    """A before ``seconds`` of play, B after; both act on every call so stateful policies keep their history."""

    def __init__(self, seconds: float, a, b):
        self.seconds, self.a, self.b = float(seconds), a, b

    def reset(self) -> None:
        for p in (self.a, self.b):
            if hasattr(p, "reset"):
                p.reset()

    def act(self, obs):
        if not obs:
            return {}
        moves_a, moves_b = self.a.act(obs), self.b.act(obs)
        elapsed = (1.0 - float(np.asarray(next(iter(obs.values()))["team_state"])[2])) * MATCH_SECONDS
        return moves_a if elapsed < self.seconds else moves_b


def split_joinswitch(descriptor: str) -> tuple[tuple[float, float, float], str, str]:
    """``joinswitch:<t0>,<t1>,<seconds>:<A>|<B>`` -> ((t0, t1, seconds), A, B)."""
    times, sep, rest = descriptor[len(JOINSWITCH_PREFIX):].partition(":")
    a, bar, b = rest.partition("|")
    t = tuple(float(x) for x in times.split(","))
    if not sep or not bar or not a or not b or len(t) != 3:
        raise ValueError(f"{descriptor!r}: expected joinswitch:<t0>,<t1>,<seconds>:<A>|<B>")
    return t, a, b


ENEMY_JOIN_HUNTERS = 4
HUNTER_COL = 10  # agent_states class one-hot (cols 9-11, COLLECTOR / HUNTER / CARRIER); col 2 = +1 own team


def enemy_hunters(obs) -> int:
    """Enemy Hunters in an observation dict (any own unit's agent_states)."""
    a = np.asarray(next(iter(obs.values()))["agent_states"])
    return int(((a[:, 2] < 0) & (a[:, HUNTER_COL] > 0.5)).sum())


class JoinSwitchPolicy(SwitchPolicy):
    """SwitchPolicy whose late half is A again when the enemy had ENEMY_JOIN_HUNTERS Hunters during [t0, t1] s."""

    def __init__(self, t0: float, t1: float, seconds: float, a, b):
        super().__init__(seconds, a, b)
        self.t0, self.t1 = float(t0), float(t1)
        self.enemy_max, self._last_elapsed = 0, 0.0

    def reset(self) -> None:
        super().reset()
        self.enemy_max, self._last_elapsed = 0, 0.0

    def act(self, obs):
        if not obs:
            return {}
        moves_a, moves_b = self.a.act(obs), self.b.act(obs)
        elapsed = (1.0 - float(np.asarray(next(iter(obs.values()))["team_state"])[2])) * MATCH_SECONDS
        if elapsed < self._last_elapsed:
            self.enemy_max = 0  # a new match without reset()
        self._last_elapsed = elapsed
        if self.t0 <= elapsed <= self.t1:
            self.enemy_max = max(self.enemy_max, enemy_hunters(obs))
        late_a = self.enemy_max >= ENEMY_JOIN_HUNTERS
        return moves_a if elapsed < self.seconds or late_a else moves_b


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

        from blackout_env.train.distill.inference import load_checkpoint_net, make_student, net_device

        net = load_checkpoint_net(path)  # a distill or a rollout-net (RolloutNet) checkpoint
        if len(_CHECKPOINT_CACHE) >= 16:  # ~24 MB each; a worker meets a few dozen at most
            _CHECKPOINT_CACHE.pop(next(iter(_CHECKPOINT_CACHE)))
        policy = _CHECKPOINT_CACHE[path] = make_student(net, device=net_device(path))  # MyPolicy, own rows only, traced on CPU
    return policy


def opponent_label(descriptor: str, opponent) -> str:
    """What the learner actually played: the mixture's chosen policy, or the checkpoint's run."""
    if descriptor.startswith(CHECKPOINT_PREFIX):
        path = Path(descriptor[len(CHECKPOINT_PREFIX):])
        return f"ckpt:{path.parent.name}/{path.stem}"
    if descriptor.startswith(PERTURB_PREFIX):
        base = split_descriptor(descriptor)[1]
        return descriptor[: len(descriptor) - len(base)] + opponent_label(base, opponent.policy)
    sample = getattr(opponent, "current_sample", None)
    return f"mixture:{sample.policy_id}" if sample is not None else descriptor


def opponent_name(descriptor: str) -> str:
    """A short, file-name-safe name for a descriptor (reports and shard files)."""
    if descriptor.startswith(CHECKPOINT_PREFIX):
        path = Path(descriptor[len(CHECKPOINT_PREFIX):])
        descriptor = f"ckpt:{path.parent.name.split('_')[0]}/{path.stem}"
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "-", descriptor)


def opponent_category(label: str) -> str:
    """checkpoint / perturbed / new_axis (the 2026-09-30 strategies) / top_tier (V17-V21) /
    heuristic -- how collection stats are split."""
    if label.startswith(CHECKPOINT_PREFIX):
        return "checkpoint"
    if label.startswith("perturb:"):
        return "perturbed"
    if re.search(r"strategic_(raider|cargo_hunter|early_rush|fortress|duelist)\b", label):
        return "new_axis"
    return "top_tier" if re.search(r"strategic_v(1[789]|2[01])\b", label) else "heuristic"
