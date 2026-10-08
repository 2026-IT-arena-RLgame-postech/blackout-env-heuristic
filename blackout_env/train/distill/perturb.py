"""Any opponent, executed the way learned policies misbehave: noise, dithering, lag, stalls.

A distilled student meets heuristics that always execute their plan cleanly, and past checkpoints
whose mistakes are whatever those runs happened to learn.  Neither covers the execution errors a
neural opponent makes systematically, so the student has never seen a *strong* plan played
sloppily: units that jitter, reverse near a goal, act late, commit to a direction too long or
simply stop.  ``PerturbedPolicy`` wraps any opponent and injects exactly those errors, each at a
set rate, so the strategy stays the base policy's and only the execution changes.

Descriptor (opponents.py): ``perturb:<key=value,...>:<base descriptor>``, where the base is any
other descriptor -- ``perturb:eps=0.2:strategic_v21``,
``perturb:sticky=4,lag=3:teacher:v21-teach``, ``perturb:dither=0.1,freeze=0.1:ckpt:<path>``.
Keys (all off by default; any combination):

* ``eps=p`` -- each unit, each tick, with probability ``p``: a uniformly random compass
  direction instead of the policy's action (sampling noise of a stochastic policy);
* ``dither=f`` -- a fraction ``f`` of unit-ticks is spent in reversal bursts of ``dither_ticks``
  (12) ticks that alternate the policy's action and its reverse every ``dither_period`` (3)
  ticks (so half of a burst, ``f / 2`` of all unit-ticks, is reversed): the back-and-forth of a
  policy sitting on a decision boundary.  The wrapper cannot see goals, so bursts start at
  random times;
* ``sticky=k`` -- each unit takes a fresh action only every ``k`` ticks (each unit on its own
  random phase) and repeats it in between: a policy that commits too long;
* ``lag=k`` -- the policy acts on the team observation from ``k`` ticks ago: a policy that
  reacts late (a unit missing from the lagged observation repeats its last action);
* ``freeze=f`` -- a fraction ``f`` of unit-ticks is frozen (zero action) in stretches of
  ``freeze_ticks`` (50, ~2 s): units that stall for no reason;
* ``seed=s`` -- added to the match seed (the wrapper is seeded by it; ``reset()``, called at
  every match start, restarts the same random stream).

Bursts and stretches start with the per-tick probability ``f / (L * (1 - f))`` while a unit is
not in one, which makes their long-run share of unit-ticks ``f``.  The stages apply in the
order listed below (lag -> policy -> sticky -> dither -> eps -> freeze), so a frozen unit stands
still whatever the others did.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from blackout_env.model.base import BaseModel

PREFIX = "perturb:"
_COMPASS = np.array([[np.cos(i * np.pi / 4), np.sin(i * np.pi / 4)] for i in range(8)], dtype=np.float32)
_INT_KEYS = ("sticky", "lag", "dither_ticks", "dither_period", "freeze_ticks", "seed")
_FLOAT_KEYS = ("eps", "dither", "freeze")


def split_descriptor(descriptor: str) -> tuple[dict, str]:
    """``perturb:eps=0.2,lag=3:strategic_v21`` -> ({"eps": 0.2, "lag": 3}, "strategic_v21")."""
    if not descriptor.startswith(PREFIX):
        raise ValueError(f"{descriptor!r} is not a {PREFIX!r} descriptor")
    parameters, sep, base = descriptor[len(PREFIX):].partition(":")
    if not sep or not base:
        raise ValueError(f"{descriptor!r}: expected perturb:<key=value,...>:<base descriptor>")
    parsed = {}
    for item in filter(None, parameters.split(",")):
        key, sep, value = item.partition("=")
        if not sep or key not in _INT_KEYS + _FLOAT_KEYS:
            raise ValueError(f"{descriptor!r}: bad perturbation {item!r} (keys: {', '.join(_FLOAT_KEYS + _INT_KEYS)})")
        parsed[key] = int(value) if key in _INT_KEYS else float(value)
    if not parsed:
        raise ValueError(f"{descriptor!r}: no perturbation given")
    return parsed, base


class _Bursts:
    """Per-unit on/off stretches of fixed length whose long-run share of ticks is ``fraction``."""

    def __init__(self, fraction: float, length: int):
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"a burst fraction must be in [0, 1), got {fraction}")
        self.length = max(1, int(length))
        self.start_p = fraction / (self.length * (1.0 - fraction)) if fraction > 0 else 0.0
        self.left: dict[str, int] = {}

    def step(self, name: str, rng: np.random.Generator) -> int:
        """Ticks already spent in the current stretch (0 = its first tick), or -1 when off."""
        left = self.left.get(name, 0)
        if left == 0:
            if self.start_p == 0.0 or rng.random() >= self.start_p:
                return -1
            left = self.length
        self.left[name] = left - 1
        return self.length - left


class PerturbedPolicy(BaseModel):
    """``policy`` with learned-policy execution errors; see the module docstring."""

    def __init__(
        self,
        policy,
        *,
        seed: int = 0,
        eps: float = 0.0,
        dither: float = 0.0,
        dither_ticks: int = 12,
        dither_period: int = 3,
        sticky: int = 1,
        lag: int = 0,
        freeze: float = 0.0,
        freeze_ticks: int = 50,
    ):
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        if sticky < 1 or lag < 0 or dither_period < 1:
            raise ValueError("sticky and dither_period must be >= 1, lag >= 0")
        self.policy = policy
        self.seed = int(seed)
        self.eps, self.sticky, self.lag = float(eps), int(sticky), int(lag)
        self.dither, self.dither_ticks, self.dither_period = float(dither), int(dither_ticks), int(dither_period)
        self.freeze, self.freeze_ticks = float(freeze), int(freeze_ticks)
        _Bursts(self.dither, self.dither_ticks), _Bursts(self.freeze, self.freeze_ticks)  # validate now
        self.reset()

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._rng = np.random.default_rng(self.seed)
        self._history: deque = deque(maxlen=self.lag + 1)
        self._last: dict[str, np.ndarray] = {}  # the policy's latest action per unit (lag fallback)
        self._held: dict[str, np.ndarray] = {}  # sticky: the action being repeated
        self._phase: dict[str, int] = {}
        self._tick = 0
        self._dithering = _Bursts(self.dither, self.dither_ticks)
        self._frozen = _Bursts(self.freeze, self.freeze_ticks)

    def act(self, obs):
        if not obs:
            return {}
        self._history.append(obs)
        proposed = self.policy.act(self._history[0])
        self._last.update({name: np.asarray(a, dtype=np.float32) for name, a in proposed.items()})
        actions = {}
        for name in sorted(obs):
            action = self._last.get(name, np.zeros(2, dtype=np.float32))
            if self.sticky > 1:
                phase = self._phase.setdefault(name, int(self._rng.integers(self.sticky)))
                if name not in self._held or (self._tick + phase) % self.sticky == 0:
                    self._held[name] = action
                action = self._held[name]
            if self.dither > 0:
                into = self._dithering.step(name, self._rng)
                if into >= 0 and (into // self.dither_period) % 2 == 1:
                    action = -action
            if self.eps > 0 and self._rng.random() < self.eps:
                action = _COMPASS[int(self._rng.integers(8))]
            if self.freeze > 0 and self._frozen.step(name, self._rng) >= 0:
                action = np.zeros(2, dtype=np.float32)
            actions[name] = np.array(action, dtype=np.float32)
        self._tick += 1
        return actions
