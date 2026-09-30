"""Any policy as a distillation teacher: label each tick with the move it actually makes.

``act:<descriptor>`` (``act:strategic_v19``, ``act:strategic_v17:steal_weight=3.0``, any
``opponents.make_opponent`` descriptor except ``self``) wraps the policy that descriptor builds.
Unlike the relabelling teachers (heuristics/teach.py), whose labels are a function of the
observation alone, most heuristics keep history (A* path caches, target hysteresis, escape
timers, role memory): 18-32% of V4/V17/V19 moves differ from a fresh instance's on the same
observation.  So an act-teacher is not a relabeller -- it is the policy itself, stepped exactly
once per learner decision tick with the learner's observation (partial windows included), so its
internal state follows the trajectory that is actually played, whoever drove.

* ``teach(obs)`` steps the policy (``policy.act(obs)``, once) and returns per unit a
  ``TeachLabel`` whose mask is the single compass direction nearest its move
  (``direction_vector_to_idx``, after the env's [-1, 1] clip), kind ``"act"``; a move shorter than
  ``ZERO_MOVE`` (Unity's ``Vector2.normalized`` returns zero below 1e-5: the unit stands) gives an
  invalid label, which ``--stand-action`` trains as STAND.  The raw moves of that call are kept in
  ``last_actions``.
* On teacher-driven ticks ``play_labelled_match`` executes ``last_actions`` -- the policy's raw
  continuous move, not the snapped label -- so a beta = 1 match plays exactly the wrapped policy
  (the label is its 8-way projection; Unity normalises the input, so only the angle differs,
  by at most 22.5 degrees).
* ``act(obs)`` is the policy's own ``act`` (as an opponent, e.g. ``teacher:act:strategic_v19``).
  Never call both ``teach`` and ``act`` on one tick: each steps the policy's state.
* ``reset()`` resets the wrapped policy (a new match).
"""

from __future__ import annotations

import numpy as np

from blackout_env.heuristics.teach import TeachLabel
from blackout_env.model.base import BaseModel
from blackout_env.model.my_policy import direction_vector_to_idx
from blackout_env.train.distill.opponents import SELF, check_opponent, make_opponent

ACT_PREFIX = "act:"
ACT_KIND = "act"
ZERO_MOVE = 1e-5  # Unity's Vector2.normalized: shorter inputs become zero (the unit stands)


def act_label(move: np.ndarray) -> TeachLabel:
    """The single-direction label of one continuous move; invalid when it stands."""
    move = np.clip(np.asarray(move, np.float32), -1.0, 1.0)  # what the env executes
    mask = np.zeros(8, dtype=bool)
    if float(np.linalg.norm(move)) < ZERO_MOVE:
        return TeachLabel(mask, False, ACT_KIND, None)
    mask[int(direction_vector_to_idx(move))] = True
    return TeachLabel(mask, True, ACT_KIND, None)


class ActTeacher(BaseModel):
    """``descriptor`` (an opponents.py descriptor) as a teacher labelling its own moves."""

    def __init__(self, descriptor: str, seed: int = 0):
        if descriptor == SELF:
            raise ValueError("'act:self' is not a teacher")
        self.descriptor = descriptor
        self.policy = make_opponent(descriptor, seed)
        self.last_actions: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.last_actions = {}

    def teach(self, obs) -> dict[str, TeachLabel]:
        actions = self.policy.act(obs) if obs else {}
        zero = np.zeros(2, dtype=np.float32)
        self.last_actions = {name: np.asarray(actions.get(name, zero), np.float32) for name in obs}
        return {name: act_label(move) for name, move in self.last_actions.items()}

    def act(self, obs):
        return self.policy.act(obs)


def is_act_teacher(name: str) -> bool:
    return isinstance(name, str) and name.startswith(ACT_PREFIX)


def check_act_teacher(name: str) -> None:
    """Raise now, not in a worker, on an ``act:`` descriptor that cannot be built."""
    descriptor = name[len(ACT_PREFIX):]
    if not descriptor or descriptor == SELF:
        raise ValueError(f"{name!r}: act: needs a policy descriptor other than 'self'")
    check_opponent(descriptor)
