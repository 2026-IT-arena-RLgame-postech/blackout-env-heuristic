"""Who the learner plays in collection: a weighted pool of opponent kinds, one draw per match.

Kinds (``--opponents kind=weight ...``):

* ``mixture`` -- HeuristicPolicyMixture: one of the 20 heuristics per match with its parameter
  cloud redrawn every absorption, here including V17-V19's own strategy knobs
  (``perturb_top_tier``), which the QMIX collections leave at their defaults;
* ``checkpoints`` -- a past MyModel checkpoint (the offline QMIX runs), uniformly over
  ``--opponent-checkpoints``: learned play no heuristic produces, including its mistakes;
* any heuristic id (``strategic_v17`` ...) -- that policy, unperturbed.

A draw is a descriptor string carried by MatchTask.opponent ("mixture", "strategic_v17",
"ckpt:<path>") and built inside the worker by ``make_opponent``.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from blackout_env.heuristics import POLICY_REGISTRY, HeuristicPolicyMixture, make_heuristic

CHECKPOINT_PREFIX = "ckpt:"
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
    return make_heuristic(descriptor)


def _checkpoint_policy(path: str):
    policy = _CHECKPOINT_CACHE.get(path)
    if policy is None:
        import torch

        from blackout_env.model.my_model import MyModel
        from blackout_env.model.my_policy import MyPolicy

        state = torch.load(path, map_location="cpu", weights_only=True)["policy_state"]
        net = MyModel(hidden_size=state["token_type_emb.weight"].shape[1])
        net.load_state_dict(state)
        if len(_CHECKPOINT_CACHE) >= 16:  # ~24 MB each; a worker meets a few dozen at most
            _CHECKPOINT_CACHE.pop(next(iter(_CHECKPOINT_CACHE)))
        policy = _CHECKPOINT_CACHE[path] = MyPolicy(net.eval())
    return policy


def opponent_label(descriptor: str, opponent) -> str:
    """What the learner actually played: the mixture's chosen policy, or the checkpoint's run."""
    if descriptor.startswith(CHECKPOINT_PREFIX):
        path = Path(descriptor[len(CHECKPOINT_PREFIX):])
        return f"ckpt:{path.parent.name}/{path.stem}"
    sample = getattr(opponent, "current_sample", None)
    return f"mixture:{sample.policy_id}" if sample is not None else descriptor


def opponent_category(label: str) -> str:
    """checkpoint / top_tier (V17-V19) / heuristic -- how collection stats are split."""
    if label.startswith(CHECKPOINT_PREFIX):
        return "checkpoint"
    return "top_tier" if re.search(r"strategic_v1[789]$", label) else "heuristic"
