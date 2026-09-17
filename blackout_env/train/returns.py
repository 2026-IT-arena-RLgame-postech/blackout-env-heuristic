"""
n-step return and SPR-window helpers built on top of SequentialReplayBuffer's raw circular
arrays (reward/done). Kept separate from the buffer itself so this indexing logic — which is
fiddly (variable per-sample truncation at episode boundaries, modular wraparound) — can be
unit-tested against hand-built done-sequences in isolation from PER/sampling concerns.
"""

from __future__ import annotations

import numpy as np


def compute_n_step_return(
    reward_full: np.ndarray,
    done_full: np.ndarray,
    anchor_idx: np.ndarray,
    capacity: int,
    n_step: int,
    gamma: float,
    potential_full: np.ndarray | None = None,
    terminal_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-sample n-step return, truncated at the first `done` within the window (no bootstrap
    past an episode end).

    With potential_full (reward v2), potential-based shaping gamma*P(s') - P(s) is added for every
    step taken, which telescopes to gamma**steps * P(bootstrap) - P(anchor): shaped with the same
    gamma the target uses, so it stays policy-invariant whatever the annealing schedule is doing.
    The bootstrap potential is kept at an absorption `done` (the next row is the same match, and
    the potential stands in for the value the cut discards) and dropped only when the truncating
    row is `terminal` (a match end: the next row is another match).

    Returns
    -------
    n_step_return : [B]  sum_{k=0}^{steps-1} gamma^k * reward[anchor+k]
    bootstrap_idx  : [B]  anchor + steps (index to evaluate the bootstrap value at; meaningless
                     when not_done is 0, since it gets multiplied out)
    not_done       : [B]  1.0 if the window never hit `done` (bootstrap applies), else 0.0
    gamma_eff      : [B]  gamma ** steps (discount to apply to the bootstrap term)
    """
    B = anchor_idx.shape[0]
    returns = np.zeros(B, dtype=np.float32)
    discount = np.ones(B, dtype=np.float32)
    steps_taken = np.zeros(B, dtype=np.int64)
    active = np.ones(B, dtype=bool)
    ended_match = np.zeros(B, dtype=bool)

    for k in range(n_step):
        idx = (anchor_idx + k) % capacity
        r = reward_full[idx]
        d = done_full[idx]

        returns += np.where(active, discount * r, 0.0)
        steps_taken += active.astype(np.int64)
        discount = np.where(active, discount * gamma, discount)
        if terminal_full is not None:
            ended_match |= active & terminal_full[idx]
        active = active & (~d)

    bootstrap_idx = (anchor_idx + steps_taken) % capacity
    gamma_eff = gamma ** steps_taken.astype(np.float32)
    not_done = active.astype(np.float32)
    if potential_full is not None:
        next_potential = np.where(ended_match, 0.0, potential_full[bootstrap_idx])
        returns = returns + gamma_eff * next_potential - potential_full[anchor_idx % capacity]
    return returns.astype(np.float32), bootstrap_idx, not_done, gamma_eff


def compute_spr_valid_mask(
    done_full: np.ndarray,
    anchor_idx: np.ndarray,
    capacity: int,
    k_step: int,
) -> np.ndarray:
    """
    valid[b, k] (k = 0..k_step-1) is True iff predicting the frame at anchor+k+1 is a
    legitimate same-episode continuation — i.e. none of anchor..anchor+k are themselves a
    terminal (`done`) frame. The frame *at* a done step is still valid data; only frames
    *after* it belong to a different episode.

    Returns
    -------
    valid : [B, k_step] bool
    """
    B = anchor_idx.shape[0]
    valid = np.ones((B, k_step), dtype=bool)
    hit = done_full[anchor_idx % capacity].copy()
    for k in range(k_step):
        valid[:, k] = ~hit
        idx = (anchor_idx + k + 1) % capacity
        hit = hit | done_full[idx]
    return valid
