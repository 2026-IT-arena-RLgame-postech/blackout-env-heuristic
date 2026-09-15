"""
Sequential, prioritized replay buffer for ONE self-play "stream" (one team's own perspective).

Transition-level, not episode-level: MyModel is a plain per-frame feedforward net (no
recurrent state to carry across steps), so unlike QMIX's usual RNN-agent setup there's no
need to pad/batch whole episodes.

"Sequential" matters here because both n-step returns and the SPR auxiliary loss need to look
a few steps *ahead* of a sampled transition. Rather than storing an explicit next_obs (2x
memory), this buffer stores one plain circular timeline per field and lets the caller read
forward from any anchor index via modular arithmetic (index+1 IS the next real transition,
as long as the window doesn't cross the write head — see `_forbidden_ranges` and
`read_window`). `done` marks the last transition of an episode; the slot right after it is
unrelated (the next episode's first frame), so window-consuming code (n-step return, SPR
rollout) must stop at the first `done` it encounters — that boundary is handled by the
*caller* (train/returns.py), since it doesn't depend on where the write head currently is.
`sample()`'s own job is only the write-head safety margin: temporarily zeroing out the
`window`-sized region behind the write head so anchors are drawn — with correctly normalized
priorities and importance-sampling weights — exclusively from indices where reading forward
`window` steps is memory-safe.

Every stored entry already carries BOTH teams' chosen actions (10 units, physical order) even
though a given stream only trains on its own team's 5 rows — the opponent's actions are kept
so the SPR transition model can condition on the true joint action (the map's future depends
on what every unit on the field did, not just "my" team).

Priorities follow Prioritized Experience Replay (Schaul et al., 2016): sampling probability
proportional to |TD-error|^alpha via a sum-tree, importance-sampling correction via a min-tree
(for the max-weight normalization) and a beta exponent the trainer anneals toward 1 over
training.
"""

from __future__ import annotations

import numpy as np

from .segment_tree import MinSegmentTree, SumSegmentTree

# Where a stored transition came from -- diagnostics only (never affects sampling/priorities),
# so TensorBoard can split batch/buffer stats by origin once on-policy data starts displacing
# the static heuristic dataset (see onpolicy_collect.py).
SOURCE_DATASET = 0
SOURCE_SELF_VS_HEURISTIC = 1
SOURCE_SELF_PLAY = 2
SOURCE_NAMES = ("dataset", "self_vs_heuristic", "self_play")


class SequentialReplayBuffer:
    def __init__(
        self,
        capacity: int,
        graphic_shape: tuple[int, int, int],
        team_state_size: int,
        agent_state_size: int,
        n_units: int = 10,
        per_alpha: float = 0.6,
        per_eps: float = 1e-3,
    ) -> None:
        pow2 = 1
        while pow2 < capacity:
            pow2 *= 2
        self.capacity = pow2
        self._size = 0
        self._pos = 0
        self._per_alpha = per_alpha
        self._per_eps = per_eps
        self._max_priority = 1.0

        self.graphic = np.zeros((self.capacity, *graphic_shape), dtype=np.float32)
        self.team_state = np.zeros((self.capacity, team_state_size), dtype=np.float32)
        self.agent_states = np.zeros((self.capacity, n_units, agent_state_size), dtype=np.float32)
        self.actions = np.zeros((self.capacity, n_units), dtype=np.int64)  # all n_units, physical order
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.bool_)
        self.source = np.zeros((self.capacity,), dtype=np.int8)
        self.source_counts = np.zeros(len(SOURCE_NAMES), dtype=np.int64)

        self._sum_tree = SumSegmentTree(self.capacity)
        self._min_tree = MinSegmentTree(self.capacity)

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def push(
        self,
        graphic: np.ndarray,
        team_state: np.ndarray,
        agent_states: np.ndarray,
        actions: np.ndarray,
        reward: float,
        done: bool,
        source: int = SOURCE_DATASET,
    ) -> None:
        i = self._pos
        if self._size == self.capacity:
            self.source_counts[self.source[i]] -= 1
        self.graphic[i] = graphic
        self.team_state[i] = team_state
        self.agent_states[i] = agent_states
        self.actions[i] = actions
        self.reward[i] = reward
        self.done[i] = done
        self.source[i] = source
        self.source_counts[source] += 1

        priority = self._max_priority ** self._per_alpha
        self._sum_tree[i] = priority
        self._min_tree[i] = priority

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _forbidden_ranges(self, window: int) -> list[tuple[int, int]]:
        """
        Non-wrapping (start, end) sub-ranges covering exactly the indices too close to the
        write head to safely read `window` steps forward from: when the buffer hasn't wrapped
        yet, that's simply the last `window` written slots; once wrapped, it's the `window`
        slots immediately preceding `pos` (which may itself wrap across raw index 0) — those
        hold the most-recently-overwritten data, whose "future" (past the write head) belongs
        to a different, newer trajectory than the one at that index.
        """
        if window <= 0:
            return []
        if self._size < self.capacity:
            start = max(0, self._size - window)
            return [(start, self._size)] if start < self._size else []
        start = (self._pos - window) % self.capacity
        end = self._pos
        if start < end:
            return [(start, end)]
        return [(start, self.capacity), (0, end)]

    def read_window(self, anchor_idx: np.ndarray, offsets: np.ndarray, field: str) -> np.ndarray:
        """anchor_idx: [B]. offsets: [W]. Returns [B, W, ...] = field[(anchor+offsets) % capacity]."""
        idx = (anchor_idx[:, None] + offsets[None, :]) % self.capacity
        return getattr(self, field)[idx]

    def sample(self, batch_size: int, window: int, beta: float) -> dict[str, np.ndarray]:
        """
        window: how many steps forward from each anchor must be safely readable (the caller
        passes max(n_step, spr_k) and is responsible for truncating consumed windows at the
        first `done` itself — this only guarantees the *memory* is safe to read).

        Anchors too close to the write head (see `_forbidden_ranges`) are excluded from
        sampling by temporarily zeroing their priority in both trees for the duration of this
        call — not by rejection-sampling and discarding draws that land there. Rejecting after
        the fact would leave `total`/`p_min` computed over the *unconditional* distribution
        (valid + invalid anchors) while the realized draws come from the valid-conditioned
        one, which both skews every importance-sampling weight by the same hard-to-predict
        factor (how much priority mass currently sits in the forbidden zone — freshly-pushed
        transitions start at max priority, so that fraction is often large, not a rare edge
        case) and, if the forbidden zone happens to hold most of the mass, can make the
        rejection loop churn for a very long time. Zeroing first makes every draw valid by
        construction and keeps `total`/`p_min` consistent with what's actually sampled.

        Returns per-anchor fields (graphic/team_state/agent_states/actions/reward/done at the
        anchor itself) plus 'indices' (for update_priorities) and 'is_weights'. Anything the
        caller needs beyond the anchor (n-step rewards, SPR future frames) should be read via
        read_window(indices, offsets, field).
        """
        assert self._size > 0, "cannot sample from an empty buffer"

        forbidden_ranges = self._forbidden_ranges(window)
        saved: dict[int, float] = {}
        for start, end in forbidden_ranges:
            for i in range(start, end):
                saved[i] = self._sum_tree[i]
                self._sum_tree[i] = 0.0
                self._min_tree[i] = float("inf")

        try:
            total = self._sum_tree.sum(0, self._size)
            if total <= 0:
                raise RuntimeError(
                    "SequentialReplayBuffer.sample: no valid anchors for this window "
                    f"(window={window}, size={self._size}, capacity={self.capacity}). "
                    "The buffer is too small relative to the requested n-step/SPR window."
                )

            indices = np.empty(batch_size, dtype=np.int64)
            for n in range(batch_size):
                mass = np.random.uniform(0, total)
                idx = self._sum_tree.find_prefixsum_idx(mass)
                indices[n] = min(idx, self._size - 1)

            p_min = self._min_tree.min(0, self._size) / total
            max_weight = (p_min * self._size) ** (-beta)
            probs = np.array([self._sum_tree[int(i)] for i in indices]) / total
            is_weights = (probs * self._size) ** (-beta) / max_weight
        finally:
            for i, v in saved.items():
                self._sum_tree[i] = v
                self._min_tree[i] = v

        return {
            "indices": indices,
            "is_weights": is_weights.astype(np.float32),
            "graphic": self.graphic[indices],
            "team_state": self.team_state[indices],
            "agent_states": self.agent_states[indices],
            "actions": self.actions[indices],
            "reward": self.reward[indices],
            "done": self.done[indices],
            "source": self.source[indices],
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self._per_eps
        for i, p in zip(indices, priorities):
            pa = float(p) ** self._per_alpha
            self._sum_tree[int(i)] = pa
            self._min_tree[int(i)] = pa
        self._max_priority = max(self._max_priority, float(priorities.max()))
