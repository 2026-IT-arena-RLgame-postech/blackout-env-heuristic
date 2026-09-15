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
proportional to |TD-error|^alpha, importance-sampling correction with a beta exponent the trainer
anneals toward 1 over training, weights normalized by the batch maximum (as Dopamine/BBF do).
`sample(normalize=False)` returns the unnormalized weights instead, so a caller drawing one batch
from several buffers can normalize across all of them together.
"""

from __future__ import annotations

import numpy as np

from .segment_tree import SumSegmentTree

# Where a stored transition came from. offline_pretrain.py keeps each on-policy source in its own
# buffer (see QMIXConfig.batch_source_fracs); a buffer that mixes sources (the online trainer's)
# uses the tag only to split TensorBoard stats by origin.
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
        # Whether this stream's own-team actions came from a heuristic (behavior-cloning target)
        # rather than from the net being trained.
        self.demo = np.zeros((self.capacity,), dtype=np.bool_)
        self.source_counts = np.zeros(len(SOURCE_NAMES), dtype=np.int64)

        self._sum_tree = SumSegmentTree(self.capacity)

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
        demo: bool = True,
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
        self.demo[i] = demo
        self.source_counts[source] += 1

        self._sum_tree[i] = self._max_priority ** self._per_alpha

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

    def sample(self, batch_size: int, window: int, beta: float, normalize: bool = True) -> dict[str, np.ndarray]:
        """
        window: how many steps forward from each anchor must be safely readable (the caller
        passes max(n_step, spr_k) and is responsible for truncating consumed windows at the
        first `done` itself — this only guarantees the *memory* is safe to read).

        normalize: divide the importance weights by this batch's maximum. Pass False when
        combining draws from several buffers, and normalize over the combined batch instead.

        Anchors too close to the write head (see `_forbidden_ranges`) are excluded from
        sampling by temporarily zeroing their priority for the duration of this call — not by
        rejection-sampling and discarding draws that land there. Rejecting after the fact would
        leave the sampling total computed over the *unconditional* distribution (valid + invalid
        anchors) while the realized draws come from the valid-conditioned one, which skews every
        importance-sampling weight (freshly-pushed transitions start at max priority, so the
        forbidden zone often holds a large share of the mass) and can make a rejection loop churn
        for a very long time. Zeroing first makes every draw valid by construction and keeps the
        total consistent with what's actually sampled.

        Returns per-anchor fields (graphic/team_state/agent_states/actions/reward/done/source/
        demo at the anchor itself) plus 'indices' (for update_priorities) and 'is_weights'.
        Anything the caller needs beyond the anchor (n-step rewards, SPR future frames) should be
        read via read_window(indices, offsets, field).
        """
        assert self._size > 0, "cannot sample from an empty buffer"

        forbidden = [i for start, end in self._forbidden_ranges(window) for i in range(start, end)]
        saved = [(i, self._sum_tree[i]) for i in forbidden]
        for i in forbidden:
            self._sum_tree[i] = 0.0

        try:
            total = self._sum_tree.sum(0, self._size)
            if total <= 0:
                raise RuntimeError(
                    "SequentialReplayBuffer.sample: no valid anchors for this window "
                    f"(window={window}, size={self._size}, capacity={self.capacity}). "
                    "The buffer is too small relative to the requested n-step/SPR window."
                )

            indices = np.empty(batch_size, dtype=np.int64)
            probs = np.empty(batch_size, dtype=np.float64)
            for n in range(batch_size):
                idx = min(self._sum_tree.find_prefixsum_idx(np.random.uniform(0, total)), self._size - 1)
                while self._sum_tree[idx] <= 0.0:
                    # Float rounding in the tree sums can walk a draw at the very top of the mass
                    # onto an empty or forbidden leaf -- redraw.
                    idx = min(self._sum_tree.find_prefixsum_idx(np.random.uniform(0, total)), self._size - 1)
                indices[n] = idx
                probs[n] = self._sum_tree[idx] / total
        finally:
            for i, v in saved:
                self._sum_tree[i] = v

        is_weights = (probs * (self._size - len(forbidden))) ** (-beta)
        if normalize:
            is_weights = is_weights / is_weights.max()

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
            "demo": self.demo[indices],
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self._per_eps
        for i, p in zip(indices, priorities):
            self._sum_tree[int(i)] = float(p) ** self._per_alpha
        self._max_priority = max(self._max_priority, float(priorities.max()))
