"""Distillation data: one compressed .npz shard per match, aggregated in a fixed-size ring.

Every row is one tick of the learner's team, already in the canonical team frame
(env/team_frame.py) so the trainer feeds it to the net unchanged:

  graphic       uint8 [T, H, W, C]  the observation times 15 (lossless: every channel is 0/1
                                    except batteries, whose amount/15 has 15ths only)
  team_state    f32   [T, 4]
  agent_states  f32   [T, 10, D]    own units in rows 0-4
  labels        uint8 [T, 5]        bit i = compass direction i is correct for own unit k
                                    (canonical frame; 0 = no label)
  kinds         uint8 [T, 5]        index into KINDS: what the label is doing

Ticks where the teacher labels no unit are not stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GRAPHIC_SCALE = 15.0
N_TEAM = 5
KINDS = (
    "", "battery", "steal", "special", "deposit", "wait_storage", "transform",
    "hunt", "patrol_defend", "patrol_raid", "idle",
)
KIND_INDEX = {kind: i for i, kind in enumerate(KINDS)}
FIELDS = ("graphic", "team_state", "agent_states", "labels", "kinds")


def encode_graphic(graphic: np.ndarray) -> np.ndarray:
    encoded = np.rint(graphic * GRAPHIC_SCALE).astype(np.uint8)
    if not np.array_equal(encoded / np.float32(GRAPHIC_SCALE), graphic.astype(np.float32)):
        raise ValueError("graphic is not in 15ths; the uint8 shard format would lose information")
    return encoded


def decode_graphic(encoded: np.ndarray) -> np.ndarray:
    return encoded.astype(np.float32) / np.float32(GRAPHIC_SCALE)


def unpack_labels(bits: np.ndarray) -> np.ndarray:
    """[..., 5] uint8 bit sets -> [..., 5, 8] bool."""
    return np.unpackbits(bits[..., None], axis=-1, bitorder="little").astype(bool)


def save_shard(path: Path, rows: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{field: rows[field] for field in FIELDS})  # maps are mostly zeros
    tmp.rename(path)  # readers only ever see complete shards


class DistillDataset:
    """The most recent ``capacity`` rows of every shard under ``root``, sampled uniformly.

    ``refresh`` appends shards that appeared since the last call, oldest first; once full, new
    rows overwrite the oldest (DAgger aggregates, but a laptop cannot hold every row ever played)."""

    def __init__(self, root: Path, capacity: int):
        self.root = Path(root)
        self.capacity = int(capacity)
        self._arrays: dict[str, np.ndarray] | None = None
        self._loaded: set[str] = set()
        self._next = 0
        self.size = 0

    def refresh(self) -> int:
        """Load new shards; returns the number of rows added."""
        added = 0
        for path in sorted(self.root.glob("*.npz")):
            if path.name in self._loaded or path.name.endswith(".tmp.npz"):
                continue
            with np.load(path) as shard:
                added += self._append({field: shard[field] for field in FIELDS})
            self._loaded.add(path.name)
        return added

    def _append(self, rows: dict[str, np.ndarray]) -> int:
        n = len(rows["labels"])
        if self._arrays is None:
            self._arrays = {
                field: np.zeros((self.capacity, *rows[field].shape[1:]), dtype=rows[field].dtype)
                for field in FIELDS
            }
        start = max(0, n - self.capacity)  # a shard larger than the ring keeps its tail
        idx = (self._next + np.arange(n - start)) % self.capacity
        for field in FIELDS:
            self._arrays[field][idx] = rows[field][start:]
        self._next = int((self._next + n - start) % self.capacity)
        self.size = min(self.capacity, self.size + n - start)
        return n

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if not self.size:
            raise RuntimeError(f"no distillation rows under {self.root}")
        idx = np.sort(rng.integers(0, self.size, batch_size))
        return {field: self._arrays[field][idx] for field in FIELDS}
