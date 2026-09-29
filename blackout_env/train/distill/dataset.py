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

For analysis only (the trainer does not load them):

  actions       uint8 [T, 5]        compass index each unit executed (canonical; 255 = stood still)
  student       uint8 [T, 5]        the student's choice, whether or not it drove (255 = not asked)
  teacher_drove bool  [T, 5]        the teacher's move was executed
  meta          str                 the match summary as JSON (task, outcome, recorder counts)

Ticks where the teacher labels no unit are not stored.
"""

from __future__ import annotations

import json
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
EXTRA_FIELDS = ("actions", "student", "teacher_drove")


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


def save_shard(path: Path, rows: dict[str, np.ndarray], meta: dict | None = None) -> None:
    tmp = path.with_suffix(".tmp.npz")
    arrays = {field: rows[field] for field in FIELDS + EXTRA_FIELDS if field in rows}
    if meta is not None:
        arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp, **arrays)  # maps are mostly zeros
    tmp.rename(path)  # readers only ever see complete shards


def shard_meta(shard) -> dict:
    return json.loads(str(shard["meta"])) if "meta" in shard else {}


class DistillDataset:
    """The most recent ``capacity`` rows of every shard under ``root``, sampled uniformly.

    New shards are picked up oldest first, either one by one (``new_shards`` + ``append``, so a
    caller can look at rows before they enter the ring) or all at once (``refresh``).  Once full,
    new rows overwrite the oldest (DAgger aggregates, but a laptop cannot hold every row ever
    played).  Each ring row remembers the student version that played it (0 = teacher only)."""

    def __init__(self, root: Path, capacity: int):
        self.root = Path(root)
        self.capacity = int(capacity)
        self._arrays: dict[str, np.ndarray] | None = None
        self._version = np.zeros(self.capacity, dtype=np.int32)
        self._loaded: set[str] = set()
        self._next = 0
        self.size = 0

    def new_shards(self):
        """Yields (rows, meta) for each shard not seen yet, and marks it seen."""
        for path in sorted(self.root.glob("*.npz")):
            if path.name in self._loaded or path.name.endswith(".tmp.npz"):
                continue
            with np.load(path) as shard:
                rows, meta = {field: shard[field] for field in FIELDS}, shard_meta(shard)
            self._loaded.add(path.name)
            yield rows, meta

    def refresh(self) -> int:
        """Append every new shard; returns the number of rows added."""
        return sum(self.append(rows, meta.get("student_version", 0)) for rows, meta in self.new_shards())

    def append(self, rows: dict[str, np.ndarray], version: int = 0) -> int:
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
        self._version[idx] = version
        self._next = int((self._next + n - start) % self.capacity)
        self.size = min(self.capacity, self.size + n - start)
        return n

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if not self.size:
            raise RuntimeError(f"no distillation rows under {self.root}")
        idx = np.sort(rng.integers(0, self.size, batch_size))
        return {field: self._arrays[field][idx] for field in FIELDS}

    def composition(self, current_version: int) -> dict[str, float]:
        """How old and how teacher-driven the ring is -- what the learner is fitting right now."""
        if not self.size:
            return {}
        versions = self._version[: self.size]
        kinds = self._arrays["kinds"][: self.size]
        labelled = kinds > 0
        out = {
            "ring/rows": self.size,
            "ring/teacher_only_frac": float(np.mean(versions == 0)),
            "ring/mean_version_age": float(np.mean(current_version - versions)),
            "ring/labelled_unit_frac": float(labelled.mean()),
        }
        for index, kind in enumerate(KINDS):
            if kind:
                out[f"ring/kind_share_{kind}"] = float((kinds == index).sum() / max(1, labelled.sum()))
        return out
