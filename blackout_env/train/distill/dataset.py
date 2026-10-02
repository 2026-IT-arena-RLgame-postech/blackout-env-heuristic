"""Distillation data: one compressed .npz shard per match, aggregated in a fixed-size ring.

Every row is one tick of the learner's team, already in the canonical team frame
(env/team_frame.py) so the trainer feeds it to the net unchanged:

  graphic       uint8 [T, H, W, C]  the observation times 15 (lossless: every channel is 0/1
                                    except batteries, whose amount/15 has 15ths only)
  team_state    f32   [T, 4]
  agent_states  f32   [T, 10, D]    own units in rows 0-4
  labels        uint8 [T, 5]        bit i = compass direction i is correct for own unit k
                                    (canonical frame; 0 = no label).  A K-bin run (P4,
                                    --action-bins K > 8): uint64, bit i = heading bin i
                                    (model/direction_bins.py); K is the meta's ``action_bins``
  kinds         uint8 [T, 5]        index into KINDS: what the label is doing

For analysis only (the trainer does not load them):

  actions       uint8 [T, 5]        compass (K-bin) index each unit executed (canonical; 255 = stood still)
  student       uint8 [T, 5]        the student's choice, whether or not it drove (255 = not asked)
  teacher_drove bool  [T, 5]        the teacher's move was executed
  meta          str                 the match summary as JSON (task, outcome, recorder counts)

Loaded into the ring beside FIELDS (for sequence windows, ``DistillDataset.sample_windows``):

  tick          int32 [T]           learner decision step within the match (0 = first).  Every
                                    full-window tick is stored in order, labelled or not; partial
                                    windows are counted but not stored, so a jump > 1 marks one.
  angles        f32   [T, 5]        P11: an act-teacher's raw heading per own unit, radians in the
                                    bins' canonical frame (team B mirrored like its label); NaN =
                                    none (no move, not an act teacher, or a shard from before P11:
                                    loaded as all NaN, so those rows carry no angle loss)

For analysis only, too: ``student_angles`` f32 [T, 5], the student's executed heading (NaN = not
asked or stood).

Shards written before 2026-09-30 dropped ticks without any label and have no ``tick``; loading
reconstructs it from ``team_state[2]`` (time left, 25 decisions/s: exact to ~0.02 tick, offset by
a per-match constant of 0 or 1 from a counted tick), or -1 = unknown when not strictly increasing.

A ring (and ``new_shards``) built with ``action_bins=K`` re-expresses every shard's labels at K
on load (``direction_bins.convert_bits``): an 8-way uint8 shard in a K-bin ring widens each
direction to the bins within +-22.5 deg; ``action_bins=None`` keeps them as stored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackout_env.model.direction_bins import bits_bins, convert_bits, unpack_bits

GRAPHIC_SCALE = 15.0
N_TEAM = 5
KINDS = (
    "", "battery", "steal", "special", "deposit", "wait_storage", "transform",
    "hunt", "patrol_defend", "patrol_raid", "idle",
    "stand",  # never stored: --stand-action's target where the teacher has no label (and stands)
    "act",  # an act-teacher's own move (act_teacher.py); after "stand" so no existing index moves
)
KIND_INDEX = {kind: i for i, kind in enumerate(KINDS)}
FIELDS = ("graphic", "team_state", "agent_states", "labels", "kinds")
EXTRA_FIELDS = ("actions", "student", "teacher_drove", "tick", "angles", "student_angles")
SEQUENCE_FIELDS = ("tick",)  # loaded into the ring beside FIELDS (not fed to the net)
ANGLES = "angles"  # loaded into the ring beside FIELDS and sampled with them (P11's angle loss)
MATCH_TICKS = 420 * 25  # match seconds x decisions per second
UNKNOWN_TICK = -1


def encode_graphic(graphic: np.ndarray) -> np.ndarray:
    encoded = np.rint(graphic * GRAPHIC_SCALE).astype(np.uint8)
    if not np.array_equal(encoded / np.float32(GRAPHIC_SCALE), graphic.astype(np.float32)):
        raise ValueError("graphic is not in 15ths; the uint8 shard format would lose information")
    return encoded


def decode_graphic(encoded: np.ndarray) -> np.ndarray:
    return encoded.astype(np.float32) / np.float32(GRAPHIC_SCALE)


def unpack_labels(bits: np.ndarray, bins: int = 8) -> np.ndarray:
    """[..., 5] uint8 (uint64: K-bin) bit sets -> [..., 5, 8 or K] bool."""
    if bits.dtype == np.uint8 and bins == 8:
        return np.unpackbits(bits[..., None], axis=-1, bitorder="little").astype(bool)
    return unpack_bits(bits, bins)


def labels_at(rows: dict, meta: dict, action_bins: int | None) -> np.ndarray:
    """A shard's labels at ``action_bins`` bins (None = as stored)."""
    labels = rows["labels"]
    if action_bins is None:
        return labels
    return convert_bits(labels, bits_bins(labels, meta.get("action_bins")), action_bins)


def save_shard(path: Path, rows: dict[str, np.ndarray], meta: dict | None = None) -> None:
    tmp = path.with_suffix(".tmp.npz")
    arrays = {field: rows[field] for field in FIELDS + EXTRA_FIELDS if field in rows}
    if meta is not None:
        arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp, **arrays)  # maps are mostly zeros
    tmp.rename(path)  # readers only ever see complete shards


def reconstruct_ticks(team_state: np.ndarray) -> np.ndarray:
    """Decision ticks of a shard without ``tick``, from its time left (team_state[2]); all
    UNKNOWN_TICK when they are not strictly increasing (not one match in order)."""
    ticks = np.rint((1.0 - team_state[:, 2].astype(np.float64)) * MATCH_TICKS).astype(np.int32)
    if len(ticks) > 1 and not np.all(np.diff(ticks) > 0):
        return np.full(len(ticks), UNKNOWN_TICK, dtype=np.int32)
    return ticks


def _ticks_of(rows) -> np.ndarray:
    if "tick" in rows:
        return np.asarray(rows["tick"], dtype=np.int32)
    return reconstruct_ticks(np.asarray(rows["team_state"]))


def angles_of(rows) -> np.ndarray:
    """A shard's ``angles`` [T, 5] float32, all NaN for a shard without them (before P11)."""
    if ANGLES in rows:
        return np.asarray(rows[ANGLES], dtype=np.float32)
    return np.full((len(rows["labels"]), N_TEAM), np.nan, dtype=np.float32)


def shard_meta(shard) -> dict:
    return json.loads(str(shard["meta"])) if "meta" in shard else {}


def new_shards(root: Path, seen: set[str], action_bins: int | None = None):
    """(rows, meta) of every complete shard under ``root`` not in ``seen``, oldest first; each is
    added to ``seen``.  ``action_bins``: labels re-expressed at that many bins (``labels_at``)."""
    for path in sorted(Path(root).glob("*.npz")):
        if path.name in seen or path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as shard:
            rows, meta = {field: shard[field] for field in FIELDS}, shard_meta(shard)
            rows["tick"] = _ticks_of(shard)
            rows[ANGLES] = angles_of(shard)
            rows["labels"] = labels_at(rows, meta, action_bins)
        seen.add(path.name)
        yield rows, meta


class DistillDataset:
    """The most recent ``capacity`` rows of every shard under ``root``, sampled uniformly.

    New shards are picked up oldest first, either one by one (``new_shards`` + ``append``, so a
    caller can look at rows before they enter the ring) or all at once (``refresh``).  Once full,
    new rows overwrite the oldest (DAgger aggregates, but a laptop cannot hold every row ever
    played).  Each ring row remembers the student version that played it (0 = teacher only), its
    match (a running count of appended shards) and its decision tick, so ``sample_windows`` can
    cut windows of consecutive ticks that never cross a match or the ring's overwrite boundary."""

    def __init__(self, root: Path, capacity: int, action_bins: int | None = None):
        self.root = Path(root)
        self.capacity = int(capacity)
        self.action_bins = action_bins  # labels re-expressed at K bins on load (None: as stored)
        self._arrays: dict[str, np.ndarray] | None = None
        self._version = np.zeros(self.capacity, dtype=np.int32)
        self._match = np.full(self.capacity, -1, dtype=np.int64)
        self._tick = np.full(self.capacity, UNKNOWN_TICK, dtype=np.int32)
        self._angles = np.full((self.capacity, N_TEAM), np.nan, dtype=np.float32)
        self._labelled = np.zeros(self.capacity, dtype=bool)
        self._matches = 0
        self._loaded: set[str] = set()
        self._next = 0
        self.size = 0

    def new_shards(self):
        """Yields (rows, meta) for each shard not seen yet, and marks it seen."""
        yield from new_shards(self.root, self._loaded, self.action_bins)

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
        self._match[idx] = self._matches
        self._tick[idx] = _ticks_of(rows)[start:]
        self._angles[idx] = angles_of(rows)[start:]
        self._labelled[idx] = rows["labels"][start:].reshape(n - start, -1).any(axis=1)
        self._matches += 1
        self._next = int((self._next + n - start) % self.capacity)
        self.size = min(self.capacity, self.size + n - start)
        return n

    def sample(self, batch_size: int, rng: np.random.Generator, labelled_only: bool = True) -> dict[str, np.ndarray]:
        """Uniform rows.  ``labelled_only`` (default) draws only rows where some unit has a label,
        as before unlabelled ticks were stored; with --stand-action the others are all-STAND
        targets, which ``labelled_only=False`` includes."""
        if not self.size:
            raise RuntimeError(f"no distillation rows under {self.root}")
        idx = rng.integers(0, self.size, batch_size)
        if labelled_only:
            idx = self._labelled_draw(idx, batch_size, rng)
        idx = np.sort(idx)
        return {**{field: self._arrays[field][idx] for field in FIELDS}, ANGLES: self._angles[idx]}

    def _labelled_draw(self, idx: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        keep = idx[self._labelled[idx]]
        for _ in range(4):  # rejection: nearly every stored tick has a label
            if len(keep) >= batch_size:
                return keep[:batch_size]
            more = rng.integers(0, self.size, 2 * batch_size)
            keep = np.concatenate((keep, more[self._labelled[more]]))
        if len(keep) >= batch_size:
            return keep[:batch_size]
        labelled = np.flatnonzero(self._labelled[: self.size])
        if not len(labelled):
            raise RuntimeError(f"no labelled distillation rows under {self.root}")
        return labelled[rng.integers(0, len(labelled), batch_size)]

    def sample_windows(self, batch_size: int, length: int, burn_in: int, rng: np.random.Generator,
                       max_gap: int = 1) -> dict[str, np.ndarray]:
        """``batch_size`` windows of ``T = burn_in + length`` consecutive ring rows.

        Each window is placed around one target row drawn uniformly from the ring, at a uniform
        position among the ``length`` trained positions, so every row is trained equally often
        at every position (windows near a match's start or end are padded).  The valid part of a
        window is the run of rows around the target in the same match with ticks increasing by
        1..``max_gap`` (1: strictly consecutive, any skipped partial window breaks the run;
        larger bridges them, which matches a student that only steps its memory on full
        windows), never past the ring's oldest row (the overwritten boundary) or its newest.

        Returns FIELDS as [B, T, ...] (zeros where invalid) plus
          angles    f32   [B, T, 5] the teacher's raw headings (NaN where invalid or none)
          tick      int32 [B, T]  decision tick (-1 where invalid or unknown)
          match     int64 [B, T]  ring match id (-1 where invalid)
          valid     bool  [B, T]  a real row of the target's run
          reset     bool  [B, T]  the run starts here (match start, a gap, or the oldest row
                                  left in the ring): zero the recurrent state before this tick.
                                  Position 0 of a window that starts mid-run is not a reset --
                                  the state starts at zero there anyway and burn-in warms it
          loss_mask bool  [B, T]  valid and not burn-in (>= 1 per window: the target)
        """
        if not self.size:
            raise RuntimeError(f"no distillation rows under {self.root}")
        if length < 1 or burn_in < 0 or max_gap < 1:
            raise ValueError("need length >= 1, burn_in >= 0, max_gap >= 1")
        span = burn_in + length
        base = self._next if self.size == self.capacity else 0  # physical index of the oldest row
        target = rng.integers(0, self.size, batch_size)  # logical: 0 = oldest row in the ring
        position = burn_in + rng.integers(0, length, batch_size)
        logical = (target - position)[:, None] + np.arange(-1, span)[None, :]  # one extra row in front
        exists = (logical >= 0) & (logical < self.size)
        physical = (base + np.clip(logical, 0, self.size - 1)) % self.capacity
        match, tick = self._match[physical], self._tick[physical]
        gap = tick[:, 1:] - tick[:, :-1]
        # link[:, j]: window row j continues the run of the row before it
        link = (exists[:, 1:] & exists[:, :-1] & (match[:, 1:] == match[:, :-1])
                & (tick[:, :-1] >= 0) & (gap > 0) & (gap <= max_gap))
        breaks = np.cumsum(~link, axis=1)
        valid = exists[:, 1:] & (breaks == breaks[np.arange(batch_size), position][:, None])
        reset = valid & ~link
        loss_mask = valid & (np.arange(span) >= burn_in)[None, :]

        rows = physical[:, 1:].reshape(-1)
        invalid = ~valid.reshape(-1)
        out = {}
        for field in FIELDS:
            values = self._arrays[field][rows]
            values[invalid] = 0
            out[field] = values.reshape(batch_size, span, *values.shape[1:])
        angles = self._angles[rows]
        angles[invalid] = np.nan
        out[ANGLES] = angles.reshape(batch_size, span, N_TEAM)
        out["tick"] = np.where(valid, tick[:, 1:], UNKNOWN_TICK).astype(np.int32)
        out["match"] = np.where(valid, match[:, 1:], -1)
        out.update(valid=valid, reset=reset, loss_mask=loss_mask)
        return out

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
            "ring/angle_unit_frac": float(np.isfinite(self._angles[: self.size]).mean()),  # P11 angle targets
        }
        for index, kind in enumerate(KINDS):
            if kind:
                out[f"ring/kind_share_{kind}"] = float((kinds == index).sum() / max(1, labelled.sum()))
        return out
