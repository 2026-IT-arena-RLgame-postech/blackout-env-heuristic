"""
Shared save/load helpers for the offline heuristic-dataset format (buffer_a.npz/buffer_b.npz):
one .npz per team stream, holding exactly the fields SequentialReplayBuffer.push() takes.

Used by:
  - collect_heuristic_dataset.py (write, one shard per process)
  - collect_heuristic_dataset_parallel.py (merge N shards into one dataset)
  - offline_pretrain.py (read, to pretrain purely offline)
  - qmix_trainer.py's --seed-dataset-dir (read, to preload an online run's buffers)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackout_env.train.replay_buffer import SequentialReplayBuffer
from blackout_env.train.reward_shaping import blocked_penalty_adjustment

FIELDS = ("graphic", "team_state", "agent_states", "actions", "reward", "done")


def npz_member_memmap(path: Path, name: str) -> np.memmap:
    """Memory-map one array of an uncompressed .npz (np.savez) without loading the whole file."""
    import zipfile

    info = zipfile.ZipFile(path).getinfo(name + ".npy")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        f.seek(info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little"))
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def save_buffer(buffer: SequentialReplayBuffer, path: Path) -> None:
    """Dumps the first len(buffer) entries (chronological order -- callers never let a
    collection run exceed capacity, so there's no ring wraparound to unscramble here)."""
    n = len(buffer)
    np.savez(path, **{field: getattr(buffer, field)[:n] for field in FIELDS})


def load_dataset_into(
    buffer: SequentialReplayBuffer,
    npz_path: Path,
    team_indices: tuple[int, ...] | None = None,
    penalty_per_unit: float = 0.0,
    reward_v2=None,
    annotate_workers: int = 8,
    drop_dead: bool = False,
) -> int:
    """Replays every saved transition through .push(), in original order. This re-derives
    priorities exactly the way collection itself did (freshly pushed = max priority) rather
    than trying to hand-restore the sum/min segment trees, and if npz_path holds more
    transitions than `buffer`'s capacity, the ring buffer's own FIFO eviction naturally keeps
    only the most recent `capacity` of them -- no special-casing needed here for that case.
    Returns the number of transitions loaded (pre-truncation, i.e. what npz_path held).

    team_indices/penalty_per_unit: if given (team_indices non-None and penalty_per_unit != 0),
    retroactively folds reward_shaping.blocked_penalty_adjustment into every loaded transition's
    reward -- see that function's docstring. Off by default (both existing callers of this
    function, qmix_trainer.py's --seed-dataset-dir and older scripts, keep their exact prior
    behavior unless they opt in); offline_pretrain.py passes real values explicitly.

    reward_v2: a RewardV2Config replaces the stored (Unity-shaped) reward with reward v2 --
    terminal outcome + confirmed score change, with the team potential stored alongside for the
    n-step return to shape with the learner's gamma (see train/reward_v2.annotate_sequence). The
    blocked penalty is still added on top.

    drop_dead: skip every training segment that starts with no battery left anywhere (see
    train/dead_segments.py), moving each shortened match's outcome onto its new last row.
    Returns the number of rows pushed.
    """
    data = np.load(npz_path)
    # NpzFile.__getitem__ re-reads and re-decompresses the whole member array from the zip on
    # every call (no caching) -- indexing it per-row inside the loop below would re-read each
    # multi-GB array once per transition. Load each field into memory exactly once instead.
    arrays = {field: data[field] for field in FIELDS}
    n = arrays["graphic"].shape[0]
    reward, done = arrays["reward"], arrays["done"]
    potential = np.zeros(n, dtype=np.float32)
    terminal = done
    if reward_v2 is not None:
        from blackout_env.train.reward_v2 import annotate_dataset

        annotated = annotate_dataset(npz_path, reward_v2, workers=annotate_workers)
        reward, done, potential, terminal = annotated["reward"], annotated["done"], annotated["potential"], annotated["terminal"]
    keep = np.ones(n, dtype=bool)
    if drop_dead:
        from blackout_env.train.dead_segments import batteries_in_play, drop_dead_segments
        from blackout_env.train.reward_v2 import TERMINAL_REWARD, match_ends

        ends = match_ends(arrays["team_state"])
        if reward_v2 is not None:
            outcome = annotated["outcome"]
        else:
            outcome = np.where(ends & done, np.clip(np.rint(reward / TERMINAL_REWARD), -1, 1) * TERMINAL_REWARD, 0.0)
            terminal = ends.copy()
            done = done | ends
        batteries = np.concatenate([batteries_in_play(arrays["graphic"][i : i + 65536], arrays["agent_states"][i : i + 65536]) for i in range(0, n, 65536)])
        reward, done, terminal, keep = drop_dead_segments(reward, done, terminal, batteries, ends, outcome)
    if team_indices is not None and penalty_per_unit != 0.0:
        reward = reward + blocked_penalty_adjustment(arrays["agent_states"], done, team_indices, penalty_per_unit)
    for i in np.flatnonzero(keep):
        buffer.push(
            arrays["graphic"][i],
            arrays["team_state"][i],
            arrays["agent_states"][i],
            arrays["actions"][i],
            float(reward[i]),
            bool(done[i]),
            potential=float(potential[i]),
            terminal=bool(terminal[i]),
        )
    return int(keep.sum())


def merge_shards(shard_paths: list[Path], out_path: Path) -> int:
    """Concatenates same-schema .npz shards (e.g. one per parallel collection worker) into one
    dataset file, in the given order. Returns the merged transition count."""
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    for p in shard_paths:
        data = np.load(p)
        for field in FIELDS:
            arrays[field].append(data[field])
    merged = {field: np.concatenate(arrays[field], axis=0) for field in FIELDS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **merged)
    return merged["graphic"].shape[0]
