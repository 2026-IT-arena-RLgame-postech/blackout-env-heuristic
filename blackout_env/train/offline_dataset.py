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

FIELDS = ("graphic", "team_state", "agent_states", "actions", "reward", "done")


def save_buffer(buffer: SequentialReplayBuffer, path: Path) -> None:
    """Dumps the first len(buffer) entries (chronological order -- callers never let a
    collection run exceed capacity, so there's no ring wraparound to unscramble here)."""
    n = len(buffer)
    np.savez(path, **{field: getattr(buffer, field)[:n] for field in FIELDS})


def load_dataset_into(buffer: SequentialReplayBuffer, npz_path: Path) -> int:
    """Replays every saved transition through .push(), in original order. This re-derives
    priorities exactly the way collection itself did (freshly pushed = max priority) rather
    than trying to hand-restore the sum/min segment trees, and if npz_path holds more
    transitions than `buffer`'s capacity, the ring buffer's own FIFO eviction naturally keeps
    only the most recent `capacity` of them -- no special-casing needed here for that case.
    Returns the number of transitions loaded (pre-truncation, i.e. what npz_path held)."""
    data = np.load(npz_path)
    n = data["graphic"].shape[0]
    for i in range(n):
        buffer.push(
            data["graphic"][i],
            data["team_state"][i],
            data["agent_states"][i],
            data["actions"][i],
            float(data["reward"][i]),
            bool(data["done"][i]),
        )
    return n


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
