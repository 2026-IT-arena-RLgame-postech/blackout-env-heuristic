"""
Check the canonical team frame against real recorded matches.

blackout_env.env.team_frame maps a Team B observation into the frame the network is trained on:
the grid is anti-transposed, positions swap x/y, unit rows roll by five and every compass index
is reflected. A sign error anywhere in that chain is silent -- training simply learns from
actions that point the wrong way -- so this verifies the mapping against data where the ground
truth is observable: what the unit actually did next.

For consecutive dataset rows it compares the stored action against the direction the unit really
moved, in the world frame and again after mirroring both. The two rates must be identical: the
mirror is a relabelling of the same physics, not a different one. It also checks that unit
positions and the grid stay aligned through the mirror, and that the map really is symmetric
about the y = x diagonal (`tests/test_team_frame.py` covers that too, on a few rows).

Usage:
    python -m examples.verify_team_frame_on_dataset --dataset-dir datasets/heuristic_mixv3_20260915
"""

from __future__ import annotations

import argparse
import ast
import io
import zipfile
from pathlib import Path

import numpy as np

from blackout_env.env.team_frame import (
    N_TEAM,
    canonical_unit_row,
    mirror_action_idx,
    mirror_agent_states,
    mirror_graphic,
)
from blackout_env.model.my_policy import DIRECTION_VECTORS

WALL, SPAWN_ALLY, SPAWN_ENEMY, STORAGE_ALLY, STORAGE_ENEMY = 1, 4, 5, 6, 7
MOVED = 0.02  # normalized units; below this the unit did not go anywhere this tick
ALIGNED = 0.7  # cosine between the commanded direction and the actual displacement


def _load(path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive, archive.open(name) as handle:
        return np.load(io.BytesIO(handle.read()))


class GraphicReader:
    """Random access to one row of a (uncompressed) graphic.npy inside the dataset zip."""

    def __init__(self, path: Path) -> None:
        self._archive = zipfile.ZipFile(path)
        self._handle = self._archive.open("graphic.npy")
        header = self._handle.read(10)
        header_len = int.from_bytes(header[8:10], "little")
        meta = ast.literal_eval(self._handle.read(header_len).decode().strip())
        self.shape, self.dtype = meta["shape"], np.dtype(meta["descr"])
        self._offset = 10 + header_len
        self._row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize

    def __getitem__(self, index: int) -> np.ndarray:
        self._handle.seek(self._offset + int(index) * self._row_bytes)
        return np.frombuffer(self._handle.read(self._row_bytes), dtype=self.dtype).reshape(self.shape[1:]).copy()

    def close(self) -> None:
        self._handle.close()
        self._archive.close()


def _cell(position: np.ndarray, height: int, width: int) -> tuple[int, int]:
    return (min(height - 1, max(0, int(round((1.0 - position[1]) * 0.5 * height - 0.5)))),
            min(width - 1, max(0, int(round((position[0] + 1.0) * 0.5 * width - 0.5)))))


def check_actions(path: Path, samples: int, rng: np.random.Generator) -> None:
    actions = _load(path, "actions.npy")
    states = _load(path, "agent_states.npy")
    done = _load(path, "done.npy")
    candidates = rng.choice(len(done) - 1, samples, replace=False)

    world_hits = mirror_hits = total = 0
    for index in candidates:
        if done[index]:
            continue
        before, after = states[index], states[index + 1]
        delta = after[:, :2] - before[:, :2]
        moving = np.linalg.norm(delta, axis=1) > MOVED
        if not moving.any():
            continue

        commanded = DIRECTION_VECTORS[actions[index]]
        cosine = (commanded * delta).sum(1) / (np.linalg.norm(commanded, axis=1) * np.linalg.norm(delta, axis=1) + 1e-9)

        m_delta = mirror_agent_states(after)[:, :2] - mirror_agent_states(before)[:, :2]
        m_commanded = DIRECTION_VECTORS[mirror_action_idx(actions[index])]
        m_cosine = (m_commanded * m_delta).sum(1) / (
            np.linalg.norm(m_commanded, axis=1) * np.linalg.norm(m_delta, axis=1) + 1e-9
        )
        m_moving = np.roll(moving, -N_TEAM)  # the mirror rolls the unit rows

        world_hits += int((cosine[moving] > ALIGNED).sum())
        mirror_hits += int((m_cosine[m_moving] > ALIGNED).sum())
        total += int(moving.sum())

    print(f"{path.name}: moving unit-ticks={total}  action matches movement: "
          f"world={world_hits / max(1, total):.4f}  mirrored={mirror_hits / max(1, total):.4f}")
    if total and abs(world_hits - mirror_hits) > 0:
        print("  MISMATCH: the mirrored action table does not describe the same motion")


def check_alignment(path: Path, samples: int, rng: np.random.Generator) -> None:
    reader = GraphicReader(path)
    states = _load(path, "agent_states.npy")
    height = width = reader.shape[1]
    wall_mismatch = cell_mismatch = checked = 0
    try:
        for index in rng.choice(len(states), samples, replace=False):
            graphic, state = reader[index], states[index]
            m_graphic, m_state = mirror_graphic(graphic), mirror_agent_states(state)
            for unit in range(state.shape[0]):
                row, col = _cell(state[unit, :2], height, width)
                m_row, m_col = _cell(m_state[canonical_unit_row(unit, team_b=True), :2], height, width)
                checked += 1
                if bool(graphic[row, col, WALL] > 0.5) != bool(m_graphic[m_row, m_col, WALL] > 0.5):
                    wall_mismatch += 1
                if (m_row, m_col) != (width - 1 - col, height - 1 - row):
                    cell_mismatch += 1
        symmetric = 0
        for index in rng.choice(len(states), min(20, samples), replace=False):
            graphic = reader[index]
            mirrored = mirror_graphic(graphic)
            symmetric += int(
                np.array_equal(mirrored[..., WALL] > 0.5, graphic[..., WALL] > 0.5)
                and np.array_equal(mirrored[..., SPAWN_ALLY] > 0.5, graphic[..., SPAWN_ENEMY] > 0.5)
                and np.array_equal(mirrored[..., STORAGE_ALLY] > 0.5, graphic[..., STORAGE_ENEMY] > 0.5)
            )
        print(f"{path.name}: unit cells checked={checked}  wall mismatches={wall_mismatch}  "
              f"cell mismatches={cell_mismatch}  maps symmetric about y=x: {symmetric}/{min(20, samples)}")
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--action-samples", type=int, default=4000)
    parser.add_argument("--alignment-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for stream in ("a", "b"):
        path = args.dataset_dir / f"buffer_{stream}.npz"
        check_actions(path, args.action_samples, rng)
    check_alignment(args.dataset_dir / "buffer_b.npz", args.alignment_samples, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
