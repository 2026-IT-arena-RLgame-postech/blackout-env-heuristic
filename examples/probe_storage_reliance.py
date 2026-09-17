"""
Does a carrying unit head for storage the map SHOWS, or for where storages USUALLY are?

Storages are drawn from a fixed set of candidates each match (StorageCountPerTeam = 3 of the
procedural candidates, plus the fixed spawn storage), and the wall layout never changes, so a
network can score well by memorising candidate positions instead of reading the storage channel.
In GUI the Run 10 final checkpoint carried batteries to a candidate that was inactive that match
and waited there.

For dataset states where own units carry a battery, this edits the graphic two ways and reads
where each carrying unit's greedy action points:
  remove  delete one ACTIVE candidate storage from the ally-storage channel
  add     paint one INACTIVE candidate in as an ally storage
and reports how the action's component toward that candidate changes. A policy that reads the
map should turn away after `remove` and toward after `add`; a heuristic answering the same
states is the reference.

Usage:
    python examples/probe_storage_reliance.py --checkpoint <ckpt.pt> --dataset-dir datasets/<name>
    python examples/probe_storage_reliance.py --heuristic strategic_v4 --dataset-dir datasets/<name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from blackout_env.env.constants import team_a_agents, unit_index
from blackout_env.heuristics import make_heuristic
from blackout_env.model.my_model import MyModel
from blackout_env.model.my_policy import DIRECTION_VECTORS, direction_vector_to_idx
from probe_boundary_chatter import npz_member_memmap

H = W = 24
STORAGE_ALLY, BATTERY, CARGO_COL = 6, 8, 4


def components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a bool [H, W] mask, as (row, col) index arrays."""
    seen = np.zeros_like(mask)
    blobs = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        blobs.append(np.array(cells))
    return blobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--heuristic")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    path = args.dataset_dir / "buffer_a.npz"
    graphic_mm = npz_member_memmap(path, "graphic")
    states_mm = npz_member_memmap(path, "agent_states")
    team_mm = npz_member_memmap(path, "team_state")
    done = np.asarray(npz_member_memmap(path, "done"))

    # Candidate blobs = components of "ever an ally storage", from match-start frames.
    starts = np.r_[0, np.flatnonzero(done)[:-1] + 1][::5][:400]
    ever = (np.asarray(graphic_mm[starts][..., STORAGE_ALLY]) > 0.5).any(0)
    candidates = components(ever)

    rng = np.random.default_rng(2)
    pool = np.sort(rng.choice(len(states_mm), 80_000, replace=False))
    carrying = (np.asarray(states_mm[pool])[:, :5, CARGO_COL] > 1e-3).any(1)
    idx = pool[carrying][: args.samples]

    model = None
    if args.checkpoint is not None:
        model = MyModel(hidden_size=args.hidden_size)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["policy_state"])
        model.eval()
    agent_for_row = {unit_index(a): a for a in team_a_agents()}

    def actions(graphic: np.ndarray, states: np.ndarray, team: np.ndarray) -> np.ndarray:
        """[5] greedy direction index per own unit."""
        if model is not None:
            with torch.no_grad():
                q, *_ = model(
                    torch.tensor(graphic).permute(2, 0, 1)[None], torch.tensor(team)[None],
                    torch.tensor(states)[None], tau=torch.full((1, 8), 0.5),
                )
            return q[0, :5].argmax(-1).numpy()
        view = {"graphic": graphic, "agent_states": states, "team_state": team}
        out = make_heuristic(args.heuristic).act({a: view for a in agent_for_row.values()})
        return np.array([int(direction_vector_to_idx(np.asarray(out[agent_for_row[u]])[None])[0]) for u in range(5)])

    results = {"remove": [], "add": []}
    for i in idx:
        graphic = np.array(graphic_mm[i])
        states = np.array(states_mm[i])
        team = np.array(team_mm[i])
        active = graphic[..., STORAGE_ALLY] > 0.5
        for blob in candidates:
            rows, cols = blob[:, 0], blob[:, 1]
            if graphic[rows, cols, BATTERY].max() > 1e-3:
                continue  # an item sitting on it would turn into a loose battery when edited
            is_active = bool(active[rows, cols].all())
            is_inactive = not active[rows, cols].any()
            if not (is_active or is_inactive):
                continue
            edited = graphic.copy()
            edited[rows, cols, STORAGE_ALLY] = 0.0 if is_active else 1.0
            before, after = actions(graphic, states, team), actions(edited, states, team)
            centre = np.array([cols.mean(), rows.mean()])  # (col, row)
            for u in range(5):
                if states[u, CARGO_COL] <= 1e-3:
                    continue
                col = (states[u, 0] + 1) * 0.5 * W - 0.5
                row = (1 - states[u, 1]) * 0.5 * H - 0.5
                to_blob = np.array([centre[0] - col, -(centre[1] - row)])  # obs x right, y up
                dist = np.linalg.norm(to_blob)
                if dist < 1.0 or dist > 10.0:
                    continue  # on it already, or too far for this one storage to decide
                to_blob /= dist
                results["remove" if is_active else "add"].append(
                    (float(DIRECTION_VECTORS[before[u]] @ to_blob), float(DIRECTION_VECTORS[after[u]] @ to_blob),
                     int(before[u] != after[u]))
                )

    label = args.heuristic or str(args.checkpoint)
    for kind, rows in results.items():
        if not rows:
            print(f"{label} {kind}: no cases")
            continue
        arr = np.array(rows)
        print(
            f"{label} {kind:6s} n={len(arr):4d}  toward-candidate before={arr[:, 0].mean():+.3f} "
            f"after={arr[:, 1].mean():+.3f}  (shift {arr[:, 1].mean() - arr[:, 0].mean():+.3f})  action changed={arr[:, 2].mean():.2f}"
        )


if __name__ == "__main__":
    main()
