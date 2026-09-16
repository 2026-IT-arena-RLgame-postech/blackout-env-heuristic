"""
Does the greedy policy send a unit back across a tile boundary from both sides?

GUI on the Run 9 40k checkpoint showed Hunters oscillating between two cells. This places each
own unit standing in free space just below / just above (a) a tile boundary and (b) its tile
centre, along x and along y, and reads the greedy action at each point. 'Trapped' = the action at
the lower point moves + along that axis AND the action at the upper point moves -, so a unit that
steps across keeps being sent back. The centre is the control: nothing in the observation should
change discontinuously there.

--freeze recomputes the unit's local wall features from its ORIGINAL position (the other inputs
still see the moved one), attributing any boundary excess to those features. On Run 9 40k
(3x3-around-the-rounded-cell patch + [-0.5, 0.5] offset):

    Hunters   boundary 19.3%  centre 0.2%   (frozen: 0.2% / 0.2%)
    others    boundary  9.5%  centre 0.8%   (frozen: 1.0% / 0.6%)

Usage:
    python examples/probe_boundary_chatter.py --checkpoint <ckpt.pt> --dataset-dir datasets/<name>
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

import blackout_env.model.my_model as my_model
from blackout_env.model.derived_obs import local_wall_features

H = W = 24
CELL = 2.0 / W
HUNTER_COL = 10  # agent_states class one-hot: [Collector, Hunter, Carrier] at columns 9-11


def npz_member_memmap(path: Path, name: str) -> np.memmap:
    """Memory-map one array of an uncompressed .npz (np.savez) without loading the whole file."""
    info = zipfile.ZipFile(path).getinfo(name + ".npy")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        f.seek(info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little"))
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=300, help="dataset states with at least one own Hunter")
    parser.add_argument("--eps", type=float, default=0.12, help="tiles either side (a Hunter moves ~0.24 per decision)")
    parser.add_argument("--freeze", action="store_true", help="local wall features from the unmoved position")
    args = parser.parse_args()

    path = args.dataset_dir / "buffer_a.npz"  # stream A is already in the canonical frame
    states_mm = npz_member_memmap(path, "agent_states")
    rng = np.random.default_rng(1)
    candidates = np.sort(rng.choice(len(states_mm), min(len(states_mm), 60_000), replace=False))
    with_hunter = (np.asarray(states_mm[candidates])[:, :5, HUNTER_COL] > 0.5).any(1)
    idx = candidates[with_hunter][: args.samples]
    graphic = torch.tensor(np.asarray(npz_member_memmap(path, "graphic")[idx]), dtype=torch.float32).permute(0, 3, 1, 2)
    team_state = torch.tensor(np.asarray(npz_member_memmap(path, "team_state")[idx]), dtype=torch.float32)
    states = torch.tensor(np.asarray(states_mm[idx]), dtype=torch.float32)

    model = my_model.MyModel(hidden_size=args.hidden_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["policy_state"])
    model.eval()
    angles = torch.arange(8) * math.pi / 4
    direction = torch.stack([torch.cos(angles), torch.sin(angles)], 1)  # (dx, dy) in obs x/y

    walkable = graphic[:, 1] < 0.5
    low, high, ref, g_rows, t_rows, meta = [], [], [], [], [], []
    for b in range(len(idx)):
        for u in range(5):
            x, y = states[b, u, 0].item(), states[b, u, 1].item()
            col = int(np.clip(round((x + 1) * 0.5 * W - 0.5), 0, W - 1))
            row = int(np.clip(round((1 - y) * 0.5 * H - 0.5), 0, H - 1))
            centre_x, centre_y = (col + 0.5) * CELL - 1, 1 - (row + 0.5) * CELL
            for axis in (0, 1):
                next_row, next_col = (row, col + 1) if axis == 0 else (row - 1, col)  # + along x / +y is up
                if not (0 <= next_row < H and 0 <= next_col < W) or not walkable[b, row, col] or not walkable[b, next_row, next_col]:
                    continue
                boundary = (centre_x + 0.5 * CELL, centre_y) if axis == 0 else (centre_x, centre_y + 0.5 * CELL)
                for kind, (px, py) in (("boundary", boundary), ("centre", (centre_x, centre_y))):
                    for sign, bucket in ((-1, low), (1, high)):
                        moved = states[b].clone()
                        moved[u, 0], moved[u, 1] = px, py
                        moved[u, axis] += sign * args.eps * CELL
                        bucket.append(moved)
                    ref.append(states[b])
                    g_rows.append(graphic[b])
                    t_rows.append(team_state[b])
                    meta.append((u, axis, kind, bool(states[b, u, HUNTER_COL] > 0.5)))

    low_t, high_t, ref_t = torch.stack(low), torch.stack(high), torch.stack(ref)
    g_t, t_t = torch.stack(g_rows), torch.stack(t_rows)
    units = torch.tensor([m[0] for m in meta])
    axes = torch.tensor([m[1] for m in meta])
    kinds = np.array([m[2] for m in meta])
    hunters = np.array([m[3] for m in meta])

    def greedy(moved: torch.Tensor) -> torch.Tensor:
        original = my_model.local_wall_features
        out = []
        try:
            with torch.no_grad():
                for i in range(0, len(moved), 256):
                    sl = slice(i, i + 256)
                    if args.freeze:
                        my_model.local_wall_features = lambda g, a, r=ref_t[sl]: local_wall_features(g, r)
                    q, *_ = model(g_t[sl], t_t[sl], moved[sl], tau=torch.full((len(moved[sl]), 8), 0.5))
                    out.append(q)
        finally:
            my_model.local_wall_features = original
        return torch.cat(out)[torch.arange(len(moved)), units].argmax(-1)

    a_low, a_high = greedy(low_t), greedy(high_t)
    trapped = ((direction[a_low, axes] > 0.1) & (direction[a_high, axes] < -0.1)).numpy()
    changed = (a_low != a_high).numpy()
    for who, mask in (("hunter", hunters), ("non-hunter", ~hunters)):
        parts = []
        for kind in ("boundary", "centre"):
            m = mask & (kinds == kind)
            parts.append(f"{kind}: trapped={trapped[m].mean():.3f} action_changed={changed[m].mean():.3f} (n={m.sum()})")
        print(f"{'frozen ' if args.freeze else ''}{who:10s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
