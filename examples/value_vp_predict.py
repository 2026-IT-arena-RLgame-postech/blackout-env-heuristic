"""Predictions of the value nets of the velocity x pooling A/B on the S1 row cache (reports/value/vel_pool_ab.md).

    python examples/value_vp_predict.py --cache datasets/value_s1_r64 --out reports/value/vel_pool_ab/preds \
        --ckpt base_s0=runs/base_s0/latest.pt vel_s0=runs/vel_s0/latest.pt ...

Same evaluation rows as examples/value_dist_ab_eval.py (heldout = all held-out rows, unseen_policy /
unseen_ood = 60k rows drawn with default_rng(0)), same for every checkpoint.  Writes ``rows.npz``
(targets, once) and ``<name>.npz`` (final-lead prediction in points, W/D/L probabilities) per
checkpoint; examples/value_vp_analyze.py turns them into the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from blackout_env.train.accel import pick_device  # noqa: E402
from blackout_env.value.baselines import split_masks  # noqa: E402
from blackout_env.value.data import open_cache  # noqa: E402
from blackout_env.value.model import REG_SCALE, load_value_net  # noqa: E402

SPLITS = ("heldout", "unseen_policy", "unseen_ood")


@torch.no_grad()
def predict(net, cache: dict, idx: np.ndarray, device, batch: int = 1024) -> dict:
    parts = []
    for a in range(0, len(idx), batch):
        j = idx[a:a + batch]
        g = torch.from_numpy(np.asarray(cache["graphic"][j])).to(device)
        ts = torch.from_numpy(np.asarray(cache["team_state"][j])).to(device)
        ag = torch.from_numpy(np.asarray(cache["agent_states"][j])).to(device)
        vel = torch.from_numpy(np.asarray(cache["agent_vel"][j])).to(device) if net.vel else None
        out = net.forward_full_u8(g, ts, ag, vel)[0].float()
        h = net.n_horizons
        lead = ts[:, 0] - ts[:, 1]
        parts.append({"p": torch.softmax(out[:, h + 1:], -1).cpu().numpy(),
                      "final_pts": ((lead * 100) + out[:, h] * (100 / REG_SCALE)).cpu().numpy()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=60_000)
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    cache = open_cache(args.cache)
    matches = json.loads((args.cache / "matches.json").read_text())
    rng = np.random.default_rng(0)
    masks = split_masks(cache, matches)
    idxs = {}
    for name in SPLITS:
        idx = np.flatnonzero(masks[name])
        if len(idx) > args.max_rows:
            idx = np.sort(rng.choice(idx, args.max_rows, replace=False))
        idxs[name] = idx
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for s, idx in idxs.items():
        ts = np.asarray(cache["team_state"][idx])
        rows.update({f"{s}/idx": idx, f"{s}/lead_pts": (ts[:, 0] - ts[:, 1]) * 100.0,
                     f"{s}/final_pts": np.asarray(cache["final"][idx]) * 100.0,
                     f"{s}/result": np.asarray(cache["result"][idx]), f"{s}/elapsed_s": np.asarray(cache["elapsed_s"][idx]),
                     f"{s}/match": np.asarray(cache["match"][idx])})
    np.savez(args.out / "rows.npz", **rows)
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        net = load_value_net(path, device)
        out = {}
        for s, idx in idxs.items():
            p = predict(net, cache, idx, device)
            out[f"{s}/p"], out[f"{s}/final_pts"] = p["p"], p["final_pts"]
        np.savez(args.out / f"{name}.npz", **out)
        print(f"{name}: predicted {', '.join(f'{s} {len(i)}' for s, i in idxs.items())}", flush=True)


if __name__ == "__main__":
    main()
