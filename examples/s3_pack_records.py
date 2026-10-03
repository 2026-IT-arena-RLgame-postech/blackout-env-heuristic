"""S3 expert iteration: search-labelled self-play rows (examples/s3_search_eval.py play --record) ->
examples/rollout_net_train.py's packed train.npz / val.npz.

Each recorded row is the searching team's canonical input at one decision and the move it executed
(the chosen branch's move, = r5's own move on the base branch).  Labels are made exactly like the
simulator DAgger collector (rollout_net_collect.label_rows: arc label nearest bin +- 1 at K = 32 +
the raw heading).  ``--branch-weight`` > 1 repeats rows executed under a non-base branch (simple
oversampling); the split is by match (seed, side), ``--val-frac`` of matches held out.  ``--r5-kl``
stores r5's own logits/offsets per row under the trainer's ``p11_q`` / ``p11_off`` keys, so
``train --p11-kl W`` regularises towards r5 instead of P11.

    PYTHONPATH=. .venv/bin/python examples/s3_pack_records.py --records 'reports/s3_search/records/*.npz' \\
        --out checkpoints/rollout_net/s3data
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--branch-weight", type=int, default=1)
    p.add_argument("--every", type=int, default=1, help="keep every n-th decision of a match")
    p.add_argument("--r5-kl", default="", help="rollout net checkpoint whose logits fill p11_q / p11_off")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    from rollout_net_collect import label_rows

    parts = [dict(np.load(f)) for f in sorted(glob.glob(args.records))]
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    keep = data["tick"] % args.every == 0
    data = {k: v[keep] for k, v in data.items()}
    n = len(data["tick"])
    labels = np.zeros((n, 5), np.uint64)
    angles = np.zeros((n, 5), np.float32)
    for i in range(n):
        labels[i], angles[i] = label_rows(data["moves"][i], bool(data["side"][i] == 1))
    match = data["seed"].astype(np.int64) * 2 + data["side"]
    ids = np.unique(match)
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, max(1, int(len(ids) * args.val_frac)), replace=False).tolist())
    is_val = np.array([m in val_ids for m in match])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"map": data["map"], "team_state": data["team_state"], "agent_states": data["agent_states"],
            "labels": labels, "angles": angles, "tick": data["tick"].astype(np.int32),
            "branch": data["branch"].astype(np.int16)}
    if args.r5_kl:
        rows.update(_net_targets(rows, args.r5_kl, args.device))
    for name, sel in (("train", ~is_val), ("val", is_val)):
        idx = np.nonzero(sel)[0]
        if name == "train" and args.branch_weight > 1:
            dev = idx[rows["branch"][idx] > 0]
            idx = np.sort(np.concatenate([idx] + [dev] * (args.branch_weight - 1)))
        np.savez(out / f"{name}.npz", **{k: v[idx] for k, v in rows.items()})
        print(name, len(idx), "rows", f"non-base {float((rows['branch'][idx] > 0).mean()):.2f}", flush=True)


def _net_targets(data, path, device, batch=4096):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from rollout_net_train import batch_tensors

    net = RolloutNet.from_checkpoint(path).to(device)
    n = len(data["tick"])
    q = np.zeros((n, 5, 32), np.float16)
    off = np.zeros((n, 5, 32), np.float16)
    with torch.inference_mode():
        for s in range(0, n, batch):
            idx = np.arange(s, min(n, s + batch))
            g, ts, ag = batch_tensors(data, idx, device, torch)
            a, b = net(g, ts, ag)
            q[idx], off[idx] = a.float().cpu().numpy(), b.float().cpu().numpy()
    return {"p11_q": q, "p11_off": off}


if __name__ == "__main__":
    main()
