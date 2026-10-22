"""Per-phase profile of PPO iterations at the r1 config (reports/rl/ppo_perf.md).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python examples/ppo_profile.py --iters 3 [--hash] [--save-batch b.npz] [--set k=v ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import torch

from blackout_env.rl import grpo
from blackout_env.rl.grpo import GRPOTrainer
from blackout_env.rl.ppo import PPOConfig, PPOTrainer
from blackout_env.sim import batch as sb

F = "checkpoints/distill/frozen"
OPPS = [f"net:{F}/{p}.pt" for p in ("p24c_v321", "p22_v31", "p21e_v30", "p20_v30", "p17_v28", "p14_v25", "d6_v42")] + ["V32.1", "mixture"]
W = [0.5] + [1.0] * 8
T = defaultdict(float)


def sync():
    torch.cuda.synchronize()


def wrap(obj, name, attr):
    f = getattr(obj, attr)

    def w(*a, **k):
        sync(); t0 = time.perf_counter()
        r = f(*a, **k)
        sync(); T[name] += time.perf_counter() - t0
        return r
    setattr(obj, attr, w)


def digest(batch):
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()[:16]


def phash(m):
    return hashlib.sha256(b"".join(p.detach().float().cpu().numpy().tobytes() for p in m.parameters())).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--segment", type=int, default=128)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--save-batch", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = PPOConfig(opponents=OPPS, opp_weights=W, pfsp="hard", games=a.games, segment=a.segment, seed=a.seed)
    for it in a.set:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    tr = PPOTrainer(cfg)
    wrap(tr, "values_np", "_values_np")
    for i, o in enumerate(tr.opps):
        wrap(o, f"pol:opp{i}", "moves")
    om = grpo.LearnerPolicy.moves
    def lm(self, *args, **kw):
        sync(); t0 = time.perf_counter()
        r = om(self, *args, **kw)
        T["pol:learner"] += time.perf_counter() - t0
        return r
    grpo.LearnerPolicy.moves = lm
    wrap(tr, "rollout", "_rollout")
    orig_gu = GRPOTrainer.update
    def gu(self, batch):
        sync(); t0 = time.perf_counter()
        r = orig_gu(self, batch)
        sync(); T["update_policy"] += time.perf_counter() - t0
        return r
    GRPOTrainer.update = gu
    orig_run = sb.BatchRollout.run
    roll = []
    def run(self, *args, **kw):
        r = orig_run(self, *args, **kw)
        roll.append(dict(self.timing, steps=int(r.steps.sum())))
        return r
    sb.BatchRollout.run = run
    for it in range(a.iters):
        T.clear(); roll.clear()
        torch.manual_seed(1000 + it)
        sync(); t0 = time.perf_counter()
        batch, cs = tr.collect()
        sync(); t1 = time.perf_counter()
        if a.save_batch and it == 0:
            np.savez(a.save_batch, **batch)
        if not a.no_update:
            us = tr.update(batch)
        else:
            us = {}
        sync(); t2 = time.perf_counter()
        tr.iter += 1
        rec = {"iter": it, "trunk_s": cs["trunk_s"], "leaf_s": cs["leaf_s"], "update_s": t2 - t1, "total_s": t2 - t0,
               "views": cs["views"], "decisions": cs["decisions"],
               "phase": {k: round(v, 2) for k, v in sorted(T.items())},
               "rollouts": [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in roll]}
        if a.hash:
            rec["hash"] = digest(batch)
            rec["loss"] = us.get("loss"); rec["vf_loss"] = us.get("vf_loss")
            rec["pol_hash"] = phash(tr.net); rec["critic_hash"] = phash(tr.critic)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
