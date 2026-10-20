"""Per-phase profile of one GRPO iteration at the production config (reports/rl/grpo_perf.md).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python examples/grpo_profile.py --iters 2 [--hash] [--cprofile]
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
import time
from collections import defaultdict

import numpy as np
import torch

from blackout_env.rl import grpo
from blackout_env.rl.grpo import GRPOConfig, GRPOTrainer
from blackout_env.sim import batch as sb

F = "checkpoints/distill/frozen"
OPPS = [f"net:{F}/{p}.pt" for p in ("p24c_v321", "p22_v31", "p21e_v30", "p20_v30", "p17_v28", "p14_v25", "d6_v42")] + ["V32.1", "mixture"]
W = [0.5] + [1.0] * 8

PHASE = ["trunk"]
T = defaultdict(float)
N = defaultdict(int)


def timed(obj, name):
    f = obj.moves

    def w(bsim, idx, teams, inputs):
        t0 = time.perf_counter()
        r = f(bsim, idx, teams, inputs)
        T[PHASE[0] + ":" + name] += time.perf_counter() - t0
        N[PHASE[0] + ":" + name] += len(idx) * len(teams)
        return r
    obj.moves = w


def digest(batch, stats):
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--groups", type=int, default=48)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sample-steps", type=int, default=75)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--opps", default="", help="comma list of indices into the 9-opponent pool (default all)")
    ap.add_argument("--det", action="store_true", help="deterministic torch kernels (for bitwise A/B comparisons)")
    ap.add_argument("--save-batch", default="", help="np.savez the first collected batch here (for update benchmarks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    sel = [int(x) for x in a.opps.split(",")] if a.opps else list(range(len(OPPS)))
    cfg = GRPOConfig(opponents=[OPPS[i] for i in sel], opp_weights=[W[i] for i in sel], pfsp="hard", sample_steps=a.sample_steps, groups=a.groups,
                     k=a.k, drop_flat=0.005, seed=a.seed)
    for it in a.set:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    tr = GRPOTrainer(cfg)
    orig_r = tr._rollout

    def rr(roots, learner, opp_of, side, horizon, *args, **kw):
        PHASE[0] = "branch" if kw.get("opps") is not None else "trunk"
        return orig_r(roots, learner, opp_of, side, horizon, *args, **kw)
    tr._rollout = rr
    for i, p in enumerate(tr.opps):
        timed(p, f"trunk_opp{i}")
    for i, p in enumerate(tr.branch_opps):
        if p is not tr.opps[i]:
            timed(p, f"branch_opp{i}")
        else:
            pass
    # learner timing: wrap LearnerPolicy.moves at class level
    orig = grpo.LearnerPolicy.moves

    def lm(self, bsim, idx, teams, inputs):
        t0 = time.perf_counter()
        r = orig(self, bsim, idx, teams, inputs)
        T[PHASE[0] + ":learner"] += time.perf_counter() - t0
        N[PHASE[0] + ":learner"] += len(idx) * len(teams)
        return r
    grpo.LearnerPolicy.moves = lm
    orig_run = sb.BatchRollout.run
    roll = []

    def run(self, *args, **kw):
        r = orig_run(self, *args, **kw)
        roll.append(dict(self.timing, steps=int(r.steps.sum())))
        return r
    sb.BatchRollout.run = run
    out = []
    for it in range(a.iters):
        T.clear(); N.clear(); roll.clear()
        torch.manual_seed(1000 + it)   # update()'s randperm / dropout-free: reproducible A/B
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pr = cProfile.Profile() if (a.cprofile and it == a.iters - 1) else None
        if pr:
            pr.enable()
        batch, cs = tr.collect()
        t1 = time.perf_counter()
        if a.save_batch and it == 0:
            np.savez(a.save_batch, **batch)
        if not a.no_update:
            us = tr.update(batch)
            torch.cuda.synchronize()
        else:
            us = {}
        t2 = time.perf_counter()
        if pr:
            pr.disable()
            pstats.Stats(pr).sort_stats("cumulative").print_stats(45)
            pstats.Stats(pr).sort_stats("tottime").print_stats(25)
        tr.iter += 1
        rec = {"iter": it, "collect_s": t1 - t0, "update_s": t2 - t1, "total_s": t2 - t0,
               "trunk_s": cs["trunk_s"], "branch_s": cs["branch_s"], "leaf_s": cs["leaf_s"],
               "decisions": cs["decisions"], "views": cs["views"],
               "dec_per_s": cs["decisions"] / (t2 - t0),
               "rollouts": [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in roll],
               "pol_time": {k: round(v, 2) for k, v in sorted(T.items())}, "pol_rows": dict(N)}
        if a.hash:
            rec["hash"] = digest(batch, cs)
            rec["R_mean"] = cs["R_mean"]; rec["kl"] = us.get("kl"); rec["loss"] = us.get("loss")
            rec["upd_hash"] = hashlib.sha256(b"".join(p.detach().float().cpu().numpy().tobytes()
                                                      for p in tr.net.parameters())).hexdigest()[:16]
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
