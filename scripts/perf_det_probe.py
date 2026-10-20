"""Which part of a GRPO collect() is non-reproducible run to run?  Two collects per variant, compare final sim states."""
import sys
import numpy as np, torch
from blackout_env.rl.grpo import GRPOConfig, GRPOTrainer

F = "checkpoints/distill/frozen"
ALL = [f"net:{F}/{p}.pt" for p in ("p24c_v321", "p22_v31", "p21e_v30", "p20_v30", "p17_v28", "p14_v25", "d6_v42")]


def final_state(**kw):
    opps = kw.pop("opps")
    cfg = GRPOConfig(opponents=opps, groups=12, k=4, heur_workers=0, graphs=False, joint=False, **kw)
    tr = GRPOTrainer(cfg)
    orig = tr.leaf.__call__
    cap = {}

    class L:
        net = tr.leaf.net

        def __call__(s, bsim, teams):
            cap["uf"] = bsim.uf.copy(); return orig(bsim, teams)
    tr.leaf = L()
    tr.collect()
    return cap["uf"]


for j, o in enumerate(ALL):
    kw = dict(opps=[o], sample_steps=1, leaf_steps=60, trunk_step=20)
    a, b = final_state(**kw), final_state(**kw)
    print(o.split("/")[-1], "identical:", np.array_equal(a, b), flush=True)
