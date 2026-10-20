"""Chunked full-match eval of one candidate vs one opponent; saves per-game margins (seed-ordered).
Same settings as grpo.eval_matches (greedy, both sides, horizon 11000)."""
import argparse, json, math, time
import numpy as np
from blackout_env.sim.batch import BatchRollout, SimState, NetPolicy
from blackout_env.rl.grpo import make_opponent_policy

ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True); ap.add_argument("--opp", required=True)
ap.add_argument("--lo", type=int, required=True); ap.add_argument("--hi", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
t0 = time.time()
pol = NetPolicy.from_checkpoint(a.cand, device=a.device)
opp = make_opponent_policy(a.opp, a.device)
seeds = list(range(a.lo, a.hi))
roots = [SimState.from_seed(s) for s in seeds]
out = {"cand": a.cand, "opp": a.opp, "seeds": seeds}
for side in (0, 1):
    pa, pb = (pol, opp) if side == 0 else (opp, pol)
    res = BatchRollout(pa, pb, packed=True).run(roots, 11000)
    out[f"margin_side{side}"] = [float(x) for x in res.lead(side)]
    print("side", side, "done", time.time() - t0, flush=True)
out["secs"] = time.time() - t0
json.dump(out, open(a.out, "w"))
