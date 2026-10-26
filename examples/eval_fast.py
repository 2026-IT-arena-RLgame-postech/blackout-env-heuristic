"""Batched full-match eval of one candidate vs several opponents: all (opponent, seed, side) games in ONE rollout.

    PYTHONPATH=. .venv/bin/python examples/eval_fast.py --cand checkpoints/rl/x/final.pt \
        --opps "net:checkpoints/distill/frozen/p22_v31.pt;net:checkpoints/distill/frozen/p24c_v321.pt;V32.1" \
        --lo 970000 --hi 970256 --out reports/rl/x_eval --tag x --chunk 128

Writes one json per opponent (and per chunk) ``<out>/<tag>__<oppname>__<chunk lo>.json`` in the format of
``eval_grpo_ckpt.py`` ({"cand","opp","seeds","margin_side0","margin_side1","secs"}; "secs" = wall time of the chunk's
whole batched rollout, shared by its opponents), so the paired-diff analysis reads them unchanged.
Defaults: fp16 + CUDA graphs on CUDA (margins may differ slightly from the fp32 eager eval_grpo_ckpt.py by argmax
flips; --fp32 --no-graphs reproduces it exactly).  Opponent specs are separated by ';' (descriptors contain commas).
"""
import argparse
import json
import os
import re
import time

from blackout_env.rl.evalfast import eval_batched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="checkpoint path")
    ap.add_argument("--opps", required=True, help="';'-separated specs: net:<ckpt> | heuristic alias / descriptor")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tag", default=None, help="file prefix (default: candidate checkpoint's basename)")
    ap.add_argument("--chunk", type=int, default=0, help="seeds per rollout (0: all of lo..hi at once); one json per chunk")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp32", action="store_true", help="disable fp16")
    ap.add_argument("--no-graphs", action="store_true", help="disable CUDA graphs")
    ap.add_argument("--heur-workers", type=int, default=0, help="heuristic opponents in this many worker processes")
    ap.add_argument("--joint", action="store_true", help="JointSeats: one forward per policy over both teams' views")
    ap.add_argument("--horizon", type=int, default=11000)
    ap.add_argument("--angle-quant", type=int, default=0, help="candidate's move angle: -1 bin centres (no offset), M>0 rounded to M directions")
    a = ap.parse_args()

    opps = [s for s in a.opps.split(";") if s]
    tag = a.tag or os.path.basename(a.cand).replace(".pt", "")
    os.makedirs(a.out, exist_ok=True)
    workers = None
    if a.heur_workers > 0 and any(not s.startswith("net:") for s in opps):
        from blackout_env.rl.hpool import HeuristicWorkers

        workers = HeuristicWorkers(a.heur_workers)   # one pool for all chunks
    step = a.chunk if a.chunk > 0 else a.hi - a.lo
    for lo in range(a.lo, a.hi, step):
        seeds = list(range(lo, min(lo + step, a.hi)))
        t0 = time.time()
        res = eval_batched(a.cand, opps, seeds, a.device, horizon=a.horizon, angle_quant=a.angle_quant, graphs=not a.no_graphs, fp16=not a.fp32,
                           joint=a.joint, workers=workers)
        secs = time.time() - t0
        for spec in opps:
            name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", os.path.basename(spec).replace(".pt", ""))
            out = {"cand": a.cand, "opp": spec, "seeds": seeds, "margin_side0": res[spec]["margin_side0"],
                   "margin_side1": res[spec]["margin_side1"], "secs": secs}
            json.dump(out, open(os.path.join(a.out, f"{tag}__{name}__{lo}.json"), "w"))
            print(f"{spec}  seeds {lo}..{seeds[-1]}  win {res[spec]['win']:.3f}  margin {res[spec]['margin']:+.1f}"
                  f" +- {res[spec]['se']:.1f}", flush=True)
        print(f"chunk {lo} done in {secs:.0f}s", flush=True)
    if workers is not None:
        workers.close()


if __name__ == "__main__":
    main()
