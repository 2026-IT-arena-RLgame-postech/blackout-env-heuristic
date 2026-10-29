"""Wall time per PPO step, sync vs async_collect (reports/rl/ppo_perf2.md).

    CUDA_VISIBLE_DEVICES=2[,0] PYTHONPATH=. python examples/ppo_async_bench.py --iters 6 [--async] [--actor-device cuda:1] --set k=v ...
"""
import argparse, json, time

import torch

from blackout_env.rl.ppo import PPOConfig, PPOTrainer

def main():
    import faulthandler, sys
    faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)   # all threads' stacks every 4 min (hang diagnosis)
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--async", dest="async_", action="store_true")
    ap.add_argument("--actor-device", default="")
    ap.add_argument("--opp-specs", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    opps = a.opp_specs.split(";")
    cfg = PPOConfig(opponents=opps, opp_weights=[float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(opps),
                    pfsp="hard", async_collect=a.async_, actor_device=a.actor_device)
    for it in a.set:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    tr = PPOTrainer(cfg)
    for i in range(a.iters):
        t0 = time.perf_counter()
        s = tr.step()
        torch.cuda.synchronize()
        print(json.dumps({"iter": i, "wall_s": round(time.perf_counter() - t0, 2), "trunk_s": round(s["trunk_s"], 2),
                          "leaf_s": round(s["leaf_s"], 2), "update_s": round(s["update_s"], 2),
                          "actor_wait_s": round(s.get("actor_wait_s", float("nan")), 2), "kl": s.get("kl"), "clipfrac": s.get("clipfrac"),
                          "views": s["views"]}), flush=True)


if __name__ == "__main__":
    main()
