"""GRPO / PPO fine-tuning on the simulator (docs/rl_finetune_plan_ko.md).

    PYTHONPATH=. .venv/bin/python examples/rl_train.py --algo grpo --out checkpoints/rl/grpo_a \
        --hours 6 --eval-every 10 --set groups=64 k=8

``--set key=value`` overrides any GRPOConfig / PPOConfig field (lists: comma separated, or ';' separated
when an item contains commas).  Logs one JSON
line per iteration to <out>/log.jsonl (eval rows have "eval": {...}); checkpoints <out>/iter_<n>.pt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from blackout_env.rl.grpo import GRPOConfig, GRPOTrainer, eval_matches
from blackout_env.rl.ppo import PPOConfig, PPOTrainer

DEFAULT_OPPS = ",".join(f"net:checkpoints/distill/frozen/{p}.pt" for p in
                        ("p24c_v321", "p22_v31", "p21e_v30", "p20_v30", "p17_v28", "p14_v25"))
# p24c / p22 are in the default pool; p5a / p16 are held out of every pool (generalisation check)
DEFAULT_EVAL = ",".join(f"net:checkpoints/distill/frozen/{p}.pt" for p in ("p24c_v321", "p22_v31", "p5a_v22", "p16_v27"))


def parse_set(cfg, items):
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    for it in items:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, list) or (cur is None and k.endswith("weights")):
            sep = ";" if ";" in v else ","   # ';' when an item itself has commas (act:V32,adapt_k=3)
            val = [type(cur[0])(x) if cur else float(x) for x in v.split(sep)] if v else []
        elif isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        else:
            val = type(cur)(v) if cur is not None else v
        assert k in fields, k
        setattr(cfg, k, val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=("grpo", "ppo"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--iters", type=int, default=10**9)
    ap.add_argument("--max-decisions", type=float, default=float("inf"),
                    help="stop after this many simulator decision steps (equal-experience budget across algorithms)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=32, help="seeds per opponent (x2 sides)")
    ap.add_argument("--eval-opps", default=DEFAULT_EVAL)
    ap.add_argument("--save-every", type=int, default=0,
                    help="ppo: write <out>/latest.pt (full resumable trainer state, atomic) every N iterations (0: off)")
    ap.add_argument("--resume", action="store_true", help="ppo: continue from <out>/latest.pt if it exists (same --set flags!)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = PPOConfig() if args.algo == "ppo" else GRPOConfig()
    cfg.opponents = DEFAULT_OPPS.split(",")
    parse_set(cfg, args.set)
    if args.algo == "ppo" and not cfg.league_dir:
        cfg.league_dir = args.out
    tr = (PPOTrainer if args.algo == "ppo" else GRPOTrainer)(cfg)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    log.write(json.dumps({"config": dataclasses.asdict(cfg), "algo": args.algo}) + "\n")
    eval_opps = args.eval_opps.split(",")
    seeds = range(900_000, 900_000 + args.eval_seeds)  # fixed eval seeds, disjoint from training draws (random 31-bit)

    def evaluate():
        out = {}
        for spec in eval_opps:
            wr, m, se = eval_matches(tr.net, spec, seeds, cfg.device)
            out[os.path.basename(spec).replace(".pt", "")] = {"win": wr, "margin": m, "se": se}
        return out

    t_start = time.time()
    decisions = 0
    latest = os.path.join(args.out, "latest.pt")

    def save_latest():
        tr.save_resume(latest, {"decisions": decisions, "wall_h": (time.time() - t_start) / 3600})

    if args.resume and os.path.exists(latest):
        ex = tr.load_resume(latest)
        decisions, t_start = int(ex["decisions"]), time.time() - ex["wall_h"] * 3600
        log.write(json.dumps({"resumed": tr.iter, "decisions_total": decisions, "wall_h": ex["wall_h"]}) + "\n"); log.flush()
        print("resumed at iter", tr.iter, "decisions", decisions, flush=True)
    else:
        ev = evaluate()
        log.write(json.dumps({"iter": 0, "eval": ev, "wall_h": 0.0}) + "\n"); log.flush()
        print("eval 0", ev, flush=True)
    while (tr.iter < args.iters and time.time() - t_start < args.hours * 3600
           and decisions < args.max_decisions):
        s = tr.step()
        decisions += s.get("decisions", 0)
        s["wall_h"] = (time.time() - t_start) / 3600
        s["decisions_total"] = decisions
        if tr.iter % args.eval_every == 0:
            tr.save(os.path.join(args.out, f"iter_{tr.iter}.pt"))
            s["eval"] = evaluate()
            print("eval", tr.iter, s["eval"], flush=True)
        log.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}) + "\n")
        log.flush()
        if args.save_every and args.algo == "ppo" and tr.iter % args.save_every == 0:
            save_latest()
    tr.save(os.path.join(args.out, "final.pt"))
    if args.save_every and args.algo == "ppo":
        save_latest()
    s = {"iter": tr.iter, "eval": evaluate(), "wall_h": (time.time() - t_start) / 3600, "final": True}
    log.write(json.dumps(s) + "\n"); log.close()
    print("final", s, flush=True)


if __name__ == "__main__":
    main()
