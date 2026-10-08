"""S4 search cost: roots searched per second for each search variant, same roots and batch.
(docs/policy_v2_design_ko.md §26 "S4 더 깊고 넓은 탐색".)

Roots = mid-match states (r5 x r5 from SimState.from_seed, ``--at-s`` seconds in).  Each variant
searches the same R roots (R = --roots, a list: 128 and 256 roots x 8 branches = the S3
default's 1024 / 2048 rollout games) ``--reps`` times after a warm-up; roots/s = R / median wall.

    PYTHONPATH=. python examples/s4_search_bench.py --roots 128,256 --out reports/s4_deep/bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

R5 = "checkpoints/rollout_net/r5.pt"
VALUE = "checkpoints/value/frozen/s2v_iter5.pt"
B8 = "alt*,samp:1.0x2"
VARIANTS = {  # tag: Searcher kwargs (branches as a spec)
    "s8e25m3": {},
    "w13r1": {"branches": "alt*,alt*:1,samp:1.0x2"},
    "w18pair": {"branches": "alt*,pair*,samp:1.0x2"},
    "w12s": {"branches": "alt*,samp:1.0x6"},
    "w13stop": {"branches": "alt*,samp:1.0x2,stop*"},
    "w50": {"window": 50},
    "t2full": {"children": "alt*"},
    "t2bf": {"children": B8, "expand": 2},
    "o4min": {"opp_branches": "samp:1.0x3"},
    "o4full": {"opp_branches": "samp:1.0x3", "opp_left": 150},
    "l200": {"leaf_decisions": 200},
    "l250": {"leaf_decisions": 250},
}


def roots_at(n: int, at_s: float, device: str):
    from blackout_env.sim import SimState
    from blackout_env.sim.batch import BatchRollout, NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet

    pol = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=device)
    res = BatchRollout(pol, pol).run([SimState.from_seed(3_000_000 + k) for k in range(n)], int(at_s * 25))
    keep = [b for b in range(n) if not res.over[b]]
    return [res.bsim.state(b) for b in keep]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", default="128,256")
    p.add_argument("--at-s", type=float, default=20.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(REPO / "reports/s4_deep/bench.json"))
    args = p.parse_args()
    import torch

    from blackout_env.sim.batch import NetPolicy
    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import ModNetPolicy, Searcher, ValueNetLeaf, parse_branches

    torch.set_num_threads(4)
    sizes = [int(x) for x in args.roots.split(",")]
    pool = roots_at(max(sizes) + 16, args.at_s, args.device)
    net = NetPolicy(RolloutNet.from_checkpoint(str(REPO / R5)), device=args.device)
    leaf = ValueNetLeaf(str(REPO / VALUE), args.device)
    out = {"device": torch.cuda.get_device_name() if args.device == "cuda" else args.device, "at_s": args.at_s, "rows": []}
    for tag in args.variants.split(","):
        kw = dict(VARIANTS[tag])
        br = parse_branches(kw.pop("branches", B8))
        for k in ("opp_branches", "children"):
            if k in kw:
                kw[k] = parse_branches(kw[k])
        s = Searcher(ModNetPolicy(net), leaf, br, margin=3.0, packed=True, **kw)
        s.search(pool[:8], [0] * 8, list(range(8)))  # warm-up
        for r in sizes:
            roots = pool[:r]
            teams = [k % 2 for k in range(r)]
            walls, gd = [], 0
            for rep in range(args.reps):
                t0 = time.perf_counter()
                o = s.search(roots, teams, [rep * 1000 + k for k in range(r)])
                walls.append(time.perf_counter() - t0)
                gd = o.timing["game_decisions"]
            w = float(np.median(walls))
            row = {"tag": tag, "roots": r, "first_stage_games": r * len(br), "wall_s": w, "roots_per_s": r / w,
                   "game_decisions_per_root": gd / r, "game_decisions_per_s": gd / w}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    ref = {r["roots"]: r["roots_per_s"] for r in out["rows"] if r["tag"] == "s8e25m3"}
    for r in out["rows"]:
        if r["roots"] in ref:
            r["cost_x"] = ref[r["roots"]] / r["roots_per_s"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        print(f"{r['tag']:10s} R={r['roots']:4d} {r['roots_per_s']:7.1f} roots/s  x{r.get('cost_x', float('nan')):.2f}  "
              f"{r['game_decisions_per_root']:6.0f} game-dec/root  {r['game_decisions_per_s'] / 1e3:5.1f}k game-dec/s")


if __name__ == "__main__":
    main()
