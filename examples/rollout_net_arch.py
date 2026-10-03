"""Rollout-net architecture experiments (docs/policy_v2_design_ko.md §22 "롤아웃 망 구조 실험"): the variant
table, random-init checkpoints for throughput benchmarks, and the two-stage r5 recipe per variant.

    PYTHONPATH=. python examples/rollout_net_arch.py list
    PYTHONPATH=. python examples/rollout_net_arch.py init --out checkpoints/rollout_net_arch/init     # one .pt per variant
    PYTHONPATH=. python examples/rollout_net_arch.py recipe --variant patch_x32 --data D --extra E --out checkpoints/rollout_net_arch
        # prints the two commands of r5's recipe (stage 1: 30k steps P11 shards lr 2e-3 KL 1.0;
        # stage 2: 20k steps from stage 1, + 50 % sim DAgger rows, lr 1e-3, KL 0.5)

Variant names: ``base`` = r5's architecture; ``<tokens>_<attn>[<map_dim>]`` with tokens f3 / s3 / patch and
attn self (one joint sequence) / x (cross, K/V-only map tokens) / l3, l5 (local windows) / fov (fovea); a
``d256_`` prefix = width 256 (FFN 512, head hidden 256).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VARIANTS = {
    "base": {},
    "f3_self": {"map_tokens": "f3"},
    "s3_self": {"map_tokens": "s3"},
    "patch_self": {"map_tokens": "patch"},
    "f3_x": {"map_tokens": "f3", "map_attn": "cross"},
    "f3_x32": {"map_tokens": "f3", "map_attn": "cross", "map_dim": 32},
    "s3_x32": {"map_tokens": "s3", "map_attn": "cross", "map_dim": 32},
    "patch_x": {"map_tokens": "patch", "map_attn": "cross"},
    "patch_x32": {"map_tokens": "patch", "map_attn": "cross", "map_dim": 32},
    "patch_l3": {"map_tokens": "patch", "map_attn": "local3", "map_dim": 32},
    "patch_l5": {"map_tokens": "patch", "map_attn": "local5", "map_dim": 32},
    "patch_fov": {"map_tokens": "patch", "map_attn": "fovea", "map_dim": 32},
    "s3_l3": {"map_tokens": "s3", "map_attn": "local3", "map_dim": 32},
    "s3_fov": {"map_tokens": "s3", "map_attn": "fovea", "map_dim": 32},
}
for _k in list(VARIANTS):
    VARIANTS["d256_" + _k] = {**VARIANTS[_k], "d": 256}


def config(name: str) -> dict:
    return {"arch": "attn", **VARIANTS[name]}


def cmd_list(args):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from blackout_env.sim.rollout_net import RolloutNet

    g = torch.rand(1, 13, 24, 24)
    t = torch.rand(1, 4)
    a = torch.rand(1, 10, 12) * 2 - 1
    a[..., 9:12] = torch.eye(3)[torch.randint(0, 3, (1, 10))]
    rows = {}
    for name in VARIANTS:
        net = RolloutNet(**config(name)).eval()
        with FlopCounterMode(display=False) as fc, torch.inference_mode():
            net(g, t, a)
        rows[name] = {"params": sum(p.numel() for p in net.parameters()), "mmac": fc.get_total_flops() / 2e6}
    b = rows["base"]["mmac"]
    for k, r in rows.items():
        print(f"{k:16s} params {r['params'] / 1e6:5.2f}M  MMAC/view {r['mmac']:6.1f}  x{r['mmac'] / b:.2f}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


def cmd_init(args):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.variants or VARIANTS:
        torch.manual_seed(0)
        net = RolloutNet(**config(name))
        torch.save({"config": net.config, "state": net.state_dict()}, out / f"{name}.pt")
        print(out / f"{name}.pt")


def cmd_recipe(args):
    cfg = json.dumps(config(args.variant))
    out = Path(args.out)
    s1, s2 = out / f"{args.variant}_s1.pt", out / f"{args.variant}.pt"
    tr = "examples/rollout_net_train.py train"
    print(f"{tr} --data {args.data} --out {s1} --config '{cfg}' --steps 30000 --lr 2e-3 --p11-kl 1.0 --amp {args.amp}")
    print(f"{tr} --data {args.data} --out {s2} --init {s1} --steps 20000 --lr 1e-3 --p11-kl 0.5 --extra {args.extra} "
          f"--extra-frac 0.5 --amp {args.amp}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("list")
    q.add_argument("--json", default="")
    q = sub.add_parser("init")
    q.add_argument("--out", default="checkpoints/rollout_net_arch/init")
    q.add_argument("--variants", nargs="*", default=[])
    q = sub.add_parser("recipe")
    q.add_argument("--variant", required=True)
    q.add_argument("--data", default="checkpoints/rollout_net/data")
    q.add_argument("--extra", default="checkpoints/rollout_net/simdata/train_ex23.npz")
    q.add_argument("--out", default="checkpoints/rollout_net_arch")
    q.add_argument("--amp", default="fp16")
    args = p.parse_args()
    {"list": cmd_list, "init": cmd_init, "recipe": cmd_recipe}[args.cmd](args)


if __name__ == "__main__":
    main()
