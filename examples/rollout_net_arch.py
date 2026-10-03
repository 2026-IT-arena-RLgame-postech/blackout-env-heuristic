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
        with FlopCounterMode(display=False) as fc:  # not inference_mode: the MHA fast path hides from the counter
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


def cmd_table(args):
    """One row per variant: params, MAC/view, rollout throughput (2080 Ti, full BatchRollout loop, N=2048,
    fp32) and its ratio to r5 = "search-equivalent" branches at a fixed search budget (B8 x ratio), held-out
    hit / heading error (P11 val, sim DAgger val, own self-play states), sim strength paired vs P11 and vs r5."""
    import math

    sys.path.insert(0, str(REPO / "examples"))
    from rollout_net_strength import summarize

    root = Path(args.root)
    macs = json.load(open(root / "macs.json"))
    thr = {}
    for f in (root / "throughput").glob("throughput_*.json"):
        for r in json.load(open(f)):
            if r["N"] == 2048:
                thr[(f.stem.removeprefix("throughput_"), r["precision"])] = r["game_decisions_per_s"]
    rows = []
    names = args.variants or sorted({p.stem.removesuffix("_full") for p in (root / "heldout").glob("*_full.json")})
    for v in names:
        row = {"variant": v}
        key = "base" if v in ("r5",) else v
        if key in macs:
            row.update(params_m=macs[key]["params"] / 1e6, mmac=macs[key]["mmac"])
        t = thr.get((key, "fp32"))
        if t:
            row["thr_fp32"] = t
            row["thr_fp16"] = thr.get((key, "fp16"))
            row["cost_x"] = thr[("base", "fp32")] / t
            row["search_B_eq"] = 8 * t / thr[("base", "fp32")]
        hf = root / "heldout" / f"{v}_full.json"
        if hf.exists():
            d = next(iter(json.load(open(hf)).values()))
            for k, x in d.items():
                if isinstance(x, dict):
                    tag = "val" if k == "val" else ("sim" if k.startswith("simval") else "self")
                    for m in ("hit", "err", "w2", "hit_open", "err_open", "w2_open", "hit_late", "err_late"):
                        row[f"{tag}_{m}"] = x.get(m)
        gj = root / "strength" / v / "games.jsonl"
        if gj.exists():
            games = [json.loads(x) for x in gj.read_text().splitlines()]
            cand = [c for c in {g["candidate"] for g in games} if c not in ("p11", "r5")] or ["r5"]
            for ref in ("p11", "r5"):
                for r in summarize(games, ref):
                    if r["candidate"] == cand[0] and "dmargin" in r:
                        row[f"d_{ref}_{r['opponent']}"] = (r["dmargin"], r["dmargin_se"], r["n"])
                    if r["candidate"] == cand[0]:
                        row[f"race_{r['opponent']}"] = r["race"]
        rows.append(row)
    json.dump(rows, open(root / "table.json", "w"), indent=1)
    f = lambda x, fmt: (fmt % x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else "-"  # noqa: E731
    lines = ["| 변형 | 파라미터 | MMAC/뷰 | 처리량 fp32 (×r5 비용) | 탐색 등가 B | 보류 hit/오차 (첫 10초) | sim DAgger hit/오차 | 자기 상태 hit/오차 (30초+) | "
             "Δ P11: V19 / P5a / V23 | Δ r5: V19 / P5a / V23 |", "|" + "---|" * 10]
    for r in rows:
        dp = " / ".join(f"{r[k][0]:+.1f}±{r[k][1]:.1f}" if k in r else "-" for k in ("d_p11_v19", "d_p11_p5a", "d_p11_v23"))
        dr = " / ".join(f"{r[k][0]:+.1f}±{r[k][1]:.1f}" if k in r else "-" for k in ("d_r5_v19", "d_r5_p5a", "d_r5_v23"))
        lines.append(f"| {r['variant']} | {f(r.get('params_m'), '%.2fM')} | {f(r.get('mmac'), '%.1f')} | "
                     f"{f(r.get('thr_fp32'), '%.0f')} ({f(r.get('cost_x'), '%.2f')}) | {f(r.get('search_B_eq'), '%.1f')} | "
                     f"{f(r.get('val_hit'), '%.3f')}/{f(r.get('val_err'), '%.1f')}° ({f(r.get('val_hit_open'), '%.3f')}/{f(r.get('val_err_open'), '%.1f')}°) | "
                     f"{f(r.get('sim_hit'), '%.3f')}/{f(r.get('sim_err'), '%.1f')}° | "
                     f"{f(r.get('self_hit'), '%.3f')}/{f(r.get('self_err'), '%.1f')}° ({f(r.get('self_hit_late'), '%.3f')}/{f(r.get('self_err_late'), '%.1f')}°) | {dp} | {dr} |")
    txt = "\n".join(lines)
    print(txt)
    (root / "table.md").write_text(txt + "\n")


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
    q = sub.add_parser("table")
    q.add_argument("--root", default="reports/rollout_net_arch")
    q.add_argument("--variants", nargs="*", default=[])
    args = p.parse_args()
    {"list": cmd_list, "init": cmd_init, "recipe": cmd_recipe, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
