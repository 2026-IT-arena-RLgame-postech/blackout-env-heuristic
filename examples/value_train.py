"""S1 value net V(s): build the row cache, fit baselines, train, evaluate (docs/policy_v2_design_ko.md "S1 가치망").

    python examples/value_train.py build --cache datasets/value_s1            # index + row cache
    python examples/value_train.py baselines --cache datasets/value_s1 --out reports/value_s1
    python examples/value_train.py train --cache datasets/value_s1 --out-dir checkpoints/value/s1 \
        --init checkpoints/distill/frozen/p5a_v22.pt --steps 40000 --device mps --amp fp16 --compile
    python examples/value_train.py eval --cache datasets/value_s1 --ckpt checkpoints/value/s1/best.pt \
        --out reports/value_s1

Targets, splits and the cache layout: blackout_env/value/{targets,data}.py.  Metrics:
blackout_env/value/metrics.py (R²/MAE of the lead change per horizon and time bucket, Brier /
accuracy / calibration of the win probability).
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

from blackout_env.value.data import SPLITS, build_cache, build_index, open_cache  # noqa: E402


def cmd_build(args) -> None:
    started = time.time()
    index = build_index(REPO, workers=args.workers)
    print(f"index: {len(index)} shards, {sum(e['rows'] or 0 for e in index)} rows "
          f"({time.time() - started:.0f} s)", flush=True)
    rows = build_cache(REPO, index, args.cache, rows_per_match=args.rows_per_match, workers=args.workers,
                       log=lambda msg: print(msg, flush=True))
    split = open_cache(args.cache)["split"]
    counts = {name: int((split == i).sum()) for i, name in enumerate(SPLITS)}
    print(f"cache: {rows} rows {counts} ({time.time() - started:.0f} s)", flush=True)


def cmd_baselines(args) -> None:
    from blackout_env.value.baselines import run_baselines

    args.out.mkdir(parents=True, exist_ok=True)
    report = run_baselines(open_cache(args.cache), json.loads((args.cache / "matches.json").read_text()),
                           save_path=args.out / "baseline_models.pt")
    (args.out / "baselines.json").write_text(json.dumps(report, indent=1))
    from blackout_env.value.metrics import format_report

    print(format_report(report))


def cmd_train(args) -> None:
    from blackout_env.value.train import train

    train(args)


def cmd_eval(args) -> None:
    from blackout_env.value.metrics import format_report
    from blackout_env.value.train import evaluate_checkpoint

    report = evaluate_checkpoint(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=1))
    print(format_report(report))


def cmd_pairwise(args) -> None:
    """Concordance on same-start pairs (value/pairwise.py) for the value net and the baselines."""
    import torch

    from blackout_env.value.baselines import SETS, load_models, predict_set
    from blackout_env.value.features import hand_features
    from blackout_env.value.model import load_value_net
    from blackout_env.value.pairwise import aligned_rows, concordance
    from blackout_env.value.targets import HORIZONS_S
    from blackout_env.value.train import predict_rows

    matches = json.loads((args.cache / "matches.json").read_text())
    entries = [m for m in matches if m["split"] in args.splits and m["path"].startswith("reports/")]
    rows = aligned_rows(REPO, entries)
    print(f"pairwise: {len(entries)} matches, {len(rows['match'])} aligned rows", flush=True)
    preds = {}
    from blackout_env.train.accel import pick_device

    device = pick_device(args.device)
    net = load_value_net(args.ckpt, device)
    p = predict_rows(net, rows, np.arange(len(rows["match"])), device)
    h20 = HORIZONS_S.index(20)
    preds["value_net"] = {"final": p["final"], "p_win": p["p_win"], "delta20": p["delta"][:, h20]}
    models = load_models(args.baselines)
    x = hand_features(rows["graphic"], rows["team_state"], rows["agent_states"])
    for name in list(SETS) + ["hand_mlp"]:
        b = predict_set(name, models, x)
        preds[name] = {"final": b["final"], "p_win": b["p_win"], "delta20": b["delta"][:, h20]}
    report = {name: concordance(rows, pred, entries) for name, pred in preds.items()}
    from blackout_env.value.pairwise import smoothness

    rng = np.random.default_rng(0)
    pick = [entries[i] for i in rng.choice(len(entries), size=min(args.smooth_matches, len(entries)), replace=False)]

    def net_predict(r):
        return predict_rows(net, r, np.arange(len(r["team_state"])), device)

    def mlp_predict(r):
        return predict_set("hand_mlp", models, hand_features(r["graphic"], r["team_state"], r["agent_states"]))

    smooth = {"value_net": smoothness(net_predict, REPO, pick), "hand_mlp": smoothness(mlp_predict, REPO, pick)}
    print("smoothness (points):", json.dumps(smooth), flush=True)
    report["smoothness"] = smooth
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"pairwise_{args.name}.json").write_text(json.dumps(report, indent=1))
    keys = [k for k in report["value_net"] if "|t" not in k]
    names = [n for n in report if n != "smoothness"]
    print(f"{'target|pred|scope':<34}" + "".join(f"{n:>13}" for n in names) + "   pairs")
    for k in keys:
        print(f"{k:<34}" + "".join(f"{report[n].get(k, {}).get('concordance', float('nan')):13.3f}" for n in names)
              + f"   {report['value_net'][k]['pairs']}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", type=Path, default=REPO / "datasets/value_s1")
    b.add_argument("--rows-per-match", type=int, default=32)
    b.add_argument("--workers", type=int, default=8)
    b.set_defaults(fn=cmd_build)

    bl = sub.add_parser("baselines")
    bl.add_argument("--cache", type=Path, default=REPO / "datasets/value_s1")
    bl.add_argument("--out", type=Path, default=REPO / "reports/value_s1")
    bl.set_defaults(fn=cmd_baselines)

    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, default=REPO / "datasets/value_s1")
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--init", type=Path, default=None, help="MyModel checkpoint (policy_state) to warm-start the trunk")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--trunk-lr-scale", type=float, default=1.0, help="trunk lr = lr x this (warm start)")
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--device", default="auto", help="auto = cuda > mps > cpu (torchrun: this rank's GPU)")
    t.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--eval-rows", type=int, default=40_000, help="held-out rows scored at each eval")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--num-threads", type=int, default=4)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--cache", type=Path, default=REPO / "datasets/value_s1")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--out", type=Path, default=REPO / "reports/value_s1")
    e.add_argument("--name", default="value_net")
    e.add_argument("--device", default="auto")
    e.add_argument("--batch-size", type=int, default=1024)
    e.add_argument("--max-rows", type=int, default=None, help="per split (None = all)")
    e.set_defaults(fn=cmd_eval)

    pw = sub.add_parser("pairwise")
    pw.add_argument("--cache", type=Path, default=REPO / "datasets/value_s1")
    pw.add_argument("--ckpt", type=Path, required=True)
    pw.add_argument("--baselines", type=Path, default=REPO / "reports/value_s1/baseline_models.pt")
    pw.add_argument("--splits", nargs="+", default=["unseen", "heldout"])
    pw.add_argument("--out", type=Path, default=REPO / "reports/value_s1")
    pw.add_argument("--name", default="value_net")
    pw.add_argument("--device", default="auto")
    pw.add_argument("--smooth-matches", type=int, default=150)
    pw.set_defaults(fn=cmd_pairwise)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
