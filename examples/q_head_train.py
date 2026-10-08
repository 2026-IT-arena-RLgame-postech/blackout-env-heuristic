"""Q head for S3 search distillation (docs/policy_v2_design_ko.md §22 C): regress the search's branch values.

The S3 search (blackout_env/sim/search.py) scores, at every search root, B branches -- base (r5's arg max),
``alt u`` (unit u's best bin outside its top bin +- 1, for one 1 s closed-loop window), and sampled
branches -- by "1 s branch + 5 s r5 x r5 rollout + V_sym".  Instead of hard labels of the chosen branch
(r6, §20: deviations not learnable), a per-unit head on the rollout net's trunk regresses
adv_u = V(alt u) - V(base) (points), and acting = r5 + switch the best unit to its alternative when its
predicted adv > margin (search.QSearcher, ``s3_search_eval.py play --qhead``).

    # roots from: s3_search_eval.py play --record-roots ...  (reports/s3_qdata/*/roots/*.npz)
    PYTHONPATH=. python examples/q_head_train.py pack --roots 'reports/s3_qdata/*/roots/*.npz' --out checkpoints/rollout_net_arch/qdata
    PYTHONPATH=. python examples/q_head_train.py train --data checkpoints/rollout_net_arch/qdata --init checkpoints/rollout_net/r5.pt \
        --out checkpoints/rollout_net_arch/q1.pt --freeze-trunk
    PYTHONPATH=. python examples/q_head_train.py eval --data checkpoints/rollout_net_arch/qdata --net checkpoints/rollout_net_arch/q1.pt

``eval`` reports, on held-out matches, the correlation of predicted and recorded adv and the realized
(recorded) gain of acting on the prediction at several margins, next to the search's own choice
(selection on the same noisy values, an optimistic upper bound) and an always-base / random-unit baseline.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))
CLIP = 40.0  # points: targets are clipped (opening races swing +-80)


def cmd_pack(args):
    files = sorted(glob.glob(args.roots))
    parts = [dict(np.load(f)) for f in files]
    branches = json.loads(str(parts[0]["branches"]))
    names = [b["name"] for b in branches]
    assert names[:6] == ["base", "alt0", "alt1", "alt2", "alt3", "alt4"], names
    keys = ["map", "team_state", "agent_states", "values", "choice", "top_bin", "alt_bin", "tick", "seed", "side"]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    match = data["seed"].astype(np.int64) * 2 + data["side"]
    # several opponents / runs can share seeds: the file index keeps matches apart
    fid = np.concatenate([np.full(len(p["tick"]), i) for i, p in enumerate(parts)])
    match = match * 1000 + fid
    ids = np.unique(match)
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, max(1, int(len(ids) * args.val_frac)), replace=False).tolist())
    is_val = np.array([m in val_ids for m in match])
    data["adv"] = (data["values"][:, 1:6] - data["values"][:, :1]).astype(np.float32)
    data["match"] = match
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sel in (("train", ~is_val), ("val", is_val)):
        np.savez(out / f"{name}.npz", **{k: v[sel] for k, v in data.items()})
        a = data["adv"][sel]
        print(name, int(sel.sum()), "roots", f"adv mean {a.mean():+.2f} sd {a.std():.2f} |adv|>3 {np.mean(np.abs(a) > 3):.2f}",
              f"search non-base {np.mean(data['choice'][sel] > 0):.2f}", flush=True)
    json.dump({"branches": branches, "files": files}, open(out / "meta.json", "w"), indent=1)


def build(init: str, q_hidden: int):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet

    ck = torch.load(init, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg["q_head"] = cfg.get("q_head") or q_hidden
    net = RolloutNet(**cfg)
    missing, unexpected = net.load_state_dict(ck["state"], strict=False)
    assert not unexpected and all(k.startswith("q_net.") for k in missing), (missing, unexpected)
    return net


def tensors(data, idx, device, torch):
    from rollout_net_train import batch_tensors

    g, ts, ag = batch_tensors(data, idx, device, torch)
    adv = torch.from_numpy(np.clip(data["adv"][idx], -CLIP, CLIP)).to(device)
    return g, ts, ag, adv


def predict(net, data, device, torch, batch=4096):
    out = []
    with torch.inference_mode():
        for s in range(0, len(data["tick"]), batch):
            idx = np.arange(s, min(len(data["tick"]), s + batch))
            g, ts, ag, _ = tensors(data, idx, device, torch)
            out.append(net.q_alt(g, ts, ag)[2].float().cpu().numpy())
    return np.concatenate(out)


def metrics(pred: np.ndarray, data) -> dict:
    adv = data["adv"].astype(np.float64)
    r = len(adv)
    res = {"roots": r, "corr": float(np.corrcoef(pred.ravel(), np.clip(adv, -CLIP, CLIP).ravel())[0, 1]),
           "mse": float(np.mean((pred - np.clip(adv, -CLIP, CLIP)) ** 2))}
    best_u = pred.argmax(1)
    op = np.asarray(data["tick"]) < 250
    for m in (0.0, 3.0):
        go = pred[np.arange(r), best_u] > m
        res[f"open_gain_m{m:g}"] = float(np.where(go, adv[np.arange(r), best_u], 0.0)[op].mean())
        res[f"later_gain_m{m:g}"] = float(np.where(go, adv[np.arange(r), best_u], 0.0)[~op].mean())
    for m in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0):
        go = pred[np.arange(r), best_u] > m
        gain = np.where(go, adv[np.arange(r), best_u], 0.0)
        res[f"gain_m{m:g}"] = float(gain.mean())
        res[f"dev_m{m:g}"] = float(go.mean())
    # the search's own choice among base + alt (its values include the sampled branches; restrict to alt)
    sbest = adv.argmax(1)
    sgo = adv[np.arange(r), sbest] > 3.0
    res["search_alt_gain_m3"] = float(np.where(sgo, adv[np.arange(r), sbest], 0).mean())  # optimistic (same noise)
    res["search_dev_m3"] = float(sgo.mean())
    res["random_unit_gain"] = float(adv.mean())
    res["oracle_any_alt"] = float(np.maximum(adv.max(1), 0).mean())
    return res


def cmd_train(args):
    import torch

    from blackout_env.sim.search import default_device

    torch.manual_seed(args.seed)
    device = args.device or default_device()
    from rollout_net_train import load_split

    train = load_split(Path(args.data) / "train.npz")
    val = load_split(Path(args.data) / "val.npz")
    net = build(args.init, args.q_hidden).to(device)
    ref = copy.deepcopy(net).eval() if not args.freeze_trunk and args.kl > 0 else None
    params = [p for n, p in net.named_parameters() if n.startswith("q_net.") or not args.freeze_trunk]
    for n, p in net.named_parameters():
        p.requires_grad_(n.startswith("q_net.") or not args.freeze_trunk)
    groups = [{"params": [p for n, p in net.named_parameters() if n.startswith("q_net.")], "lr": args.lr}]
    if not args.freeze_trunk:
        groups.append({"params": [p for n, p in net.named_parameters() if not n.startswith("q_net.")], "lr": args.trunk_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)
    print("train roots", len(train["tick"]), "val", len(val["tick"]), "trainable", sum(p.numel() for p in params), flush=True)
    rng = np.random.default_rng(args.seed)
    n = len(train["tick"])
    train_open = np.asarray(train["tick"]) < 250
    best, log, t0 = -1e9, [], time.time()
    for step in range(1, args.steps + 1):
        idx = np.sort(rng.choice(n, args.batch))
        g, ts, ag, adv = tensors(train, idx, device, torch)
        net.train()
        logits, off, pred = net.q_alt(g, ts, ag, detach_trunk=args.freeze_trunk)
        if args.target == "bce":  # P(adv > bce_margin): the head's output is a logit (act threshold = logit)
            wpos = torch.where(adv > args.bce_margin, args.pos_weight, 1.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, (adv > args.bce_margin).float(), weight=wpos)
        else:
            w = torch.where(torch.from_numpy(train_open[idx]).to(device)[:, None], args.opening_weight, 1.0)
            loss = (torch.nn.functional.huber_loss(pred, adv, delta=args.huber, reduction="none") * w).mean()
        if ref is not None:
            with torch.no_grad():
                rl, ro = ref(g, ts, ag)
            kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(rl, -1), log_target=True,
                                            reduction="none").sum(-1).mean()
            j = rl.argmax(-1, keepdim=True)
            oreg = (((off.gather(-1, j) - ro.gather(-1, j)) / (np.pi / 32)) ** 2).mean()
            loss = loss + args.kl * (kl + oreg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            m = metrics(predict(net, val, device, torch), val)
            row = {"step": step, "loss": float(loss), "s": round(time.time() - t0), **{k: round(v, 4) for k, v in m.items()}}
            log.append(row)
            print(json.dumps(row), flush=True)
            score = m["corr"] if args.target == "huber" else max(m[k] for k in m if k.startswith("gain_m"))
            if score > best:
                best = score
                save = {"config": net.config, "state": {k: v.cpu() for k, v in net.state_dict().items()}, "args": vars(args),
                        "step": step, "val": m}
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(save, args.out)
    ck = torch.load(args.out, map_location="cpu", weights_only=False)
    ck["log"] = log
    torch.save(ck, args.out)
    print("best step", ck["step"], json.dumps(ck["val"]), flush=True)


def cmd_eval(args):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.sim.search import default_device
    from rollout_net_train import load_split

    device = args.device or default_device()
    val = load_split(Path(args.data) / "val.npz")
    out = {}
    for path in args.net:
        net = RolloutNet.from_checkpoint(path).to(device)
        out[path] = metrics(predict(net, val, device, torch), val)
        # the net's alt bins must be the recorded ones (same trunk as the search's r5 when frozen)
        with torch.inference_mode():
            idx = np.arange(min(4096, len(val["tick"])))
            g, ts, ag, _ = tensors(val, idx, device, torch)
            from blackout_env.sim.rollout_net import alt_bins

            lg = net(g, ts, ag)[0]
            out[path]["alt_bin_match"] = float((alt_bins(lg).cpu().numpy() == np.asarray(val["alt_bin"][idx])).mean())
        print(path, json.dumps(out[path]), flush=True)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pack")
    q.add_argument("--roots", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--val-frac", type=float, default=0.1)
    q = sub.add_parser("train")
    q.add_argument("--data", required=True)
    q.add_argument("--init", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--q-hidden", type=int, default=256)
    q.add_argument("--freeze-trunk", action="store_true")
    q.add_argument("--kl", type=float, default=1.0, help="unfrozen trunk: KL to the init policy (keeps r5's moves)")
    q.add_argument("--steps", type=int, default=10000)
    q.add_argument("--batch", type=int, default=512)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--trunk-lr", type=float, default=1e-4)
    q.add_argument("--wd", type=float, default=1e-4)
    q.add_argument("--huber", type=float, default=5.0)
    q.add_argument("--target", default="huber", choices=["huber", "bce"])
    q.add_argument("--bce-margin", type=float, default=3.0)
    q.add_argument("--pos-weight", type=float, default=1.0)
    q.add_argument("--opening-weight", type=float, default=1.0, help="huber: weight of roots in the first 10 s")
    q.add_argument("--eval-every", type=int, default=1000)
    q.add_argument("--device", default="")
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("eval")
    q.add_argument("--data", required=True)
    q.add_argument("--net", nargs="+", required=True)
    q.add_argument("--device", default="")
    q.add_argument("--json", default="")
    args = p.parse_args()
    {"pack": cmd_pack, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
