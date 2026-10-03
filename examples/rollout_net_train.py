"""Train the small rollout net (blackout_env/sim/rollout_net.py) on distillation shards.

    # 1. pack shards (canonical rows, map bit-packed to uint16) into one .npz per split
    PYTHONPATH=. .venv/bin/python examples/rollout_net_train.py pack \
        --shards /Users/mac/project/26rl/blackout-env/checkpoints/distill/p11/shards --out reports/rollout_net/data
    # 2. train (act-teacher strategic_v23 labels: K-bin arc set + raw angle, as P11)
    PYTHONPATH=. .venv/bin/python examples/rollout_net_train.py train --data reports/rollout_net/data \
        --out checkpoints/rollout_net/r1.pt
    # 3. row-level fidelity vs the teacher (and P11 on the same rows)
    PYTHONPATH=. .venv/bin/python examples/rollout_net_train.py eval --data reports/rollout_net/data --net checkpoints/rollout_net/r1.pt

Losses are P11's (train/distill/loss.py): set-valued CE on the arc label + angle-offset loss
(arc 1), rows in the first 10 s weighted x1.5.  ``--p11-kl W`` adds W x KL(P11 || net) on the
logits and an offset regression to P11's offsets on its arg max bin (targets precomputed by
``pack --p11``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P11 = "/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p11_v22.pt"
OPENING_TICKS = 250  # decisions (10 s)


def pack_graphic(g: np.ndarray) -> np.ndarray:
    """uint8 [T, H, W, 13] (x15, canonical) -> uint16 [T, H, W]: base | battery << 3 | item << 7."""
    base = g[..., :8].argmax(-1).astype(np.uint16)
    bat = g[..., 8].astype(np.uint16)
    items = g[..., 9:13]
    item = np.where(items.max(-1) > 0, items.argmax(-1) + 1, 0).astype(np.uint16)
    return base | (bat << 3) | (item << 7)


def _load(path: str):
    from blackout_env.model.direction_bins import bits_bins, convert_bits

    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    labels = z["labels"]
    if bits_bins(labels, meta.get("action_bins")) != 32:
        labels = convert_bits(labels, bits_bins(labels, meta.get("action_bins")), 32)
    t = len(labels)
    tick = z["tick"] if "tick" in z.files else np.full(t, -1, np.int32)
    angles = z["angles"] if "angles" in z.files else np.full((t, 5), np.nan, np.float32)
    return {"map": pack_graphic(z["graphic"]), "team_state": z["team_state"].astype(np.float32),
            "agent_states": z["agent_states"].astype(np.float32), "labels": labels.astype(np.uint64),
            "angles": angles.astype(np.float32), "tick": tick.astype(np.int32)}


def cmd_pack(args):
    paths = sorted(str(p) for p in Path(args.shards).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if args.max_shards:
        paths = paths[: args.max_shards]
    n_val = max(1, int(len(paths) * args.val_frac))
    splits = {"train": paths[:-n_val], "val": paths[-n_val:]}  # the last shards (latest student versions) held out
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, ps in splits.items():
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            parts = list(ex.map(_load, ps, chunksize=8))
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if args.p11:
            data.update(p11_targets(data))
        np.savez(out / f"{name}.npz", **data)
        print(name, len(ps), "shards", len(data["tick"]), "rows", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------ tensors


def expand_map(packed, torch):
    """uint16-as-int32 [B, H, W] -> float32 [B, 13, H, W] (the env graphic, canonical)."""
    p = packed.long()
    base = torch.nn.functional.one_hot(p & 7, 8)
    bat = ((p >> 3) & 15).float().unsqueeze(-1) / 15.0
    item = (p >> 7) & 7
    items = torch.nn.functional.one_hot(item, 8)[..., 1:5]
    return torch.cat([base.float(), bat, items.float()], -1).permute(0, 3, 1, 2).contiguous()


def unpack_labels(bits: np.ndarray) -> np.ndarray:
    shifts = np.arange(32, dtype=np.uint64)
    return ((bits[..., None] >> shifts) & np.uint64(1)).astype(bool)


def batch_tensors(data, idx, device, torch):
    m = torch.from_numpy(data["map"][idx].astype(np.int32)).to(device)
    g = expand_map(m, torch)
    ts = torch.from_numpy(data["team_state"][idx]).to(device)
    ag = torch.from_numpy(data["agent_states"][idx]).to(device)
    return g, ts, ag


def p11_targets(data, device="mps", batch=2048):
    """P11's logits and offsets on every row (the --p11-kl targets), fp16 autocast."""
    import torch

    from blackout_env.train.distill.inference import _OwnQ, load_net

    net = load_net(torch.load(P11, map_location="cpu", weights_only=True)["policy_state"]).eval()
    own = _OwnQ(net).to(device)
    n = len(data["tick"])
    q = np.zeros((n, 5, 32), np.float16)
    off = np.zeros((n, 5, 32), np.float16)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
        for s in range(0, n, batch):
            idx = np.arange(s, min(n, s + batch))
            g, ts, ag = batch_tensors(data, idx, device, torch)
            a, b = own(g, ts, ag)
            q[idx], off[idx] = a.float().cpu().numpy(), b.float().cpu().numpy()
    return {"p11_q": q, "p11_off": off}


# ------------------------------------------------------------------------------ train


def headings(logits, off, k=32):
    """Executed world heading of the arg max bin + its offset (radians, canonical frame)."""
    import torch

    j = logits.argmax(-1)
    o = off.gather(-1, j.unsqueeze(-1)).squeeze(-1)
    return j.float() * (2 * math.pi / k) + o


def ang_err_deg(pred, target):
    import torch

    d = torch.remainder(pred - target + math.pi, 2 * math.pi) - math.pi
    return d.abs() * 180 / math.pi


def evaluate(net_fn, data, device, torch, n=None, batch=4096):
    """Row metrics of a (graphic, team, agents) -> (logits, offsets) function on the labelled units."""
    rows = np.arange(len(data["tick"])) if n is None else np.random.default_rng(1).choice(len(data["tick"]), n, replace=False)
    acc = {k: [0.0, 0.0] for k in ("hit", "err", "w2", "hit_open", "err_open", "w2_open", "p11_agree", "p11_err")}
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        g, ts, ag = batch_tensors(data, idx, device, torch)
        with torch.inference_mode():
            logits, off = net_fn(g, ts, ag)
        logits, off = logits.float(), off.float()
        labels = torch.from_numpy(unpack_labels(data["labels"][idx])).to(device)
        valid = labels.any(-1)
        ang = torch.from_numpy(data["angles"][idx]).to(device)
        has = torch.isfinite(ang)
        hit = labels.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        err = ang_err_deg(headings(logits, off), torch.nan_to_num(ang))
        opening = torch.from_numpy((data["tick"][idx] >= 0) & (data["tick"][idx] < OPENING_TICKS)).to(device)[:, None].expand_as(valid)
        for key, mask, val in (("hit", valid, hit.float()), ("err", has, err), ("w2", has, (err <= 2).float()),
                               ("hit_open", valid & opening, hit.float()), ("err_open", has & opening, err),
                               ("w2_open", has & opening, (err <= 2).float())):
            acc[key][0] += float(val[mask].sum())
            acc[key][1] += float(mask.sum())
        if "p11_q" in data:
            pq = torch.from_numpy(data["p11_q"][idx].astype(np.float32)).to(device)
            po = torch.from_numpy(data["p11_off"][idx].astype(np.float32)).to(device)
            moving = torch.ones_like(valid)
            agree = (pq.argmax(-1) == logits.argmax(-1)).float()
            e = ang_err_deg(headings(logits, off), headings(pq, po))
            acc["p11_agree"][0] += float(agree.sum()); acc["p11_agree"][1] += float(moving.sum())
            acc["p11_err"][0] += float(e.sum()); acc["p11_err"][1] += float(moving.sum())
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def labelled_rows(labels: np.ndarray) -> np.ndarray:
    """Rows with a label (any of the 32 bins) on at least one own unit."""
    return np.flatnonzero(((labels & np.uint64(0xFFFFFFFF)) != 0).any(-1))


def load_split(path, in_memory: bool = False) -> dict:
    """A packed split (train.npz / val.npz): memory-mapped (np.savez stores it uncompressed), so
    torchrun ranks on one machine share the page cache; ``in_memory`` reads it into RAM."""
    from blackout_env.train.accel import load_npz

    data = load_npz(path, mmap=True)
    return {k: np.array(v) for k, v in data.items()} if in_memory else data


class BatchSource:
    """Host side of a training batch: row indices (the same RNG draws, so the same batches as
    before the CUDA work) gathered from the train split and the ``--extra`` split.  Called from
    the prefetch thread.  uint16 maps / uint64 label bits travel as int16 / int64 views (torch
    has no unsigned shifts on every device); they are decoded on the device (``TrainStep``)."""

    def __init__(self, train, extra, pool, epool, batch, extra_frac, seed, keys):
        self.train, self.extra, self.pool, self.epool = train, extra, pool, epool
        self.batch, self.extra_frac, self.keys = batch, extra_frac, keys
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> dict:
        if self.extra is not None:  # a fixed fraction of each batch from the extra rows
            ne = int(self.batch * self.extra_frac)
            parts = [(self.train, np.sort(self.rng.choice(self.pool, self.batch - ne))),
                     (self.extra, np.sort(self.rng.choice(self.epool, ne)))]
        else:
            parts = [(self.train, np.sort(self.rng.choice(self.pool, self.batch)))]
        out = {k: np.concatenate([src[k][i] for src, i in parts]) for k in self.keys}
        out["map"] = out["map"].view(np.int16)
        out["labels"] = out["labels"].view(np.int64)
        return out


def train_step_module(torch):
    from blackout_env.train.distill.loss import angle_offset_loss, set_valued_cross_entropy

    class TrainStep(torch.nn.Module):
        """Device side of a step: decode the packed batch, forward (autocast), losses.  One module
        so that ``--compile`` makes one graph of it (and DDP wraps it whole)."""

        def __init__(self, net, args, amp, device):
            super().__init__()
            self.net, self.args, self.amp, self.device_type = net, args, amp, device.type
            self.register_buffer("shifts", torch.arange(32, device=device), persistent=False)

        def forward(self, map, team_state, agent_states, labels, angles, tick, p11_q=None, p11_off=None,
                    ce_total=None, angle_total=None):
            from blackout_env.train.accel import autocast

            args = self.args
            g = expand_map(map, torch)
            masks = ((labels.unsqueeze(-1) >> self.shifts) & 1) == 1
            opening = (tick >= 0) & (tick < OPENING_TICKS)
            w = torch.where(opening, args.opening_weight, 1.0).float()[:, None].expand(-1, 5)
            with autocast(torch.device(self.device_type), self.amp):
                logits, off = self.net(g, team_state, agent_states)
            logits, off = logits.float(), off.float()
            ce, st = set_valued_cross_entropy(logits, masks, w, total=ce_total)
            al, ast = angle_offset_loss(off, angles, w, arc_bins=self.net.config["angle_arc_bins"], total=angle_total)
            loss = ce + args.angle_weight * al
            if args.p11_kl > 0:
                pq, po = p11_q.float(), p11_off.float()
                kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(pq, -1),
                                                log_target=True, reduction="none").sum(-1).mean()
                j = pq.argmax(-1, keepdim=True)
                oreg = (((off.gather(-1, j) - po.gather(-1, j)) / (math.pi / 32)) ** 2).mean()
                loss = loss + args.p11_kl * (kl + oreg)
            stats = torch.stack([loss.detach(), st["ce"], st["accuracy"], ast["angle_loss"]])
            return loss, stats

    return TrainStep


def global_totals(b, args, dist, torch):
    """DDP: the loss normalisers (sum of row weights over labelled units / over rows with an
    angle) summed over every rank, divided by the world size -- so that DDP's mean of the ranks'
    gradients is exactly the gradient of the global batch's weighted mean (not a mean of means)."""
    if not dist.enabled:
        return None, None
    w = torch.where((b["tick"] >= 0) & (b["tick"] < OPENING_TICKS), args.opening_weight, 1.0).float()[:, None]
    t = torch.stack([(w * (b["labels"] != 0)).sum(), (w * torch.isfinite(b["angles"])).sum()])
    dist.all_reduce_sum(t)
    t = t / dist.world
    return t[0].clamp_min(1e-8), t[1].clamp_min(1e-8)


def cmd_train(args):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.accel import Dist, Prefetcher, adamw, check_amp, grad_scaler, pick_device, to_device

    device = pick_device(args.device)
    dist = Dist(device)
    amp = check_amp(device, args.amp)
    torch.manual_seed(args.seed)
    log_main = print if dist.main else (lambda *a, **k: None)
    train = load_split(Path(args.data) / "train.npz", args.in_memory)
    val = load_split(Path(args.data) / "val.npz", True) if dist.main else None
    pool = labelled_rows(train["labels"])
    extra = epool = None
    keys = ["map", "team_state", "agent_states", "labels", "angles", "tick"] + (["p11_q", "p11_off"] if args.p11_kl > 0 else [])
    if args.extra:  # e.g. simulator DAgger rows (rollout_net_collect.py): a fixed fraction of each batch
        extra = load_split(args.extra, args.in_memory)
        epool = labelled_rows(extra["labels"])
        keys = [k for k in keys if k in extra]
        log_main("extra rows", len(extra["tick"]), "labelled", len(epool), flush=True)
    log_main("train rows", len(train["tick"]) + (len(extra["tick"]) if extra is not None else 0), "labelled", len(pool),
             "val", len(val["tick"]) if val is not None else 0, flush=True)
    if args.batch % dist.world:
        raise SystemExit(f"--batch {args.batch} is not divisible by {dist.world} ranks")
    cfg = json.loads(args.config) if args.config else {}
    net = RolloutNet.from_checkpoint(args.init).train().to(device) if args.init else RolloutNet(**cfg).to(device)
    log_main("params", sum(p.numel() for p in net.parameters()), cfg, f"device {device} x{dist.world} amp {amp}"
             f"{' compile' if args.compile else ''}", flush=True)
    opt = adamw(net.parameters(), device, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = grad_scaler(device, amp)
    step_fn = dist.wrap(train_step_module(torch)(net, args, amp, device))
    if args.compile:
        step_fn = torch.compile(step_fn)
    # rank r draws its own rows (seed + 1000 r); the global batch is --batch split over the ranks
    source = BatchSource(train, extra, pool, epool, args.batch // dist.world, args.extra_frac,
                         args.seed + 1000 * dist.rank, keys)
    batches = Prefetcher(source, device, depth=args.prefetch) if args.prefetch > 0 else None
    t0 = t_win = time.time()
    log = []
    last_logged = 0
    stat_sum = torch.zeros(4, device=device)
    for step in range(1, args.steps + 1):
        b = to_device(batches.next() if batches else source(), device)
        ce_total, angle_total = global_totals(b, args, dist, torch)
        loss, stats = step_fn(b["map"], b["team_state"], b["agent_states"], b["labels"], b["angles"], b["tick"],
                              b.get("p11_q"), b.get("p11_off"), ce_total, angle_total)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        stat_sum += stats
        if step % args.log_every == 0 or step == args.steps:
            # loss / ce / acc / angle_loss of the last step as before (under DDP: of the global
            # batch, = the mean over ranks); loss_mean: over the steps since the last log
            n_steps = step - last_logged
            last = dist.all_reduce_sum(stats.clone()) / dist.world
            mean = dist.all_reduce_sum(stat_sum.clone()) / (dist.world * n_steps)
            stat_sum.zero_()
            now = time.time()
            row = {"step": step, "loss": float(last[0]), "ce": float(last[1]), "acc": float(last[2]),
                   "angle_loss": float(last[3]), "loss_mean": float(mean[0]), "s": now - t0,
                   "ms_step": 1000 * (now - t_win) / n_steps, "rows_s": args.batch * n_steps / (now - t_win)}
            if scaler.is_enabled():
                row["loss_scale"] = scaler.get_scale()
            if step % args.eval_every == 0 or step == args.steps:
                if dist.main:
                    net.eval()
                    row.update({f"val_{k}": v for k, v in evaluate(lambda a, b, c: net(a, b, c), val, device, torch,
                                                                    n=min(60000, len(val["tick"]))).items()})
                    net.train()
                    save_rollout_net(args.out, net, args, step=step)
                dist.barrier()
            log.append(row)
            log_main(json.dumps(row), flush=True)
            last_logged = step
            t_win = time.time()
    if batches:
        batches.stop()
    if dist.main:
        save_rollout_net(args.out, net, args, log=log)
    dist.close()


def save_rollout_net(path, net, args, **extra):
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": net.config, "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "args": vars(args), **extra}, path)


def cmd_eval(args):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.distill.inference import _OwnQ, load_net

    from blackout_env.train.accel import pick_device

    val = dict(np.load(Path(args.data) / "val.npz"))
    device = pick_device(args.device)
    out = {}
    if args.net:
        net = RolloutNet.from_checkpoint(args.net).to(device)
        out["small"] = evaluate(lambda a, b, c: net(a, b, c), val, device, torch)
    if args.with_p11:
        p11 = _OwnQ(load_net(torch.load(P11, map_location="cpu", weights_only=True)["policy_state"]).eval()).to(device)

        def f(a, b, c):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                return p11(a, b, c)

        out["p11"] = evaluate(f, val, device, torch)
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.04)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--p11", action="store_true", help="also store P11's logits/offsets per row (MPS)")
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="", help="RolloutNet kwargs as JSON")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--angle-weight", type=float, default=1.0)
    p.add_argument("--opening-weight", type=float, default=1.5)
    p.add_argument("--p11-kl", type=float, default=0.0)
    p.add_argument("--amp", nargs="?", const="bf16", default="none", choices=["none", "fp16", "bf16"],
                   help="autocast of the forward; bare --amp = bf16 (as before; MPS).  CUDA 2080 Ti: fp16 "
                   "(+ GradScaler; Turing has no bf16)")
    p.add_argument("--compile", action="store_true", help="torch.compile the whole step (decode + net + losses)")
    p.add_argument("--prefetch", type=int, default=4, help="batches built ahead in a thread (0 = inline)")
    p.add_argument("--in-memory", action="store_true", help="read the splits into RAM instead of memory-mapping")
    p.add_argument("--extra", default="", help="another packed train.npz mixed into every batch")
    p.add_argument("--extra-frac", type=float, default=0.5)
    p.add_argument("--init", default="", help="start from this RolloutNet checkpoint (its config)")
    p.add_argument("--device", default="auto", help="auto = cuda > mps > cpu (torchrun: this rank's GPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=5000)
    p = sub.add_parser("eval")
    p.add_argument("--data", required=True)
    p.add_argument("--net", default="")
    p.add_argument("--with-p11", action="store_true")
    p.add_argument("--device", default="auto")
    args = parser.parse_args()
    {"pack": cmd_pack, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
