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
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P11 = os.environ.get("BLACKOUT_P11", "/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p11_v22.pt")
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


def mmap_npz(path) -> dict:
    """Memory-map every member of an uncompressed ``np.savez`` archive (ZIP_STORED: the .npy bytes sit
    verbatim in the file), so concurrent trainings share one page-cache copy and nothing is loaded
    up front.  Compressed members are loaded normally."""
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        for info in zf.infolist():
            name = info.filename.removesuffix(".npy")
            if info.compress_type != zipfile.ZIP_STORED:
                out[name] = np.load(path)[name]
                continue
            fh.seek(info.header_offset)
            local = fh.read(30)
            n_name, n_extra = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
            fh.seek(info.header_offset + 30 + n_name + n_extra)
            version = np.lib.format.read_magic(fh)
            shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
            out[name] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape,
                                  order="F" if fortran else "C")
    return out


def load_split(path) -> dict:
    """A packed split: ``<name>.npz`` (memory-mapped when uncompressed, see ``mmap_npz``) or a
    directory of per-key ``.npy`` files (``convert``)."""
    path = Path(path)
    d = path.with_suffix("") if path.suffix == ".npz" else path
    if d.is_dir() and any(d.glob("*.npy")):
        return {f.stem: np.load(f, mmap_mode="r") for f in sorted(d.glob("*.npy"))}
    return mmap_npz(path)


def cmd_convert(args):
    """npz -> a directory of .npy files next to it (same stem), for ``load_split``'s mmap."""
    for src in args.npz:
        z = np.load(src)
        out = Path(src).with_suffix("")
        out.mkdir(parents=True, exist_ok=True)
        for k in z.files:
            np.save(out / f"{k}.npy", z[k])
        print(src, "->", out, z.files, flush=True)


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
    m = torch.from_numpy(np.asarray(data["map"][idx]).astype(np.int32)).to(device)
    g = expand_map(m, torch)
    ts = torch.from_numpy(np.ascontiguousarray(data["team_state"][idx])).to(device)
    ag = torch.from_numpy(np.ascontiguousarray(data["agent_states"][idx])).to(device)
    return g, ts, ag


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def autocast(torch, device, amp: str):
    """amp: '' (off) | 'fp16' | 'bf16' -> a torch.autocast context."""
    if not amp:
        return torch.autocast(device_type=device.split(":")[0], enabled=False)
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


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
    acc = {k: [0.0, 0.0] for k in ("hit", "err", "w2", "hit_open", "err_open", "w2_open", "hit_late", "err_late",
                                   "p11_agree", "p11_err")}
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        g, ts, ag = batch_tensors(data, idx, device, torch)
        with torch.inference_mode():
            logits, off = net_fn(g, ts, ag)
        logits, off = logits.float(), off.float()
        labels = torch.from_numpy(unpack_labels(np.asarray(data["labels"][idx]))).to(device)
        valid = labels.any(-1)
        ang = torch.from_numpy(np.ascontiguousarray(data["angles"][idx])).to(device)
        has = torch.isfinite(ang)
        hit = labels.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        err = ang_err_deg(headings(logits, off), torch.nan_to_num(ang))
        tk = np.asarray(data["tick"][idx])
        opening = torch.from_numpy((tk >= 0) & (tk < OPENING_TICKS)).to(device)[:, None].expand_as(valid)
        late = torch.from_numpy(tk >= 3 * OPENING_TICKS).to(device)[:, None].expand_as(valid)  # 30 s +
        for key, mask, val in (("hit", valid, hit.float()), ("err", has, err), ("w2", has, (err <= 2).float()),
                               ("hit_open", valid & opening, hit.float()), ("err_open", has & opening, err),
                               ("w2_open", has & opening, (err <= 2).float()),
                               ("hit_late", valid & late, hit.float()), ("err_late", has & late, err)):
            acc[key][0] += float(val[mask].sum())
            acc[key][1] += float(mask.sum())
        if "p11_q" in data:
            pq = torch.from_numpy(np.asarray(data["p11_q"][idx]).astype(np.float32)).to(device)
            po = torch.from_numpy(np.asarray(data["p11_off"][idx]).astype(np.float32)).to(device)
            moving = torch.ones_like(valid)
            agree = (pq.argmax(-1) == logits.argmax(-1)).float()
            e = ang_err_deg(headings(logits, off), headings(pq, po))
            acc["p11_agree"][0] += float(agree.sum()); acc["p11_agree"][1] += float(moving.sum())
            acc["p11_err"][0] += float(e.sum()); acc["p11_err"][1] += float(moving.sum())
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def _gather(datasets, picks, keys):
    """Host-side batch: rows ``picks[i]`` of ``datasets[i]`` concatenated, numpy."""
    out = {}
    for k in keys:
        parts = [np.asarray(d[k][ix]) for d, ix in zip(datasets, picks) if len(ix)]
        out[k] = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return out


def cmd_train(args):
    import queue
    import threading

    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.distill.loss import angle_offset_loss, set_valued_cross_entropy

    torch.manual_seed(args.seed)
    device = args.device or default_device()
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train = load_split(Path(args.data) / "train.npz")
    val = load_split(Path(args.data) / "val.npz")
    pool = np.nonzero(unpack_labels(np.asarray(train["labels"])).any(-1).any(-1))[0]
    datasets, pools = [train], [pool]
    if args.extra:  # e.g. simulator DAgger rows (rollout_net_collect.py): a fixed fraction of each batch
        extra = load_split(args.extra)
        epool = np.nonzero(unpack_labels(np.asarray(extra["labels"])).any(-1).any(-1))[0]
        datasets.append(extra)
        pools.append(epool)
        print("extra rows", len(extra["tick"]), "labelled", len(epool), flush=True)
    keys = ["map", "team_state", "agent_states", "labels", "angles", "tick"]
    if args.p11_kl > 0:
        keys += ["p11_q", "p11_off"]
    print("train rows", len(train["tick"]), "labelled", len(pool), "val", len(val["tick"]), flush=True)
    cfg = json.loads(args.config) if args.config else {}
    net = RolloutNet.from_checkpoint(args.init).train().to(device) if args.init else RolloutNet(**cfg).to(device)
    print("params", sum(p.numel() for p in net.parameters()), net.config, "device", device, "amp", args.amp, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, **({"fused": True} if device.startswith("cuda") else {}))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
    def producer(q: queue.Queue, stop: threading.Event, worker: int):
        rng = np.random.default_rng([args.seed, worker])  # one generator per thread (not thread-safe)
        while not stop.is_set():
            if args.extra:
                ne = int(args.batch * args.extra_frac)
                picks = [np.sort(rng.choice(pools[0], args.batch - ne)), np.sort(rng.choice(pools[1], ne))]
            else:
                picks = [np.sort(rng.choice(pools[0], args.batch))]
            b = _gather(datasets, picks, keys)
            t = {k: torch.from_numpy(np.ascontiguousarray(b[k].astype(np.int32) if k == "map" else
                                                          (b[k].astype(np.float32) if k in ("p11_q", "p11_off") else b[k])))
                 for k in keys if k not in ("labels", "tick")}
            t["labels"] = torch.from_numpy(unpack_labels(b["labels"]))
            tick = b["tick"]
            w = np.where((tick >= 0) & (tick < OPENING_TICKS), args.opening_weight, 1.0).astype(np.float32)
            t["w"] = torch.from_numpy(w)
            if device.startswith("cuda"):
                t = {k: v.pin_memory() for k, v in t.items()}
            while not stop.is_set():
                try:
                    q.put(t, timeout=1.0)
                    break
                except queue.Full:
                    continue

    q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    workers = [threading.Thread(target=producer, args=(q, stop, i), daemon=True) for i in range(args.loader_threads)]
    for th in workers:
        th.start()
    t0 = time.time()
    log = []
    try:
        for step in range(1, args.steps + 1):
            b = {k: v.to(device, non_blocking=True) for k, v in q.get().items()}
            g = expand_map(b["map"], torch)
            labels, ang = b["labels"], b["angles"]
            w = b["w"][:, None].expand(-1, 5)
            with autocast(torch, device, args.amp):
                logits, off = net(g, b["team_state"], b["agent_states"])
            logits, off = logits.float(), off.float()
            ce, st = set_valued_cross_entropy(logits, labels, w)
            al, ast = angle_offset_loss(off, ang, w, arc_bins=net.config["angle_arc_bins"])
            loss = ce + args.angle_weight * al
            if args.p11_kl > 0:
                pq, po = b["p11_q"], b["p11_off"]
                kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(pq, -1),
                                                log_target=True, reduction="none").sum(-1).mean()
                j = pq.argmax(-1, keepdim=True)
                oreg = (((off.gather(-1, j) - po.gather(-1, j)) / (math.pi / 32)) ** 2).mean()
                loss = loss + args.p11_kl * (kl + oreg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % args.log_every == 0 or step == args.steps:
                row = {"step": step, "loss": float(loss), "ce": float(st["ce"]), "acc": float(st["accuracy"]),
                       "angle_loss": float(ast["angle_loss"]), "s": time.time() - t0}
                if step % args.eval_every == 0 or step == args.steps:
                    net.eval()
                    row.update({f"val_{k}": v for k, v in evaluate(lambda a, b_, c: net(a, b_, c), val, device, torch, n=min(60000, len(val["tick"]))).items()})
                    net.train()
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"config": net.config, "state": {k: v.cpu() for k, v in net.state_dict().items()},
                                "args": vars(args), "step": step}, args.out)
                log.append(row)
                print(json.dumps(row), flush=True)
    finally:
        stop.set()
        for th in workers:
            th.join(timeout=5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": net.config, "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "args": vars(args), "log": log}, args.out)


def cmd_eval(args):
    import torch

    from blackout_env.sim.rollout_net import RolloutNet
    from blackout_env.train.distill.inference import _OwnQ, load_net

    device = args.device or default_device()
    sets = {"val": load_split(Path(args.data) / "val.npz")}
    for extra in args.val2:
        sets[Path(extra).parent.name + "/" + Path(extra).stem] = load_split(extra)
    out = {}
    for name in args.net:
        net = RolloutNet.from_checkpoint(name).to(device)
        out[name] = {k: evaluate(lambda a, b, c: net(a, b, c), d, device, torch) for k, d in sets.items()}
        out[name]["params"] = sum(p.numel() for p in net.parameters())
        print(name, json.dumps(out[name]), flush=True)
    if args.with_p11:
        p11 = _OwnQ(load_net(torch.load(P11, map_location="cpu", weights_only=True)["policy_state"]).eval()).to(device)

        def f(a, b, c):
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                return p11(a, b, c)

        out["p11"] = {k: evaluate(f, d, device, torch) for k, d in sets.items()}
        print("p11", json.dumps(out["p11"]), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)


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
    p.add_argument("--amp", default="", choices=["", "fp16", "bf16"], help="autocast dtype ('' = fp32)")
    p.add_argument("--loader-threads", type=int, default=2)
    p.add_argument("--extra", default="", help="another packed train.npz mixed into every batch")
    p.add_argument("--extra-frac", type=float, default=0.5)
    p.add_argument("--init", default="", help="start from this RolloutNet checkpoint (its config)")
    p.add_argument("--device", default="", help="default: cuda > mps > cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=5000)
    p = sub.add_parser("eval")
    p.add_argument("--data", required=True)
    p.add_argument("--net", nargs="*", default=[])
    p.add_argument("--val2", nargs="*", default=[], help="more held-out splits (e.g. the sim DAgger simval/val.npz)")
    p.add_argument("--with-p11", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument("--json", default="")
    p = sub.add_parser("convert", help="npz -> directory of .npy (memory-mapped by load_split)")
    p.add_argument("npz", nargs="+")
    args = parser.parse_args()
    {"pack": cmd_pack, "train": cmd_train, "eval": cmd_eval, "convert": cmd_convert}[args.cmd](args)


if __name__ == "__main__":
    main()
