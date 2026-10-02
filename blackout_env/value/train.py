"""S1 value-net training and evaluation over the row cache (data.py); CLI: examples/value_train.py."""

from __future__ import annotations

import contextlib
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from blackout_env.value.baselines import split_masks
from blackout_env.value.data import SPLITS, open_cache
from blackout_env.value.metrics import evaluate
from blackout_env.value.model import TRUNK_MODULES, ValueNet, load_value_net, save_value_net, value_loss

INPUTS = ("graphic", "team_state", "agent_states")
TARGETS = ("delta", "delta_mask", "final", "result")


def gather(cache: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.sort(idx)
    return {k: np.asarray(cache[k][idx]) for k in INPUTS + TARGETS + ("elapsed_s",)}


def to_device(batch: dict, device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in batch.items()}


class Prefetcher:
    """Random training batches gathered from the memory-mapped cache in a background thread."""

    def __init__(self, cache: dict, rows: np.ndarray, batch_size: int, seed: int, depth: int = 6):
        self.cache, self.rows, self.batch_size = cache, rows, batch_size
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(depth)
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            self.q.put(gather(self.cache, self.rows[self.rng.integers(0, len(self.rows), self.batch_size)]))

    def next(self) -> dict:
        return self.q.get()

    def stop(self):
        self._stop = True


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def predict_rows(net: ValueNet, cache: dict, idx: np.ndarray, device, batch_size: int = 1024, amp=None) -> dict:
    net.eval()
    parts = []
    for a in range(0, len(idx), batch_size):
        b = to_device({k: np.asarray(cache[k][idx[a:a + batch_size]]) for k in INPUTS}, device)
        ctx = torch.autocast(device.type, dtype=amp) if amp else contextlib.nullcontext()
        with ctx:
            parts.append(net.predict(b["graphic"], b["team_state"], b["agent_states"]))
    net.train()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def eval_rows(cache: dict, idx: np.ndarray) -> dict:
    return {k: np.asarray(cache[k][idx]) for k in ("delta", "delta_mask", "final", "result", "elapsed_s", "team_state")}


def quick_metrics(report: dict) -> dict:
    out = {f"r2_{k}": v["all"].get("r2") for k, v in report["delta"].items()}
    out.update(r2_final=report["final"]["all"].get("r2"), r2_final_rem=report["final_remaining"]["all"].get("r2"),
               brier=report["win"]["all"]["brier"], acc=report["win"]["all"]["acc"], ece=report["win"]["all"]["ece"])
    return out


def train(args) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    cache = open_cache(args.cache)
    matches = json.loads((Path(args.cache) / "matches.json").read_text())
    split = np.asarray(cache["split"])
    train_rows = np.flatnonzero(split == SPLITS.index("train"))
    rng = np.random.default_rng(args.seed + 1)
    masks = split_masks(cache, matches)
    eval_sets = {name: np.sort(rng.choice(np.flatnonzero(m), size=min(args.eval_rows, int(m.sum())), replace=False))
                 for name, m in masks.items() if name in ("heldout", "unseen_policy", "unseen_ood") and m.any()}

    net = ValueNet().to(device)
    if args.init is not None:
        init = torch.load(args.init, map_location="cpu", weights_only=True)
        net.load_trunk(init["policy_state"])
    if args.compile:
        for name in TRUNK_MODULES:
            getattr(net.trunk, name).compile()
    params = [{"params": list(net.trunk_parameters()), "lr_scale": args.trunk_lr_scale},
              {"params": net.head_parameters(), "lr_scale": 1.0}]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "fn"}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(f"train rows {len(train_rows)}, eval {{{', '.join(f'{k}: {len(v)}' for k, v in eval_sets.items())}}}", flush=True)

    prefetch = Prefetcher(cache, train_rows, args.batch_size, args.seed)
    best = math.inf
    totals: dict[str, torch.Tensor] = {}
    n_since = 0
    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            reports = {name: evaluate(predict_rows(net, cache, idx, device, amp=amp), eval_rows(cache, idx))
                       for name, idx in eval_sets.items()}
            quick = {name: quick_metrics(r) for name, r in reports.items()}
            train_stats = {k: float(v) / max(1, n_since) for k, v in totals.items()}
            totals, n_since = {}, 0
            select = quick["heldout"]  # checkpoint selection: held-out matches only, never the unseen sets
            score = sum(1 - (select[f"r2_{h}"] or 0) for h in ("2s", "5s", "10s", "20s")) + 4 * select["brier"]
            entry = {"step": step, "elapsed_s": round(time.time() - started, 1), "lr": lr_at(step, args),
                     "train": train_stats, **quick, "select_score": score}
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(json.dumps({"step": step, "t": entry["elapsed_s"], "loss": train_stats.get("loss"),
                              **{k: {m: round(v, 4) for m, v in q.items() if v is not None} for k, q in quick.items()}}),
                  flush=True)
            meta = {"step": step, "init": str(args.init), "cache": str(args.cache), "quick": quick}
            save_value_net(out_dir / "latest.pt", net, meta)
            if score < best:
                best = score
                save_value_net(out_dir / "best.pt", net, meta)
            if step == args.steps:
                break
        lr = lr_at(step, args)
        for group in opt.param_groups:
            group["lr"] = lr * group["lr_scale"]
        batch = to_device(prefetch.next(), device)
        ctx = torch.autocast(device.type, dtype=amp) if amp else contextlib.nullcontext()
        with ctx:
            out = net.forward_u8(batch["graphic"], batch["team_state"], batch["agent_states"])
        lead = batch["team_state"][:, 0] - batch["team_state"][:, 1]
        loss, stats = value_loss(out, batch, lead)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        for k, v in stats.items():  # summed on the device: no host sync per step
            totals[k] = totals[k] + v if k in totals else v.clone()
        n_since += 1
    prefetch.stop()
    print(f"done: {args.steps} steps in {time.time() - started:.0f} s; best select score {best:.4f}", flush=True)


def evaluate_checkpoint(args) -> dict:
    device = torch.device(args.device)
    net = load_value_net(args.ckpt, device)
    cache = open_cache(args.cache)
    matches = json.loads((Path(args.cache) / "matches.json").read_text())
    report = {}
    rng = np.random.default_rng(0)
    for name, mask in split_masks(cache, matches).items():
        idx = np.flatnonzero(mask)
        if args.max_rows is not None and len(idx) > args.max_rows:
            idx = np.sort(rng.choice(idx, args.max_rows, replace=False))
        report[name] = evaluate(predict_rows(net, cache, idx, device, args.batch_size), eval_rows(cache, idx))
        print(f"eval {name}: {len(idx)} rows", flush=True)
    return {args.name: report}
