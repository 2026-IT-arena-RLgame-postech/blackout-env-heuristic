"""S1.6: fitted value iteration on simulator branch data, and its validation on the S1.5 branch set
(docs/policy_v2_design_ko.md "S1.6 분기 데이터 가치망").

    # 1. data: examples/value_branch_data.py --out datasets/value_s2v
    # 2. train: ITERS rounds; round k's targets bootstrap from round k-1's net (round 1: S1b)
    python examples/value_branch_train.py train --data datasets/value_s2v --out-dir <ckpt dir> \
        --init <main>/checkpoints/value/frozen/s1b_value.pt --iters 4 --steps 4000 --device mps
    # 3. validation on S1.5's roots (seeds 950000+; disjoint from the training seeds 960000+):
    #    regenerate the window-end / 5 s-rollout states S1.5 scored (deterministic replay, checked
    #    against the stored state hashes and S1b's stored predictions), then score any net on them
    python examples/value_branch_train.py regen --s15 <main>/reports/value_s1/branch --out <dir>/s15_views.npz
    python examples/value_branch_train.py score --views <dir>/s15_views.npz --s15 <main>/reports/value_s1/branch \
        --ckpt name=path ... --out <dir>
    python examples/value_branch_train.py bias --ckpt name=path ...   # V at match start from both sides

Batches: (1 - shard_frac) simulator rows (branch ends + on-trajectory states, each in a random one
of the two teams' views with the antisymmetric target, value/branch.py) + shard_frac rows of the
S1 shard cache (its own targets) so the net keeps what S1 learned on Unity matches.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import queue
import sys
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = Path("/Users/mac/project/26rl/blackout-env")  # read-only: checkpoints, datasets, reports
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from blackout_env.value.branch import (  # noqa: E402
    H2, H5, VIEWS, bootstrap_targets, holdout_mask, load_raw, mirror_targets)
from blackout_env.train.accel import pinned  # noqa: E402
from blackout_env.value.targets import HORIZONS_S  # noqa: E402

INPUTS = ("graphic", "team_state", "agent_states")


# ------------------------------------------------------------------------------ net helpers


def predict_states(net, states: dict, idx: np.ndarray, device, batch: int = 1024) -> dict:
    """Decoded predictions (score/100) of ``states`` rows ``idx`` (one view)."""
    import torch

    from blackout_env.value.model import decode_outputs

    net.eval()
    parts = []
    with torch.no_grad():
        for a in range(0, len(idx), batch):
            j = idx[a:a + batch]
            g = torch.from_numpy(states["graphic"][j]).to(device)
            ts = torch.from_numpy(states["team_state"][j]).to(device)
            ag = torch.from_numpy(states["agent_states"][j]).to(device)
            parts.append(decode_outputs(net.forward_u8(g, ts, ag).float(), ts[:, 0] - ts[:, 1]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def predict_sym(net, states: dict, idx: np.ndarray, device) -> dict:
    from blackout_env.value.evaluate import symmetrize

    own = predict_states(net, states["own"], idx, device)
    mir = predict_states(net, states["mir"], idx, device)
    return symmetrize(own, mir)


MSE_DEN = 10.0  # "mse": 0.5 x^2 / 10 (tenths of points) = Huber beta 10 (100 pts) inside +-100 pts
QUANTILE_KAPPA = 1.0  # quantile-Huber kappa (10 pts)


def final_head_loss(pred, target, kind: str = "huber", beta=None, quantiles=None):
    """Loss of the remaining-final-change head, both in tenths of points.  ``kind``: "huber"
    (beta, default HUBER_BETA = 10 pts: nearly L1, whose minimiser on a bimodal outcome such as a
    symmetric opening, +80 or -80, is anything in between -- not the mean), "mse" (targets the
    mean), "quantile" (``quantiles`` [B, Q] at taus (i + 0.5) / Q, quantile-Huber; the head's mean
    = their mean)."""
    import torch
    from torch.nn import functional as F

    from blackout_env.value.model import HUBER_BETA

    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=HUBER_BETA if beta is None else beta)
    if kind == "mse":
        return (0.5 * (pred - target) ** 2 / MSE_DEN).mean()
    if kind == "quantile":
        q = quantiles.float()
        n = q.shape[1]
        tau = (torch.arange(n, device=q.device, dtype=torch.float32) + 0.5) / n
        u = target.float()[:, None] - q
        hub = F.smooth_l1_loss(q, target.float()[:, None].expand_as(q), beta=QUANTILE_KAPPA, reduction="none")
        return ((tau - (u.detach() < 0).float()).abs() * hub / QUANTILE_KAPPA).sum(1).mean() / n * 2.0
    raise ValueError(kind)


def soft_value_loss(out, t: dict, lead, final_beta=None, final_loss: str = "huber", quantiles=None):
    """value_loss with a soft result target ``t['p']`` [B, 3]; the final head's loss is
    ``final_head_loss(kind=final_loss)``."""
    import torch
    from torch.nn import functional as F

    from blackout_env.value.model import HUBER_BETA, REG_SCALE

    out = out.float()
    h = out.shape[1] - 4
    mask = t["delta_mask"].float()
    reg = F.smooth_l1_loss(out[:, :h], t["delta"].float() * REG_SCALE, beta=HUBER_BETA, reduction="none")
    per_h = (reg * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    delta_loss = per_h.mean()
    rem_t = (t["final"].float() - lead.float()) * REG_SCALE
    fl = final_head_loss(out[:, h], rem_t, final_loss, final_beta, quantiles)
    ce = -(t["p"].float() * torch.log_softmax(out[:, h + 1:], dim=-1)).sum(-1).mean()
    loss = delta_loss + fl + ce
    return loss, {"loss": loss.detach(), "delta": delta_loss.detach(), "final": fl.detach(), "ce": ce.detach(),
                  "final_l1_pts": (out[:, h].detach() - rem_t).abs().mean() * (100 / REG_SCALE)}


def antisym_outputs(out_a, out_b):
    """Antisymmetric reading of the raw outputs of a state's two views: regressions (lead changes,
    remaining final change) (a - b) / 2, result logits (a + reversed b) / 2."""
    import torch

    h = out_a.shape[1] - 4
    reg = (out_a[:, :h + 1].float() - out_b[:, :h + 1].float()) / 2
    logits = (out_a[:, h + 1:].float() + torch.flip(out_b[:, h + 1:].float(), dims=[1])) / 2
    return torch.cat((reg, logits), dim=1)


# ------------------------------------------------------------------------------ train


class Sampler:
    """Background batches: simulator rows (random view, antisymmetric target) + shard rows."""

    def __init__(self, data, targets, rows, shard, shard_rows, batch, shard_frac, seed, depth=6, group_roots=0,
                 device=None, paired=False):
        self.data, self.targets, self.rows = data, targets, rows
        self.paired = paired  # also the other view of every sim row (keys + "_b"); no shard rows then
        if paired and shard is not None and shard_frac > 0:
            raise ValueError("paired views need shard_frac 0 (shard rows have one view)")
        import torch

        self.device = device if device is not None else torch.device("cpu")
        self.shard, self.shard_rows = shard, shard_rows
        self.n_shard = int(round(batch * shard_frac)) if shard is not None else 0
        self.groups = root_groups(data["rows"], rows) if group_roots else []
        self.group_roots = group_roots
        self.n_sim = batch - self.n_shard - (self._group_size() if group_roots else 0)
        self.rng = np.random.default_rng(seed)
        self.q = queue.Queue(depth)
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _group_size(self):
        return self.group_roots * len(__import__("value_branch_eval").BRANCHES)

    def _grouped(self):
        """``group_roots`` roots with all their branch rows (one view per root); the rest of the
        rows to a fixed size are filled with group -1 rows (excluded from the ranking term)."""
        pick = self.rng.integers(0, len(self.groups), self.group_roots)
        r = np.concatenate([self.groups[g] for g in pick])
        gid = np.concatenate([np.full(len(self.groups[g]), i) for i, g in enumerate(pick)])
        view = np.concatenate([np.full(len(self.groups[g]), self.rng.integers(0, 2)) for g in pick])
        size = self._group_size()
        r, gid, view = r[:size], gid[:size], view[:size]
        if len(r) < size:
            extra = self.rows[self.rng.integers(0, len(self.rows), size - len(r))]
            r = np.concatenate([r, extra])
            gid = np.concatenate([gid, np.full(len(extra), -1)])
            view = np.concatenate([view, self.rng.integers(0, 2, len(extra))])
        return r, view, gid

    def _sim_batch(self, grouped=False):
        if grouped:
            r, view, gid = self._grouped()
        else:
            r = self.rows[self.rng.integers(0, len(self.rows), self.n_sim)]
            view = self.rng.integers(0, 2, self.n_sim)  # 0 own, 1 mirror
            gid = np.full(len(r), -1)
        st_idx = self.data["rows"]["idx"][r]
        out = {}
        for k in INPUTS:
            own, mir = self.data["states"]["own"][k], self.data["states"]["mir"][k]
            out[k] = np.where(view.reshape(-1, *([1] * (own.ndim - 1))) == 0, own[st_idx], mir[st_idx])
        for k in ("delta", "delta_mask", "final", "p"):
            a, b = self.targets["own"][k][r], self.targets["mir"][k][r]
            out[k] = np.where(view.reshape(-1, *([1] * (a.ndim - 1))) == 0, a, b)
            if self.paired:
                out[k + "_b"] = np.where(view.reshape(-1, *([1] * (a.ndim - 1))) == 0, b, a)
        if self.paired:
            for k in INPUTS:
                own, mir = self.data["states"]["own"][k], self.data["states"]["mir"][k]
                out[k + "_b"] = np.where(view.reshape(-1, *([1] * (own.ndim - 1))) == 0, mir[st_idx], own[st_idx])
        out["group"] = gid.astype(np.int64)
        return out

    def _shard_batch(self):
        idx = np.sort(self.shard_rows[self.rng.integers(0, len(self.shard_rows), self.n_shard)])
        c = self.shard
        out = {k: np.asarray(c[k][idx]) for k in INPUTS}
        res = np.asarray(c["result"][idx]).astype(np.int64) + 1
        p = np.zeros((len(idx), 3), np.float32)
        p[np.arange(len(idx)), res] = 1.0
        out.update(delta=np.asarray(c["delta"][idx]), delta_mask=np.asarray(c["delta_mask"][idx]),
                   final=np.asarray(c["final"][idx]), p=p, group=np.full(len(idx), -1, np.int64))
        return out

    def _run(self):
        while not self.stop_flag:
            parts = ([self._sim_batch(grouped=True)] if self.group_roots else []) + [self._sim_batch()] \
                + ([self._shard_batch()] if self.n_shard else [])
            batch = {k: np.concatenate([p[k] for p in parts]).astype(
                np.float32 if parts[0][k].dtype == np.float64 else parts[0][k].dtype) for k in parts[0]}
            self.q.put({k: pinned(v, self.device) for k, v in batch.items()})  # host tensors, pinned on cuda

    def next(self):
        return self.q.get()

    def stop(self):
        self.stop_flag = True


def root_groups(rows: dict, train_rows: np.ndarray) -> list[np.ndarray]:
    """Branch rows of each (match, root) among ``train_rows`` (roots with >= 2 branches)."""
    sel = train_rows[rows["kind"][train_rows] == 1]
    key = rows["match"][sel].astype(np.int64) * 100_000 + rows["root"][sel]
    order = np.argsort(key, kind="stable")
    sel, key = sel[order], key[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    return [g for g in np.split(sel, cuts) if len(g) >= 2]


def rank_loss(out, t: dict, lead):
    """Within-root term: Huber between the root-centred predicted final lead and the root-centred
    target (rows with ``group`` >= 0; one root = one group).  Only differences between branches
    of the same root matter to a search; the plain regression is dominated by between-root variance."""
    import torch
    from torch.nn import functional as F

    from blackout_env.value.model import HUBER_BETA, REG_SCALE

    h = out.shape[1] - 4
    g = t["group"]
    m = g >= 0
    if not bool(m.any()):
        return out.sum() * 0.0
    g = g[m]
    pred = lead[m].float() * REG_SCALE + out[m, h].float()
    tgt = t["final"][m].float() * REG_SCALE
    n = int(g.max()) + 1
    cnt = torch.zeros(n, device=out.device).index_add_(0, g, torch.ones_like(pred))
    mp = torch.zeros(n, device=out.device).index_add_(0, g, pred) / cnt.clamp_min(1)
    mt = torch.zeros(n, device=out.device).index_add_(0, g, tgt) / cnt.clamp_min(1)
    return F.smooth_l1_loss(pred - mp[g], tgt - mt[g], beta=HUBER_BETA)


def make_targets(net, data: dict, device) -> tuple[dict, dict]:
    """Bootstrapped targets for every row from ``net`` (both views) + timing/stat info."""
    rows = data["rows"]
    started = time.time()
    nxt = rows["next_idx"]
    live = nxt >= 0
    uniq, inv = np.unique(nxt[live], return_inverse=True)
    v = predict_sym(net, data["states"], uniq, device)
    n = len(nxt)
    v_next = {"final": np.zeros(n, np.float32), "delta": np.zeros((n, len(HORIZONS_S)), np.float32),
              "p": np.zeros((n, 3), np.float32)}
    for k in v_next:
        v_next[k][live] = v[k][inv]
    t_own = bootstrap_targets(rows["lead"], rows["lead2"], rows["lead5"], rows["ended"], v_next)
    return {"own": t_own, "mir": mirror_targets(t_own)}, {"seconds": time.time() - started, "states": int(len(uniq))}


def heldout_metrics(net, data, targets, rows, device) -> dict:
    """Held-out rows: MAE of V_sym's final lead vs the bootstrapped target (the fit), and for
    on-trajectory rows vs the true match final (an unbiased check: the trajectory IS V23 vs the
    opponent); mean V_sym final - target (bias)."""
    st = data["rows"]["idx"][rows]
    v = predict_sym(net, data["states"], st, device)
    vo = predict_states(net, data["states"]["own"], st, device)
    out = {}
    tgt = targets["own"]["final"][rows]
    out["mae_sym_vs_target"] = float(np.mean(np.abs(v["final"] - tgt)) * 100)
    out["mae_own_vs_target"] = float(np.mean(np.abs(vo["final"] - tgt)) * 100)
    out["bias_own_minus_target"] = float(np.mean(vo["final"] - tgt) * 100)
    traj = data["rows"]["kind"][rows] == 0
    true = data["rows"]["match_final"][rows] / 100.0
    if traj.any():
        out["traj_mae_sym_vs_true"] = float(np.mean(np.abs(v["final"][traj] - true[traj])) * 100)
        out["traj_mae_own_vs_true"] = float(np.mean(np.abs(vo["final"][traj] - true[traj])) * 100)
        out["traj_bias_own_vs_true"] = float(np.mean(vo["final"][traj] - true[traj]) * 100)
    return out


def cmd_train(args):
    import torch

    from blackout_env.train.accel import adamw, check_amp, pick_device
    from blackout_env.value.data import SPLITS, open_cache
    from blackout_env.value.model import ValueNet, _strip_compile, load_value_net, save_value_net
    from blackout_env.value.train import compile_value_net

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = pick_device(args.device)
    check_amp(device, args.amp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items() if k != "fn"}, indent=1))
    log = (out_dir / "log.jsonl").open("a")
    t0 = time.time()
    data = load_raw(Path(args.data) / "raw")
    rows = data["rows"]
    finals = {m["task"]["id"]: m["final"] for m in data["metas"]}
    rows["match_final"] = np.array([data["metas"][m]["final"] for m in rows["match"]], np.float32)
    hold = holdout_mask(rows["seed"])
    train_rows, hold_rows = np.flatnonzero(~hold), np.flatnonzero(hold)
    kinds = {k: int((rows["kind"] == i).sum()) for i, k in enumerate(("traj", "branch"))}
    print(f"data: {len(rows['kind'])} rows {kinds}, {len(data['states']['own']['graphic'])} states, "
          f"{len(finals)} matches; train {len(train_rows)} held-out {len(hold_rows)} ({time.time() - t0:.0f}s)", flush=True)
    shard = shard_rows = None
    if args.shard_frac > 0:
        shard = open_cache(Path(args.shard_cache))
        shard_rows = np.flatnonzero(np.asarray(shard["split"]) == SPLITS.index("train"))

    if args.antisym and args.final_loss == "quantile":
        raise SystemExit("--antisym with quantile heads is not supported")
    paired = args.antisym or args.pair_weight > 0
    nq = args.n_quantiles if args.final_loss == "quantile" else None
    net = load_value_net(args.init, device=device, n_quantiles=nq, antisym=args.antisym or None).train()
    target_net = load_value_net(args.init, device=device)
    if args.compile:  # cuda: the whole net; MPS: TRUNK_MODULES only (value/train.compile_value_net)
        compile_value_net(net, device)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    base = heldout_metrics(target_net, data, make_targets(target_net, data, device)[0], hold_rows, device)
    print(json.dumps({"iter": 0, "heldout": base}), flush=True)
    log.write(json.dumps({"iter": 0, "heldout": base}) + "\n")
    for it in range(args.start_iter, args.start_iter + args.iters):
        targets, info = make_targets(target_net, data, device)
        f = targets["own"]["final"]
        print(f"iter {it}: targets from {info['states']} next states in {info['seconds']:.0f}s; "
              f"final mean {f.mean() * 100:+.2f} (own view), |rem| mean {np.abs(targets['own']['remaining']).mean() * 100:.1f}",
              flush=True)
        params = [{"params": list(net.trunk_parameters()), "lr_scale": args.trunk_lr_scale},
                  {"params": net.head_parameters(), "lr_scale": 1.0}]
        opt = adamw(params, device, lr=args.lr, weight_decay=0.01)  # fused on cuda
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")
        sampler = Sampler(data, targets, train_rows, shard, shard_rows, args.batch_size, args.shard_frac,
                          args.seed * 1000 + it, group_roots=args.group_roots, device=device, paired=paired)
        totals, started = {}, time.time()
        for step in range(args.steps):
            lr = args.lr * (min(1.0, (step + 1) / args.warmup) if step < args.warmup else
                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup))))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            b = {k: v.to(device, non_blocking=True) for k, v in sampler.next().items()}  # pinned on cuda
            ctx = torch.autocast(device.type, dtype=amp) if amp else contextlib.nullcontext()
            net.train()
            with ctx:
                out, qs = net.forward_full_u8(b["graphic"], b["team_state"], b["agent_states"])
                if paired:
                    out2, qs2 = net.forward_full_u8(b["graphic_b"], b["team_state_b"], b["agent_states_b"])
            lead_b = b["team_state"][:, 0] - b["team_state"][:, 1]
            if args.antisym:
                out = antisym_outputs(out, out2)
            loss, stats = soft_value_loss(out, b, lead_b, args.final_beta, args.final_loss, qs)
            if paired and not args.antisym:  # explicit symmetry augmentation: both views + consistency
                tb = {k: b[k + "_b"] for k in ("delta", "delta_mask", "final", "p")}
                loss2, _ = soft_value_loss(out2, tb, -lead_b, args.final_beta, args.final_loss, qs2)
                h = out.shape[1] - 4
                cons = (0.5 * (out[:, h].float() + out2[:, h].float()) ** 2 / MSE_DEN).mean()
                loss = (loss + loss2) / 2 + args.pair_weight * cons
                stats["pair_cons"] = cons.detach()
            if args.rank_weight > 0:
                rl = rank_loss(out, b, lead_b)
                loss = loss + args.rank_weight * rl
                stats["rank"] = rl.detach()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            for k, v in stats.items():
                totals[k] = totals[k] + v if k in totals else v.clone()
            if (step + 1) % args.log_every == 0:
                tr = {k: round(float(v) / args.log_every, 4) for k, v in totals.items()}
                totals = {}
                print(json.dumps({"iter": it, "step": step + 1, "t": round(time.time() - started), "train": tr}), flush=True)
        sampler.stop()
        # the next round's target net = this net (uncompiled copy)
        target_net = ValueNet(hidden_size=net.trunk.hidden_size, n_horizons=net.n_horizons,
                              n_quantiles=net.n_quantiles, antisym=net.antisym).to(device)
        target_net.load_state_dict(_strip_compile(net.state_dict()))
        target_net.eval()
        hm = heldout_metrics(target_net, data, targets, hold_rows, device)
        entry = {"iter": it, "train_seconds": round(time.time() - started), "heldout": hm}
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        save_value_net(out_dir / f"iter{it}.pt", target_net,
                       {"iter": it, "init": str(args.init), "data": str(args.data), "shard_frac": args.shard_frac,
                        "steps": args.steps, "final_loss": args.final_loss, "final_beta": args.final_beta,
                        "pair_weight": args.pair_weight, "heldout": hm})
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------ S1.5 regen


def _regen_worker(job):
    """Replays one S1.5 root match and its roots' branches (value_branch_eval code paths, no
    ground-truth rollouts: those are stored) and returns the encoded views S1.5 scored."""
    os.chdir(job["cwd"])  # P10's ckpt: descriptor is relative to the main checkout
    import value_branch_eval as vb
    from blackout_env.value.evaluate import encode_view, team_view

    vb._init_threads()
    task, root_ticks = job["task"], job["root_ticks"]
    team = task["learner_team"]
    snaps = {}

    def on_tick(tick, sim, opp, shadow):
        if tick in root_ticks:
            snaps[tick] = (sim.copy(), vb.clone_opponent(task["opponent"], opp),
                           {k: copy.deepcopy(v) for k, v in shadow.items()})

    vb.play_root_match(task, stop_tick=max(root_ticks) + 1, on_tick=on_tick, shadows=True)
    out = []
    for tick in root_ticks:
        sim_root, opp_root, shadows = snaps[tick]
        rng_seed = zlib.crc32(f"{task['id']}:{tick}".encode()) % 1_000_000
        gt, views = vb.run_root(sim_root, task["opponent"], opp_root, shadows, team, tick, rng_seed, 0)
        views = views + [(("root",), sim_root.obs())]
        enc = []
        for key, obs in views:
            enc.append(("|".join(map(str, key)), encode_view(team_view(obs, team)), encode_view(team_view(obs, 1 - team))))
        hashes = [br["windows"][str(vb.WINDOWS[-1])]["pos_hash"] for br in gt["branches"]]
        out.append({"match": task["id"], "tick": tick, "hashes": hashes, "views": enc})
    return out


def cmd_regen(args):
    s15 = Path(args.s15)
    jobs = []
    for f in sorted((s15 / "raw").glob("*.json")):
        r = json.loads(f.read_text())
        jobs.append({"task": r["task"], "root_ticks": sorted(x["tick"] for x in r["roots"]), "cwd": str(args.cwd)})
    print(f"{len(jobs)} matches, {sum(len(j['root_ticks']) for j in jobs)} roots", flush=True)
    keys, root_of, enc = [], [], {v: [[], [], []] for v in VIEWS}
    hashes = {}
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_regen_worker, j) for j in sorted(jobs, key=lambda j: -max(j["root_ticks"]))]
        for i, fut in enumerate(as_completed(futs)):
            for root in fut.result():
                rid = f"{root['match']}@{root['tick']}"
                hashes[rid] = root["hashes"]
                for key, own, mir in root["views"]:
                    keys.append(key)
                    root_of.append(rid)
                    for v, e in (("own", own), ("mir", mir)):
                        for j in range(3):
                            enc[v][j].append(e[j])
            if i % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}] {time.time() - started:.0f}s", flush=True)
    arrays = {f"{v}_{k}": np.stack(enc[v][j]) for v in VIEWS for j, k in enumerate(INPUTS)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keys=np.array(keys), root=np.array(root_of), hashes=json.dumps(hashes), **arrays)
    print(f"{len(keys)} views in {time.time() - started:.0f}s -> {args.out}", flush=True)


# ------------------------------------------------------------------------------ S1.5 score


def preds_for(net_path, views, device) -> dict:
    """value_branch_eval.evaluate_views' per-view prediction dicts for every regenerated view."""
    from blackout_env.value.baselines import load_models, predict_set
    from blackout_env.value.features import hand_features
    from blackout_env.value.model import load_value_net

    net = load_value_net(net_path, device=device)
    st = {v: {k: views[f"{v}_{k}"] for k in INPUTS} for v in VIEWS}
    idx = np.arange(len(views["keys"]))
    po = predict_states(net, st["own"], idx, device)
    pm = predict_states(net, st["mir"], idx, device)
    models = load_models(MAIN / "reports/value_s1/baseline_models.pt")
    hm = predict_set("hand_mlp", models, hand_features(st["own"]["graphic"], st["own"]["team_state"], st["own"]["agent_states"]))
    lead = st["own"]["team_state"][:, 0] - st["own"]["team_state"][:, 1]
    out = {}
    for i, (rid, key) in enumerate(zip(views["root"], views["keys"])):
        out.setdefault(str(rid), {})[str(key)] = {
            "lead": float(lead[i] * 100), "v_final": float(po["final"][i] * 100), "m_final": float(pm["final"][i] * 100),
            "v_d": [float(x * 100) for x in po["delta"][i]], "m_d": [float(x * 100) for x in pm["delta"][i]],
            "v_p": [float(x) for x in po["p"][i]], "m_p": [float(x) for x in pm["p"][i]],
            "h_final": float(hm["final"][i] * 100), "h_d20": float(hm["delta"][i][-1] * 100), "h_pwin": float(hm["p_win"][i])}
    return out


def load_s15_roots(s15: Path) -> list[dict]:
    roots = []
    for f in sorted((s15 / "raw").glob("*.json")):
        roots += json.loads(f.read_text())["roots"]
    return roots


SCORE_PREDICTORS = ("lead", "v_k1", "sym_k1", "sym_k10", "lead_roll125", "v_roll125", "sym_roll125")


def cmd_score(args):
    import torch

    import value_branch_eval as vb
    from blackout_env.train.accel import pick_device

    device = pick_device(args.device)
    views = dict(np.load(args.views))
    hashes = json.loads(str(views["hashes"]))
    roots = load_s15_roots(Path(args.s15))
    # determinism check: regenerated window-end state hashes == stored
    bad = 0
    for r in roots:
        rid = f"{r['match']}@{r['tick']}"
        stored = [br["windows"][str(vb.WINDOWS[-1])]["pos_hash"] for br in r["branches"]]
        bad += int(hashes.get(rid) != stored)
    print(f"{len(roots)} roots, hash mismatches {bad}", flush=True)
    results = {"hash_mismatch_roots": bad}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        preds = preds_for(path, views, device)
        rs = []
        diffs = []
        for r in roots:
            rid = f"{r['match']}@{r['tick']}"
            new = copy.deepcopy(r)
            if name == "s1b":  # check: S1b on the regenerated views == S1b's stored predictions
                for k, p in r["preds"].items():
                    if k in preds[rid]:
                        diffs.append(abs(p["v_final"] - preds[rid][k]["v_final"]))
            new["preds"] = preds[rid]
            rs.append(new)
        if diffs:
            print(f"{name}: |stored - regenerated| S1b final max {max(diffs):.4f} pts over {len(diffs)} views", flush=True)
            results["s1b_regen_max_abs_diff"] = max(diffs)
        per = {}
        for bi, bname in enumerate(vb.BUCKET_NAMES + ("all",)):
            sel = [r for r in rs if bname == "all" or vb.bucket_of(r["t_s"]) == bi]
            ms = [vb.root_metrics(r, vb.WINDOWS[-1]) for r in sel]
            per[bname] = vb.summarize(ms, sel)
        results[name] = per
        print(format_table(name, per), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"score_{args.tag}.json").write_text(json.dumps(results, indent=1, default=float))
    (out / f"score_{args.tag}.txt").write_text("\n".join(format_table(n, results[n]) for n in results
                                                       if isinstance(results[n], dict)))


def format_table(name: str, per: dict) -> str:
    lines = [f"\n##### {name}"]
    lines.append(f"{'bucket':8s} {'predictor':13s} {'conc>0':>7s} {'>=5':>7s} {'>2se':>7s} {'rho':>6s} {'regret':>7s} "
                 f"{'gain':>12s} {'share':>6s} {'calib':>7s}")
    for bname, s in per.items():
        oracle = s["gain_vs_no_search"]["oracle_cv"][0]
        for n in SCORE_PREDICTORS:
            c = s["concordance"].get("final", {}).get(n, {})
            g = s["gain_vs_no_search"].get(n, [np.nan, np.nan])
            rho = s["spearman"].get("final", {}).get(n, [None])[0]
            cal = s["calib_v23_branch"].get(n, {}).get("bias", np.nan)
            lines.append(f"{bname:8s} {n:13s} {c.get('>=0', [np.nan])[0]:7.3f} {c.get('>=5', [np.nan])[0]:7.3f} "
                         f"{c.get('>2se', [np.nan])[0]:7.3f} {('-' if rho is None else f'{rho:.3f}'):>6s} "
                         f"{s['regret'].get(n, np.nan):7.2f} {g[0]:+6.2f}±{g[1]:.2f} {g[0] / oracle if oracle else np.nan:6.2f} "
                         f"{cal:+7.1f}")
        lines.append(f"{bname:8s} {'oracle_cv':13s} {'':7s} {'':7s} {'':7s} {'':6s} {s['regret']['oracle_cv']:7.2f} "
                     f"{oracle:+6.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ bias


def cmd_bias(args):
    import torch

    from blackout_env.sim import SimState
    from blackout_env.value.evaluate import ValueEvaluator

    res = {}
    obs = [SimState.from_seed(s).obs() for s in args.seeds]
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        ev = ValueEvaluator(path, device=args.device)
        a = ev.evaluate_obs(obs, 0)
        b = ev.evaluate_obs(obs, 1)
        res[name] = {"final_A": [round(float(x) * 100, 1) for x in a["final"]],
                     "final_B": [round(float(x) * 100, 1) for x in b["final"]],
                     "pwin_A": [round(float(x), 3) for x in a["p_win"]], "pwin_B": [round(float(x), 3) for x in b["p_win"]],
                     "mean_final_A": float(a["final"].mean() * 100), "mean_final_B": float(b["final"].mean() * 100),
                     "mean_pwin_A": float(a["p_win"].mean()), "mean_pwin_B": float(b["p_win"].mean())}
        print(name, json.dumps(res[name]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    del torch


# ------------------------------------------------------------------------------ diag (docs §27)

DIAG_BUCKETS = ((0, 10), (10, 30), (30, 60), (60, 150), (150, 1e9))


def _bucket_names():
    return [f"{a}-{b if b < 1e8 else ''}s" for a, b in DIAG_BUCKETS]


def _calib(pred, true) -> dict:
    """pred / true in points: bias, MAE, R^2 (vs the set's own mean), slope of true on pred."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) < 2:
        return {"n": int(len(pred))}
    var = np.var(true)
    cov = np.mean((pred - pred.mean()) * (true - true.mean()))
    return {"n": int(len(pred)), "bias": float(np.mean(pred - true)), "mae": float(np.mean(np.abs(pred - true))),
            "r2": float(1 - np.mean((pred - true) ** 2) / var) if var > 0 else float("nan"),
            "slope": float(cov / np.var(pred)) if np.var(pred) > 0 else float("nan")}


def _binned(pred, true, edges=(-200, -60, -30, -10, 10, 30, 60, 200)) -> list:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pred >= a) & (pred < b)
        if m.sum():
            out.append({"bin": [a, b], "n": int(m.sum()), "pred": float(pred[m].mean()), "true": float(true[m].mean())})
    return out


def diag_net(net_path, data, rows_sel, device, start_obs=None) -> dict:
    """Bias / calibration diagnosis of one net on simulator rows (docs §27):

    * defect = (V_own(s) + V_mir(s)) / 2 of the final lead (pts) -- the part of a single-view
      reading that a symmetric reading removes; 0 for a perfectly antisymmetric V -- per time bucket
    * on-trajectory rows (V23 vs its opponent, true match final F known): own view vs F, mirror
      view vs -F, symmetric reading vs F (bias, MAE, R^2, slope), per bucket + binned calibration
    * the exact 2 s / 5 s lead changes vs the delta heads (R^2), all rows
    * match-start states from both sides (``start_obs``): mean final lead / P(win)."""
    from blackout_env.value.evaluate import symmetrize
    from blackout_env.value.model import load_value_net

    net = load_value_net(net_path, device=device)
    rows = data["rows"]
    st = rows["idx"][rows_sel]
    own = predict_states(net, data["states"]["own"], st, device)
    mir = predict_states(net, data["states"]["mir"], st, device)
    sym = symmetrize(own, mir)
    t = rows["t_s"][rows_sel]
    kind = rows["kind"][rows_sel]
    true = rows["match_final"][rows_sel].astype(float)
    lead = rows["lead"][rows_sel].astype(float)
    res = {"antisym_net": bool(getattr(net, "antisym", False)), "n_quantiles": int(getattr(net, "n_quantiles", 0)),
           "buckets": {}}
    names = _bucket_names() + ["all"]
    for bi, name in enumerate(names):
        m = np.ones(len(t), bool) if name == "all" else (t >= DIAG_BUCKETS[bi][0]) & (t < DIAG_BUCKETS[bi][1])
        d = (own["final"][m] + mir["final"][m]) / 2 * 100
        tr = m & (kind == 0)
        e = {"n": int(m.sum()), "defect_mean": float(d.mean()) if len(d) else float("nan"),
             "defect_abs": float(np.abs(d).mean()) if len(d) else float("nan"),
             "pwin_defect": float(np.mean(own["p"][m, 2] + mir["p"][m, 2] - 1)) if m.any() else float("nan"),
             "traj_own": _calib(own["final"][tr] * 100, true[tr]),
             "traj_mir": _calib(mir["final"][tr] * 100, -true[tr]),
             "traj_sym": _calib(sym["final"][tr] * 100, true[tr]),
             "traj_lead": _calib(lead[tr], true[tr])}
        d2 = (rows["lead2"][rows_sel] - rows["lead"][rows_sel]).astype(float)
        d5 = (rows["lead5"][rows_sel] - rows["lead"][rows_sel]).astype(float)
        e["d2_sym"] = _calib(sym["delta"][m, H2] * 100, d2[m])
        e["d5_sym"] = _calib(sym["delta"][m, H5] * 100, d5[m])
        res["buckets"][name] = e
    tr = kind == 0
    res["binned_own"] = _binned(own["final"][tr] * 100, true[tr])
    res["binned_sym"] = _binned(sym["final"][tr] * 100, true[tr])
    if start_obs is not None:
        from blackout_env.value.evaluate import ValueEvaluator

        ev = ValueEvaluator(net, device=str(device), batch=256)
        a, b = ev.evaluate_obs(start_obs, 0), ev.evaluate_obs(start_obs, 1)
        res["start"] = {"n": len(start_obs), "final_A": float(a["final"].mean() * 100), "final_B": float(b["final"].mean() * 100),
                        "final_A_sd": float(a["final"].std() * 100), "pwin_A": float(a["p_win"].mean()),
                        "pwin_B": float(b["p_win"].mean()),
                        "sym_abs": float(np.abs(a["final"] - b["final"]).mean() / 2 * 100)}
    return res


def cmd_diag(args):
    from blackout_env.sim import SimState
    from blackout_env.train.accel import pick_device

    device = pick_device(args.device)
    data = load_raw(Path(args.data) / "raw")
    rows = data["rows"]
    rows["match_final"] = np.array([data["metas"][m]["final"] for m in rows["match"]], np.float32)
    hold = holdout_mask(rows["seed"])
    sets = {"heldout": np.flatnonzero(hold), "all": np.arange(len(hold))}
    start_obs = [SimState.from_seed(s).obs() for s in range(args.start_seed, args.start_seed + args.n_start)]
    out = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        out[name] = {k: diag_net(path, data, r, device, start_obs if k == "heldout" else None) for k, r in sets.items()}
        print(format_diag(name, out[name]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_suffix(".txt").write_text("\n".join(format_diag(n, r) for n, r in out.items()))


def format_diag(name, r) -> str:
    lines = [f"\n##### {name}"]
    st = r["heldout"].get("start")
    if st:
        lines.append(f"start ({st['n']} seeds): V_A {st['final_A']:+.1f} V_B {st['final_B']:+.1f} "
                     f"P(win) {st['pwin_A']:.2f}/{st['pwin_B']:.2f}  |A-B|/2 {st['sym_abs']:.1f}")
    for split in ("heldout", "all"):
        lines.append(f"[{split}] bucket   n      defect |defect| pwinDef | own bias/MAE/R2      | mir bias/MAE/R2      "
                     f"| sym bias/MAE/R2/slope    | lead R2 | d5 R2")
        for bn, e in r[split]["buckets"].items():
            o, m_, s_ = e["traj_own"], e["traj_mir"], e["traj_sym"]
            if o.get("n", 0) < 2:
                continue
            lines.append(f"  {bn:8s} {e['n']:6d} {e['defect_mean']:+7.1f} {e['defect_abs']:7.1f} {e['pwin_defect']:+6.2f} | "
                         f"{o['bias']:+6.1f} {o['mae']:5.1f} {o['r2']:5.2f} | {m_['bias']:+6.1f} {m_['mae']:5.1f} {m_['r2']:5.2f} | "
                         f"{s_['bias']:+6.1f} {s_['mae']:5.1f} {s_['r2']:5.2f} {s_['slope']:4.2f} | {e['traj_lead']['r2']:5.2f} | "
                         f"{e['d5_sym'].get('r2', float('nan')):5.2f}")
    lines.append("binned sym (traj, all): " + "  ".join(f"[{b['bin'][0]},{b['bin'][1]}) n{b['n']} {b['pred']:+.0f}->{b['true']:+.0f}"
                                               for b in r["all"]["binned_sym"]))
    lines.append("binned own (traj, all): " + "  ".join(f"[{b['bin'][0]},{b['bin'][1]}) n{b['n']} {b['pred']:+.0f}->{b['true']:+.0f}"
                                               for b in r["all"]["binned_own"]))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=REPO / "datasets/value_s2v")
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--init", type=Path, default=MAIN / "checkpoints/value/frozen/s1b_value.pt")
    t.add_argument("--shard-cache", type=Path, default=MAIN / "datasets/value_s1_r64")
    t.add_argument("--shard-frac", type=float, default=0.25)
    t.add_argument("--iters", type=int, default=4)
    t.add_argument("--start-iter", type=int, default=1, help="first iteration number (resuming from --init = iter k-1)")
    t.add_argument("--group-roots", type=int, default=0, help="roots per batch with all their branches (rank term)")
    t.add_argument("--rank-weight", type=float, default=0.0)
    t.add_argument("--final-beta", type=float, default=None, help="Huber beta of the final head (tenths of pts)")
    t.add_argument("--final-loss", choices=["huber", "mse", "quantile"], default="huber")
    t.add_argument("--n-quantiles", type=int, default=32)
    t.add_argument("--antisym", action="store_true", help="train V(s) = (f(s) - f(mirror s)) / 2 (both views per row)")
    t.add_argument("--pair-weight", type=float, default=0.0,
                   help="> 0: both views per row + this weight x consistency (rem(s) + rem(mirror s))^2")
    t.add_argument("--steps", type=int, default=4000, help="per iteration")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--trunk-lr-scale", type=float, default=1.0)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    t.add_argument("--amp", choices=["none", "bf16", "fp16"], default="fp16")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("regen")
    r.add_argument("--s15", type=Path, default=MAIN / "reports/value_s1/branch")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--cwd", type=Path, required=True, help="dir whose checkpoints/ resolves P10 (symlink to main)")
    r.set_defaults(fn=cmd_regen)
    s = sub.add_parser("score")
    s.add_argument("--views", type=Path, required=True)
    s.add_argument("--s15", type=Path, default=MAIN / "reports/value_s1/branch")
    s.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tag", default="run")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_score)
    b = sub.add_parser("bias")
    b.add_argument("--ckpt", nargs="+", required=True)
    b.add_argument("--seeds", type=int, nargs="+", default=[950000, 950001, 950002, 960000, 960001, 960002, 1, 2])
    b.add_argument("--device", default="cpu")
    b.add_argument("--out", type=Path, default=None)
    b.set_defaults(fn=cmd_bias)
    d = sub.add_parser("diag", help="bias / calibration diagnosis on the simulator rows (docs §27)")
    d.add_argument("--data", type=Path, default=REPO / "datasets/value_s2v")
    d.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    d.add_argument("--out", type=Path, required=True, help=".json (+ .txt next to it)")
    d.add_argument("--start-seed", type=int, default=970000)
    d.add_argument("--n-start", type=int, default=64)
    d.add_argument("--device", default="auto")
    d.set_defaults(fn=cmd_diag)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
