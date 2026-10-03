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
    VIEWS, bootstrap_targets, holdout_mask, load_raw, mirror_targets)
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


def soft_value_loss(out, t: dict, lead, final_beta=None):
    """value_loss with a soft result target ``t['p']`` [B, 3] and per-row weights."""
    import torch
    from torch.nn import functional as F

    from blackout_env.value.model import HUBER_BETA, REG_SCALE

    out = out.float()
    h = out.shape[1] - 4
    mask = t["delta_mask"].float()
    reg = F.smooth_l1_loss(out[:, :h], t["delta"].float() * REG_SCALE, beta=HUBER_BETA, reduction="none")
    per_h = (reg * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    delta_loss = per_h.mean()
    # final_beta: Huber beta of the final-lead head (tenths of points).  The default 1 (10 pts) is
    # nearly L1, whose minimiser on a bimodal outcome (e.g. a symmetric opening: +80 or -80) is
    # anything in between -- not the mean; a large beta (-> MSE) targets the mean.
    final_loss = F.smooth_l1_loss(out[:, h], (t["final"].float() - lead.float()) * REG_SCALE,
                                  beta=HUBER_BETA if final_beta is None else final_beta)
    ce = -(t["p"].float() * torch.log_softmax(out[:, h + 1:], dim=-1)).sum(-1).mean()
    loss = delta_loss + final_loss + ce
    return loss, {"loss": loss.detach(), "delta": delta_loss.detach(), "final": final_loss.detach(), "ce": ce.detach()}


# ------------------------------------------------------------------------------ train


class Sampler:
    """Background batches: simulator rows (random view, antisymmetric target) + shard rows."""

    def __init__(self, data, targets, rows, shard, shard_rows, batch, shard_frac, seed, depth=6, group_roots=0,
                 device=None):
        self.data, self.targets, self.rows = data, targets, rows
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

    net = load_value_net(args.init, device=device).train()
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
                          args.seed * 1000 + it, group_roots=args.group_roots, device=device)
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
                out = net.forward_u8(b["graphic"], b["team_state"], b["agent_states"])
            lead_b = b["team_state"][:, 0] - b["team_state"][:, 1]
            loss, stats = soft_value_loss(out, b, lead_b, args.final_beta)
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
        target_net = ValueNet(hidden_size=net.trunk.hidden_size, n_horizons=net.n_horizons).to(device)
        target_net.load_state_dict(_strip_compile(net.state_dict()))
        target_net.eval()
        hm = heldout_metrics(target_net, data, targets, hold_rows, device)
        entry = {"iter": it, "train_seconds": round(time.time() - started), "heldout": hm}
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        save_value_net(out_dir / f"iter{it}.pt", target_net,
                       {"iter": it, "init": str(args.init), "data": str(args.data), "shard_frac": args.shard_frac,
                        "steps": args.steps, "heldout": hm})
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
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
