"""Offline critic bench (2026-11-05): which critic set-up predicts the outcome best, measured on branch-rollout truth.

1. evalset   -- examples/value_branch_diag.py ... --save DIR/eval_<opp>.npz  (roots' learner views, opponent, T, mu, var)
2. collect   -- PPOTrainer.collect() with frozen weights (lr = critic_lr = 0), the learner's sampled play vs the RL pool,
               stop_exhausted; a share of each game's rows is kept with its Monte-Carlo target R (PPO's final reward) once
               the game ends; -> DIR/train_*.npz
3. train     -- fit a critic variant on the MC targets (PPO's loss / batch / epochs per "iteration" of 32k rows) and
               report, every few epochs, R2 / bias on every eval set (by opponent and T).

    PYTHONPATH=. python examples/critic_bench.py collect --out DIR --iters 120 [rl13 config flags]
    PYTHONPATH=. python examples/critic_bench.py train --data DIR --variant base|lr3|lr10|hl|opp|trunk ... --out DIR/res_<v>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("BLACKOUT_TPD", "1")
LAM, SCALE = 0.2, 30.0


def reward(lead):
    lead = np.asarray(lead, np.float64)
    return np.sign(lead) + LAM * np.tanh(lead / SCALE)


# ---------------------------------------------------------------------------------------------------------------- collect
def cmd_collect(a):
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    pool = json.load(open(a.pool))
    cfg = PPOConfig(init=a.init, value=a.value, device=a.device, opponents=[d for d, _ in pool],
                    opp_weights=[float(w) for _, w in pool], pfsp="hard", games=a.games, segment=a.segment,
                    heur_workers=a.heur_workers, lr=0.0, critic_lr=0.0, critic_warmup=10 ** 9, stop_exhausted=True,
                    league=False, async_collect="false", seed=a.seed, critic_fp16=True)
    tr = PPOTrainer(cfg)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pend: dict[int, list] = {}            # game slot -> kept rows (g, t, a, opp, sec) of the game in flight
    out = {"g": [], "t": [], "a": [], "opp": [], "sec": [], "R": []}
    shard = 0

    def hook(d):
        nonlocal shard
        keep = rng.random(len(d["game"])) < a.p_store
        for b in np.unique(d["game"][keep]):
            rows = np.nonzero(keep & (d["game"] == b))[0]
            sec = (1.0 - d["t"][rows, 2]) * 420.0
            pend.setdefault(int(b), []).append((d["g"][rows], d["t"][rows], d["a"][rows],
                                                np.full(len(rows), d["opp_of"][b], np.int16), sec.astype(np.float32)))
        for b in np.nonzero(d["over"])[0]:
            parts = pend.pop(int(b), [])
            if not parts:
                continue
            R = float(d["final_r"][b])
            for g, t, aa, o, s in parts:
                out["g"].append(g); out["t"].append(t); out["a"].append(aa); out["opp"].append(o); out["sec"].append(s)
                out["R"].append(np.full(len(g), R, np.float32))

    tr.diag_hook = hook
    t0 = time.time()
    for it in range(a.iters):
        tr.collect()
        n = sum(len(x) for x in out["R"])
        if n >= a.shard_rows or it == a.iters - 1:
            np.savez(os.path.join(a.out, f"train_{shard:03d}.npz"), **{k: np.concatenate(v) for k, v in out.items()})
            print(f"iter {it}: shard {shard} rows {n} ({time.time() - t0:.0f}s)", flush=True)
            shard += 1
            for v in out.values():
                v.clear()
    json.dump({"opponents": [d for d, _ in pool]}, open(os.path.join(a.out, "opps.json"), "w"))


# ------------------------------------------------------------------------------------------------------------------ train
def load_train(d, max_rows):
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(d, "train_*.npz")))]
    x = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len(x["R"]) > max_rows:
        idx = np.random.default_rng(0).choice(len(x["R"]), max_rows, replace=False)
        x = {k: v[idx] for k, v in x.items()}
    return x


def load_evals(d, opp_names):
    ev = []
    for f in sorted(glob.glob(os.path.join(d, "eval_*.npz"))):
        e = dict(np.load(f, allow_pickle=True))
        name = str(e["opponent"])
        oi = opp_names.index(name) if name in opp_names else -1
        ev.append((os.path.basename(f)[5:-4], oi, e))
    return ev


def build_critic(v, value_path, init_policy, dev, n_opp):
    from blackout_env.value.model import load_value_net

    cr = load_value_net(value_path, device=dev)
    if v.get("trunk") and init_policy:
        cr.load_trunk(torch.load(init_policy, map_location=dev, weights_only=True)["policy_state"])
    if v.get("opp"):
        cr.enable_opp(max(n_opp, 1) + 8)
    if v.get("hl"):
        cr.n_bins = 101
        cr.bin_out = torch.nn.Linear(cr.head[-1].in_features, 101).to(dev)
        e = torch.linspace(-1.4, 1.4, 102, device=dev)
        cr.register_buffer("bin_centers", (e[:-1] + e[1:]) / 2, persistent=False)
        cr._hl_edges = e
    return cr


def value_of(cr, g, t, a, o, hl):
    from blackout_env.sim.batch import unpack_graphic
    from blackout_env.value.model import REG_SCALE

    g = unpack_graphic(g) if g.dim() == 3 else g
    with torch.autocast(g.device.type, dtype=torch.float16, enabled=g.device.type == "cuda"):
        out, lg = cr.forward_full(g, t, a, opp=o)
    out = out.float()
    if hl:
        return (torch.softmax(lg.float(), -1) * cr.bin_centers).sum(-1), lg.float()
    h = out.shape[1] - 4
    lead = (t[:, 0] - t[:, 1]).float() * 100.0
    final = lead + out[:, h] / REG_SCALE * 100.0
    p = torch.softmax(out[:, h + 1:], -1)
    return (p[:, 2] - p[:, 0]) + LAM * torch.tanh(final / SCALE), None


@torch.no_grad()
def evaluate(cr, ev, dev, hl, use_opp):
    cr.eval()
    res = {}
    for name, oi, e in ev:
        n = len(e["mu"])
        v = []
        for s in range(0, n, 512):
            to = lambda k: torch.from_numpy(np.ascontiguousarray(e[k][s:s + 512])).to(dev)
            o = torch.full((min(512, n - s),), max(oi, 0), device=dev, dtype=torch.long) if use_opp and oi >= 0 else None
            v.append(value_of(cr, to("pk"), to("tm"), to("am"), o, hl)[0].cpu().numpy())
        v = np.concatenate(v).astype(np.float64)
        for T in np.unique(e["T"]):
            m = e["T"] == T
            mu, irr, k = e["mu"][m], e["var_w"][m].mean(), int(e["k"])
            expl = mu.var(ddof=1) - irr / k
            mse = np.mean((v[m] - mu) ** 2) - irr / k
            res[f"{name}@{T:g}"] = dict(r2=float(1 - mse / expl), bias=float((v[m] - mu).mean()), n=int(m.sum()))
    cr.train()
    return res


def hl_target(x, e, sigma):
    x = x.float().clamp(float(e[0]), float(e[-1]))
    cdf = torch.special.erf((e[None] - x[:, None]) / (sigma * 2 ** 0.5))
    m = (cdf[:, 1:] - cdf[:, :-1]) / 2
    return m / m.sum(1, keepdim=True).clamp_min(1e-12)


VARIANTS = {
    "base": dict(lr=1e-5),
    "lr3": dict(lr=3e-5),
    "lr10": dict(lr=1e-4),
    "hl": dict(lr=1e-5, hl=True),
    "hl_lr3": dict(lr=3e-5, hl=True),
    "opp": dict(lr=1e-5, opp=True),
    "trunk": dict(lr=1e-5, trunk=True),
    "trunk_lr3": dict(lr=3e-5, trunk=True),
}


def cmd_train(a):
    dev = torch.device(a.device)
    v = VARIANTS[a.variant]
    opp_names = json.load(open(os.path.join(a.data, "opps.json")))["opponents"]
    x = load_train(a.data, a.max_rows)
    ev = load_evals(a.evals or a.data, opp_names)
    cr = build_critic(v, a.value, a.init_policy, dev, len(opp_names))
    if a.critic_init and not v.get("trunk"):
        sd = torch.load(a.critic_init, map_location=dev, weights_only=True)
        cr.load_state_dict({k: s for k, s in sd.items() if k in cr.state_dict()}, strict=False)
    hl = bool(v.get("hl"))
    opt = torch.optim.AdamW(cr.parameters(), lr=v["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10 ** 9, enabled=dev.type == "cuda")
    n = len(x["R"])
    G = {k: torch.from_numpy(x[k]) for k in ("g", "t", "a", "R", "opp")}
    sigma = 0.75 * 2.8 / 101
    log = {"variant": a.variant, "rows": n, "evals": []}
    r0 = evaluate(cr, ev, dev, hl, v.get("opp"))
    log["evals"].append({"rows_seen": 0, **r0})
    print(a.variant, "start", {k: round(r["r2"], 3) for k, r in r0.items()}, flush=True)
    seen, nxt = 0, a.eval_every
    cr.train()
    rng = np.random.default_rng(0)
    t0 = time.time()
    while seen < a.rows_total:
        idx = torch.from_numpy(rng.choice(n, a.minibatch, replace=False))
        to = lambda k: G[k][idx].to(dev, non_blocking=True)
        o = to("opp").long() if v.get("opp") else None
        val, lg = value_of(cr, to("g"), to("t"), to("a"), o, hl)
        R = to("R")
        loss = -(hl_target(R, cr._hl_edges, sigma) * torch.log_softmax(lg, -1)).sum(-1).mean() if hl \
            else F.smooth_l1_loss(val, R, beta=0.1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(cr.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        seen += a.minibatch
        if seen >= nxt:
            r = evaluate(cr, ev, dev, hl, v.get("opp"))
            log["evals"].append({"rows_seen": seen, **r})
            print(a.variant, seen, f"{time.time() - t0:.0f}s", {k: round(q["r2"], 3) for k, q in r.items()}, flush=True)
            nxt += a.eval_every
            json.dump(log, open(a.out, "w"))
    json.dump(log, open(a.out, "w"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--init", required=True)
    c.add_argument("--value", required=True)
    c.add_argument("--pool", required=True, help="pool json [[descriptor, weight], ...]")
    c.add_argument("--iters", type=int, default=120)
    c.add_argument("--games", type=int, default=256)
    c.add_argument("--segment", type=int, default=128)
    c.add_argument("--heur-workers", type=int, default=6)
    c.add_argument("--p-store", type=float, default=0.1)
    c.add_argument("--shard-rows", type=int, default=150_000)
    c.add_argument("--device", default="cuda")
    c.add_argument("--seed", type=int, default=11)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--evals", default="")
    t.add_argument("--variant", required=True, choices=list(VARIANTS))
    t.add_argument("--value", required=True)
    t.add_argument("--critic-init", default="", help="start from a PPO critic (_critic.pt) instead of the bare value net")
    t.add_argument("--init-policy", default="", help="policy checkpoint for the trunk variants")
    t.add_argument("--max-rows", type=int, default=2_000_000)
    t.add_argument("--rows-total", type=int, default=6_000_000, help="training rows to see (~ PPO: 64k per iteration)")
    t.add_argument("--minibatch", type=int, default=1024)
    t.add_argument("--eval-every", type=int, default=500_000)
    t.add_argument("--device", default="cuda")
    t.add_argument("--out", required=True)
    a = p.parse_args()
    {"collect": cmd_collect, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
