"""Offline diagnostic: how does the PPO critic (trained against the fixed pool) behave against other opponents?

Collect  (GPU, no parameter updates, PPOTrainer.collect() verbatim + a read-only hook):
    PYTHONPATH=. python examples/critic_shift_diag.py collect --out runs/csd [--target 200]
Analyse  (CPU, numpy/matplotlib, reads the collect output):
    python examples/critic_shift_diag.py analyze --data runs/csd --out reports/rl/critic_shift_diag
Critic-only fine-tune simulation (GPU):
    PYTHONPATH=. python examples/critic_shift_diag.py finetune --data runs/csd --out reports/rl/critic_shift_diag

See reports/rl/critic_shift_diag.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

F = "checkpoints/distill/frozen"
R = "checkpoints/rl"
OPPS = {
    "p24c_v321": f"net:{F}/p24c_v321.pt",
    "p22_v31": f"net:{F}/p22_v31.pt",
    "V32.1": "V32.1",
    "r2reg_final": f"net:{R}/ppo_r2_reg/final.pt",
    "r2reg_it300": f"net:{R}/ppo_r2_reg/iter_300.pt",
    "r1_final": f"net:{R}/ppo_r1/final.pt",
    "r3b_final": f"net:{R}/grpo_r3b/final.pt",
}
EXTRA = {"p5a_v22": f"net:{F}/p5a_v22.pt", "p16_v27": f"net:{F}/p16_v27.pt"}   # held-out crushed opponents (--opps)
POOL = ["p24c_v321", "p22_v31", "V32.1"]
NEW = ["r2reg_final", "r2reg_it300", "r1_final", "r3b_final"]
NAMES = list(OPPS)
EPISODE_SECONDS = 420.0
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 1e9)]
BUCKET_NAMES = ["0-10", "10-20", "20-30", "30-45", ">45"]


# ------------------------------------------------------------------------------------------------ collect

def cmd_collect(a):
    import torch
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    os.makedirs(a.out, exist_ok=True)
    names = a.opps.split(",") if a.opps else NAMES
    specs = {**OPPS, **EXTRA}
    K = len(names)
    G = a.games
    cfg = PPOConfig(init=f"{R}/ppo_r2_reg/final.pt", opponents=[specs[n] for n in names], opp_weights=[1.0] * K, pfsp="none",
                    games=G, segment=128, critic_fp16=True, seed=a.seed)
    tr = PPOTrainer(cfg)
    tr.critic.load_state_dict(torch.load(a.critic, map_location=tr.dev, weights_only=True))
    tr.critic.eval()
    rng = np.random.default_rng(a.seed + 12345)
    # balanced, deterministic opponent / side assignment
    started = np.zeros(K, np.int64)
    side_cnt = np.zeros((K, 2), np.int64)

    def assign(b):
        o = int(np.argmin(started))
        s = int(np.argmin(side_cnt[o]))
        started[o] += 1
        side_cnt[o, s] += 1
        tr.trunk_opp[b], tr.trunk_side[b] = o, s

    uid = np.zeros(G, np.int64)
    next_uid = [a.seed * 10**6]
    for b in range(G):
        assign(b)
        uid[b] = next_uid[0]; next_uid[0] += 1
    ended_cnt = np.zeros(K, np.int64)
    cur: dict = {}

    def hook(d):
        game, g, t, ag, over, bs = d["game"], d["g"], d["t"], d["a"], d["over"], d["bsim"]
        n = len(game)
        # position of each row inside its game's segment (rows of a game are in time order)
        order = np.argsort(game, kind="stable")
        gs = game[order]
        first = np.r_[0, np.nonzero(np.diff(gs))[0] + 1]
        cnt = np.diff(np.r_[first, n])
        pos = np.empty(n, np.int32)
        pos[order] = np.arange(n) - np.repeat(first, cnt)
        last = np.zeros(n, bool)
        last[order[np.r_[first[1:], n] - 1]] = True
        rew = np.where(last & over[game], d["final_r"][game], 0.0)
        rec = dict(uid=uid[game].copy(), game=game.astype(np.int16), opp=d["opp_of"][game].astype(np.int8), side=d["side"][game].astype(np.int8),
                   pos=pos, elapsed=((1.0 - t[:, 2]) * EPISODE_SECONDS).astype(np.float32),
                   own_score=(t[:, 0] * 100).astype(np.float32), opp_score=(t[:, 1] * 100).astype(np.float32),
                   absph=t[:, 3].astype(np.float32), v=d["v"].astype(np.float32), adv=d["adv"].astype(np.float32),
                   ret=d["ret"].astype(np.float32), rew=rew.astype(np.float32))
        # segments whose views are kept for the critic fine-tune simulation
        present = np.unique(game)
        sel = present[rng.random(len(present)) < a.p_store]
        seg = {}
        if len(sel):
            rows = np.concatenate([np.nonzero(game == b)[0] for b in sel])
            seglen = np.array([(game == b).sum() for b in sel])
            seg.update(sg=g[rows], st=t[rows], sa=ag[rows], srow=rows, seglen=seglen, segb=sel.astype(np.int16),
                       seguid=uid[sel], segover=over[sel], segfin=d["final_r"][sel], segboot=d["boot"][sel],
                       segopp=d["opp_of"][sel].astype(np.int8), segside=d["side"][sel].astype(np.int8))
            live = sel[~over[sel]]
            from blackout_env.sim.batch import N_AGENT_COLS, H, W
            pk = np.zeros((len(live), 2, H, W), np.int16)
            tm = np.zeros((len(live), 2, 4), np.float32)
            am = np.zeros((len(live), 2, 10, N_AGENT_COLS), np.float32)
            if len(live):
                bs.render_packed(live, pk, tm, am)
                r = np.arange(len(live))
                s_ = d["side"][live]
                pk, tm, am = pk[r, s_], tm[r, s_], am[r, s_]
            seg.update(bootlive=live.astype(np.int16), bootg=pk, boott=tm, boota=am)
        ee = np.nonzero(over)[0]
        end = dict(euid=uid[ee], eopp=d["opp_of"][ee].astype(np.int8), eside=d["side"][ee].astype(np.int8),
                   elead=d["lead"][ee].astype(np.float32), efin=d["final_r"][ee].astype(np.float32))
        cur.clear()
        cur.update(rec=rec, seg=seg, end=end, ended=ee, opp_of=d["opp_of"].copy())

    tr.diag_hook = hook
    t0 = time.time()
    it = 0
    while True:
        batch, st = tr.collect()
        ee = cur["ended"]
        ep = {k: v for k, v in cur["rec"].items()}
        np.savez(os.path.join(a.out, f"it_{it:05d}.npz"), **ep, **cur["end"])
        if cur["seg"]:
            np.savez(os.path.join(a.out, f"seg_{it:05d}.npz"), **cur["seg"])
        for b in ee:
            ended_cnt[cur["opp_of"][b]] += 1
            assign(b)
            uid[b] = next_uid[0]; next_uid[0] += 1
        it += 1
        if it % 5 == 0 or it <= 3:
            print(f"it {it} t={time.time()-t0:.0f}s ended={ended_cnt.tolist()} v_mean={st['v_mean']:.3f} ev={st['expl_var']:.3f} "
                  f"abs_adv={st['abs_adv']:.4f}", flush=True)
        os.sync() if it % 20 == 0 else None
        if ended_cnt.min() >= a.target or (time.time() - t0) / 3600 > a.hours or it >= a.max_iters:
            break
    json.dump({"names": names, "ended": ended_cnt.tolist(), "iters": it, "secs": time.time() - t0,
               "args": {k: v for k, v in vars(a).items() if k != "fn"}}, open(os.path.join(a.out, "meta.json"), "w"))
    print("done", ended_cnt.tolist(), flush=True)
    os._exit(0)   # heuristic worker teardown noise


# ------------------------------------------------------------------------------------------------ analyse

def load_rows(path):
    fs = sorted(glob.glob(os.path.join(path, "it_*.npz")))
    cols, ends = {}, {}
    for f in fs:
        z = np.load(f)
        for k in z.files:
            (ends if k in ("euid", "eopp", "eside", "elead", "efin") else cols).setdefault(k, []).append(z[k])
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    ends = {k: np.concatenate(v) for k, v in ends.items()}
    return cols, ends


def ev_of(ret, v):
    return 1.0 - float(np.var(ret - v) / (np.var(ret) + 1e-12))


def bucket_of(el):
    b = np.zeros(len(el), np.int8)
    for i, (lo, hi) in enumerate(BUCKETS):
        b[(el >= lo) & (el < hi)] = i
    return b


LEAD_BINS = [(-1e9, -20), (-20, -5), (-5, 5), (5, 20), (20, 1e9)]
LEAD_NAMES = ["<-20", "-20..-5", "-5..5", "5..20", ">20"]


def group_stats(key, x):
    """per-group (n, sum, sumsq) of x for integer keys: (unique keys, inverse, n, s1, s2)."""
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    return u, inv, n, np.bincount(inv, x), np.bincount(inv, x * x)


def nested_anova(opp, uid, blk, x):
    """exact nested sums of squares about the grand mean: opponent / game within opponent / 5 s block within game / residual"""
    mu = x.mean()
    tot = float(((x - mu) ** 2).sum())
    out = {"total": tot}
    for nm, key in (("opp", opp.astype(np.int64)), ("game", uid), ("block", uid * 1000 + blk)):
        u, inv, n, s1, s2 = group_stats(key, x)
        out[f"cum_{nm}"] = float((s1 ** 2 / n).sum() - len(x) * mu ** 2)    # SS explained by the grouping up to this level
    out["ss_opp"] = out["cum_opp"]
    out["ss_game"] = out["cum_game"] - out["cum_opp"]
    out["ss_block"] = out["cum_block"] - out["cum_game"]
    out["ss_resid"] = tot - out["cum_block"]
    return out


def norm_variants(mu, sd, share):
    """per-opponent (shift c, scale d) of the four advantage normalisations for a batch with game shares ``share``
    (mu, sd = per-opponent mean / std of the raw advantages); also the batch std of the raw advantages."""
    mu, sd, share = (np.asarray(x, np.float64) for x in (mu, sd, share))
    mb = float((share * mu).sum())
    var_g = float((share * (sd ** 2 + mu ** 2)).sum() - mb ** 2)          # = adv.std() ** 2 of the batch
    var_c = float((share * sd ** 2).sum())                                  # std after per-opponent centering
    z = np.zeros_like(mu)
    return {"a_global_std": (z, np.full_like(mu, np.sqrt(var_g))),
            "b_center_global": (np.full_like(mu, mb), np.full_like(mu, np.sqrt(var_g))),
            "c_center_opp_gstd": (mu.copy(), np.full_like(mu, np.sqrt(var_c))),
            "d_center_opp_ostd": (mu.copy(), sd.copy())}, np.sqrt(var_g)


def mix_table(mu, sd, names, pool_idx, new_i, share_new):
    """expected batch statistics of the four normalisations: pool-only (equal shares over the pool) vs pool + new at share_new."""
    k = len(pool_idx)
    sh = np.zeros(len(names))
    sh[pool_idx] = 1.0 / k
    base, sg_base = norm_variants(mu, sd, sh)
    sh2 = sh * (1 - share_new)
    sh2[new_i] = share_new
    var, sg = norm_variants(mu, sd, sh2)
    out = {"global_std_pool_only": float(sg_base), "global_std_mixed": float(sg), "std_inflation": float(sg / sg_base)}
    act = [i for i in range(len(names)) if sh2[i] > 0]
    for v, (c, d) in var.items():
        m2 = lambda i: (sd[i] ** 2 + (mu[i] - c[i]) ** 2) / d[i] ** 2          # second moment of the normalised advantages of opponent i
        pool_scale = float(np.sqrt(sum(sh2[i] * m2(i) for i in pool_idx) / (1 - share_new)))
        pool_scale0 = float(np.sqrt(sum(sh[i] * (sd[i] ** 2 + (mu[i] - base[v][0][i]) ** 2) / base[v][1][i] ** 2 for i in pool_idx)))
        out[v] = {"new_offset_norm": float((mu[new_i] - c[new_i]) / d[new_i]),            # mean of the new opponent's normalised advantages
                  "pool_offset_norm": float(np.mean([(mu[i] - c[i]) / d[i] for i in pool_idx])),
                  "pool_signal_scale": pool_scale, "pool_signal_ratio": pool_scale / pool_scale0,
                  "new_scale": float(np.sqrt(m2(new_i))),
                  # factor applied to each pool opponent's raw advantages, relative to today's pool-only global-std scaling
                  "gain_vs_base": {names[i]: float((1 / d[i]) / (1 / base["a_global_std"][1][i])) for i in pool_idx},
                  # share of the normalised advantages' second moment that is the new opponent's constant offset
                  "offset_power_share": float(share_new * ((mu[new_i] - c[new_i]) / d[new_i]) ** 2 / sum(sh2[i] * m2(i) for i in act))}
    return out


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    c, e = load_rows(a.data)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    names = meta["names"]
    K = len(names)
    adv = c["adv"].astype(np.float64)                                 # == ret - V (ret = adv + V)
    assert np.allclose(c["ret"] - c["v"], c["adv"], atol=1e-4)
    el = c["elapsed"]
    bk = bucket_of(el)
    lead_now = c["own_score"] - c["opp_score"]
    ld = np.zeros(len(el), np.int8)
    for i, (lo, hi) in enumerate(LEAD_BINS):
        ld[(lead_now >= lo) & (lead_now < hi)] = i
    uid = c["uid"]
    # MC return = the game's final reward (sparse terminal reward, gamma = 1): known for finished games
    fin_of = dict(zip(e["euid"].tolist(), e["efin"].tolist()))
    u_, inv_ = np.unique(uid, return_inverse=True)
    mc = np.array([fin_of.get(int(u), np.nan) for u in u_])[inv_]
    has_mc = ~np.isnan(mc)
    mx = np.zeros(len(u_))
    np.maximum.at(mx, inv_, el)
    tail = (el > mx[inv_] - 20 / 25.0) & has_mc                      # last 20 decisions (= 1/(1-lambda)) of a finished game
    blk = (el // 5.0).astype(np.int64)
    out = {"names": names, "ended": meta["ended"], "per_opp": {}}
    mu = np.zeros(K)
    sd = np.zeros(K)
    gstat = []
    for i, n in enumerate(names):
        m = c["opp"] == i
        em = e["eopp"] == i
        r, v, ret = adv[m], c["v"][m].astype(np.float64), c["ret"][m].astype(np.float64)
        mu[i], sd[i] = r.mean(), r.std()
        u, inv, gn, gs, gss = group_stats(uid[m], r)
        gstat.append((gn, gs, gss))
        gm = gs / gn
        mm = m & has_mc
        mcv = mc[mm]
        vm = c["v"][mm].astype(np.float64)
        o = dict(rows=int(m.sum()), games_ended=int(em.sum()), games_seen=int(len(u)),
                 win=float((e["efin"][em] > 0).mean()), margin=float(e["elead"][em].mean()),
                 margin_se=float(e["elead"][em].std(ddof=1) / np.sqrt(max(em.sum(), 1))),
                 bias_gae=float(r.mean()), bias_gae_se=float(np.sqrt(np.sum(gn ** 2 * (gm - r.mean()) ** 2)) / gn.sum()),
                 ev_gae=ev_of(ret, v), abs_adv=float(np.abs(r).mean()), adv_std=float(r.std()),
                 v_mean=float(v.mean()), v_std=float(v.std()), ret_mean=float(ret.mean()), ret_std=float(ret.std()),
                 bias_mc=float((mcv - vm).mean()), mc_rows=int(mm.sum()), ev_mc=ev_of(mcv, vm), mc_std=float(mcv.std()),
                 bias_gae_tail20=float(adv[m & tail].mean()), abs_adv_tail20=float(np.abs(adv[m & tail]).mean()),
                 bias_gae_notail=float(adv[m & ~tail & has_mc].mean()), abs_adv_notail=float(np.abs(adv[m & ~tail & has_mc]).mean()),
                 tail_ss_share=float((adv[m & tail] ** 2).sum() / (adv[m & has_mc] ** 2).sum()))
        u2, inv2, n2, s2, ss2 = group_stats(uid[mm], mc[mm] - c["v"][mm])
        gm2 = s2 / n2
        o["bias_mc_se"] = float(np.sqrt(np.sum(n2 ** 2 * (gm2 - o["bias_mc"]) ** 2)) / n2.sum())
        o["bucket"], o["lead_state"] = {}, {}
        for j, bn in enumerate(BUCKET_NAMES):
            q = m & (bk == j)
            qq = q & has_mc
            if q.sum() > 0:
                o["bucket"][bn] = dict(n=int(q.sum()), bias_gae=float(adv[q].mean()), ev_gae=ev_of(c["ret"][q], c["v"][q]),
                                       abs_adv=float(np.abs(adv[q]).mean()), adv_std=float(adv[q].std()),
                                       bias_mc=float((mc[qq] - c["v"][qq]).mean()) if qq.sum() else None,
                                       v_mean=float(c["v"][q].mean()))
        for j, bn in enumerate(LEAD_NAMES):
            q = m & (ld == j)
            qq = q & has_mc
            if q.sum() > 100:
                o["lead_state"][bn] = dict(n=int(q.sum()), bias_gae=float(adv[q].mean()), abs_adv=float(np.abs(adv[q]).mean()),
                                           bias_mc=float((mc[qq] - c["v"][qq]).mean()) if qq.sum() else None, v_mean=float(c["v"][q].mean()))
        an = nested_anova(np.zeros(m.sum(), np.int8), uid[m], blk[m], r)
        o["anova_frac"] = {k: an[f"ss_{k}"] / an["total"] for k in ("game", "block", "resid")}
        # telescoping identity of GAE(gamma=1, lambda): sum_t adv_t ~= (1/(1-lambda)) * (R - V_0) per finished game
        # (a constant critic offset b only enters through the first and last ~20 decisions)
        ge = np.unique(uid[mm])
        first_v = np.zeros(len(u_)); first_el = np.full(len(u_), 1e9)
        order = np.lexsort((el, uid))
        us, st = np.unique(uid[order], return_index=True)
        first_v_g = c["v"][order][st].astype(np.float64)
        ng = np.bincount(np.searchsorted(us, uid[mm]), minlength=len(us))
        sel = ng > 0
        rg = np.array([fin_of[int(u)] for u in us[sel]])
        o["identity_pred_mean_adv"] = float(20.0 * (rg - first_v_g[sel]).sum() / ng[sel].sum())
        o["identity_obs_mean_adv"] = float(adv[mm].mean())
        out["per_opp"][n] = o
    # ---------------- variance decomposition of the advantages (pooled subsets)
    def dec(idx):
        m = np.isin(c["opp"], idx)
        an = nested_anova(c["opp"][m], uid[m], blk[m], adv[m])
        t = an["total"]
        ss_ob = 0.0     # opponent x elapsed-bucket structure (time-dependent bias) within opponent
        for i in idx:
            mi = c["opp"] == i
            for j in range(len(BUCKETS)):
                q = mi & (bk == j)
                if q.sum():
                    ss_ob += q.sum() * (adv[q].mean() - adv[mi].mean()) ** 2
        return {"n": int(m.sum()), "var": t / m.sum(), **{f"frac_{k}": an[f"ss_{k}"] / t for k in ("opp", "game", "block", "resid")},
                "frac_opp_x_timebucket": ss_ob / t}
    pool_idx = [names.index(n) for n in POOL]
    out["decomp"] = {"pool": dec(pool_idx)}
    for nn in NEW:
        out["decomp"]["pool+" + nn] = dec(pool_idx + [names.index(nn)])
    # ---------------- mixtures / normalisations, with a game-cluster bootstrap
    rng = np.random.default_rng(0)
    mix = {}
    for nn in NEW:
        ni = names.index(nn)
        for sh in (0.05, 0.10, 0.20):
            pt = mix_table(mu, sd, names, pool_idx, ni, sh)
            bs = []
            for _ in range(a.boot):
                mu_b, sd_b = mu.copy(), sd.copy()
                for i in range(K):
                    gn, gs, gss = gstat[i]
                    pick = rng.integers(0, len(gn), len(gn))
                    n_, s1, s2 = gn[pick].sum(), gs[pick].sum(), gss[pick].sum()
                    mu_b[i] = s1 / n_
                    sd_b[i] = np.sqrt(max(s2 / n_ - mu_b[i] ** 2, 0))
                bs.append(mix_table(mu_b, sd_b, names, pool_idx, ni, sh))

            def ci(f):
                v = np.array([f(b) for b in bs])
                return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            pt["ci"] = {"std_inflation": ci(lambda b: b["std_inflation"])}
            for vname in ("a_global_std", "b_center_global", "c_center_opp_gstd", "d_center_opp_ostd"):
                pt["ci"][vname] = {"new_offset_norm": ci(lambda b, vname=vname: b[vname]["new_offset_norm"]),
                                   "pool_signal_ratio": ci(lambda b, vname=vname: b[vname]["pool_signal_ratio"])}
            mix[f"{nn}@{int(sh * 100)}"] = pt
    out["mix"] = mix
    if a.crushed:
        c2, e2 = load_rows(a.crushed)
        n2 = json.load(open(os.path.join(a.crushed, "meta.json")))["names"]
        pool_sd = out["mix"][f"{NEW[0]}@5"]["global_std_pool_only"]
        out["crushed"] = {}
        for i, n in enumerate(n2):
            m = c2["opp"] == i
            em = e2["eopp"] == i
            r = c2["adv"][m].astype(np.float64)
            out["crushed"][n] = dict(rows=int(m.sum()), games_ended=int(em.sum()), win=float((e2["efin"][em] > 0).mean()),
                                     margin=float(e2["elead"][em].mean()), bias_gae=float(r.mean()), ev_gae=ev_of(c2["ret"][m], c2["v"][m]),
                                     abs_adv=float(np.abs(r).mean()), adv_std=float(r.std()), v_mean=float(c2["v"][m].mean()),
                                     amplification_d_vs_a=float(pool_sd / r.std()))
    json.dump(out, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    # ---------------- figures
    cols = plt.cm.tab10(np.arange(K))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 0.38
    for j, key in enumerate(("bias_gae", "bias_mc")):
        ax[0].bar(np.arange(K) + (j - .5) * w, [out["per_opp"][n][key] for n in names], w,
                  yerr=[1.96 * out["per_opp"][n][key + "_se"] for n in names],
                  label={"bias_gae": "mean(ret-V) = mean GAE adv", "bias_mc": "mean(final reward - V) (MC)"}[key])
    ax[0].set_xticks(range(K))
    ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_title("critic bias per opponent (95% CI, game-clustered)")
    ax[0].legend(fontsize=8)
    for i, n in enumerate(names):
        ax[1].plot(range(len(BUCKET_NAMES)), [(out["per_opp"][n]["bucket"].get(b, {}).get("bias_mc") or np.nan) for b in BUCKET_NAMES], "-o", color=cols[i], label=n)
        ax[2].plot(range(len(BUCKET_NAMES)), [out["per_opp"][n]["bucket"].get(b, {}).get("bias_gae", np.nan) for b in BUCKET_NAMES], "-o", color=cols[i], label=n)
    for k_, t in ((1, "MC bias (final reward - V) by elapsed time [s]"), (2, "GAE bias (mean adv) by elapsed time [s]")):
        ax[k_].set_xticks(range(len(BUCKET_NAMES)))
        ax[k_].set_xticklabels(BUCKET_NAMES)
        ax[k_].axhline(0, color="k", lw=.5)
        ax[k_].set_title(t)
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "bias_by_opponent_time.png"), dpi=110)
    plt.close()

    fig, ax = plt.subplots(2, K, figsize=(2.4 * K, 5), sharey="row")
    for i, n in enumerate(names):
        m = c["opp"] == i
        ax[0, i].hist(c["v"][m], bins=60, range=(-1.3, 1.3), color=cols[i])
        ax[0, i].set_title(n, fontsize=8)
        ax[1, i].hist(c["ret"][m], bins=60, range=(-1.3, 1.3), color=cols[i])
    ax[0, 0].set_ylabel("V(s)")
    ax[1, 0].set_ylabel("return (GAE)")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "hist_v_ret.png"), dpi=110)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    po = [out["per_opp"][n] for n in names]
    ax[0].bar(range(K), [o["adv_std"] for o in po], color=cols)
    ax[0].set_xticks(range(K))
    ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_title("raw advantage std per opponent")
    for vn, st in (("a_global_std", "-"), ("c_center_opp_gstd", ":")):
        for nn, col in zip(NEW, plt.cm.tab10(np.arange(4) + 4)):
            ax[1].plot([5, 10, 20], [mix[f"{nn}@{s}"][vn]["pool_signal_ratio"] for s in (5, 10, 20)], st, color=col, marker="o", label=f"{nn} ({vn[0]})")
    ax[1].set_xlabel("new-opponent share [%]")
    ax[1].set_title("pool games' normalised-advantage scale vs pool-only")
    ax[1].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "normalisation.png"), dpi=110)
    plt.close()
    for n, o in out["per_opp"].items():
        print(f"{n:12s} games {o['games_ended']:4d} win {o['win']:.3f} margin {o['margin']:+6.1f}  biasGAE {o['bias_gae']:+.4f}  biasMC {o['bias_mc']:+.4f}  "
              f"EVgae {o['ev_gae']:.4f} EVmc {o['ev_mc']:.3f}  |adv| {o['abs_adv']:.4f} std {o['adv_std']:.4f}")


# ------------------------------------------------------------------------------------------------ fine-tune simulation

def gae_np(v, off, ln, nxt, lam):
    """GAE(gamma=1, lam) per contiguous segment (collect()'s loop): v = rows concatenated, segment s = rows off[s]:off[s]+ln[s];
    nxt[s] = value after the segment's last step (finished game: the reward, else the critic's bootstrap)."""
    adv = np.zeros(len(v))
    for s in range(len(off)):
        vb = v[off[s]:off[s] + ln[s]].astype(np.float64)
        gae = 0.0
        for j in range(ln[s] - 1, -1, -1):
            vn = nxt[s] if j == ln[s] - 1 else vb[j + 1]
            gae = (vn - vb[j]) + lam * gae
            adv[off[s] + j] = gae
    return adv


def cmd_finetune(a):
    import copy

    import torch
    from blackout_env.rl.ppo import PPOConfig, PPOTrainer

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    names = meta["names"]
    # ---- stored segments (all rows of a segment contiguous, in time order)
    G, T, A, seg = [], [], [], []
    BG, BT, BA = [], [], []
    base = nb = 0
    files = sorted(glob.glob(os.path.join(a.data, "seg_*.npz")))
    tot = np.zeros(len(names))
    for f in files:                                            # pass 1: segments per opponent (memory cap: keep a random subset)
        np.add.at(tot, np.load(f)["segopp"], 1)
    keep_p = np.minimum(1.0, a.max_seg_per_opp / tot)
    lrng = np.random.default_rng(a.seed + 99)
    print("segments per opponent", tot.astype(int).tolist(), "keep p", np.round(keep_p, 3).tolist(), flush=True)
    for f in files:
        z = np.load(f)
        so = z["segopp"]
        ks = np.nonzero(lrng.random(len(so)) < keep_p[so])[0]
        sg_, st_, sa_ = z["sg"], z["st"], z["sa"]
        off = np.r_[0, np.cumsum(z["seglen"])[:-1]]
        rows = np.concatenate([np.arange(off[j], off[j] + z["seglen"][j]) for j in ks]) if len(ks) else np.zeros(0, np.int64)
        G.append(sg_[rows]); T.append(st_[rows]); A.append(sa_[rows])
        bidx = {int(b): j for j, b in enumerate(z["bootlive"])}
        boot_rows = []
        noff = 0
        for j in ks:
            bi = -1
            if not z["segover"][j]:
                bi = nb + len(boot_rows)
                boot_rows.append(bidx[int(z["segb"][j])])
            seg.append((base + noff, int(z["seglen"][j]), int(z["segopp"][j]), int(z["seguid"][j]), bool(z["segover"][j]),
                        float(z["segfin"][j]), bi))
            noff += int(z["seglen"][j])
        BG.append(z["bootg"][boot_rows]); BT.append(z["boott"][boot_rows]); BA.append(z["boota"][boot_rows])
        nb += len(boot_rows)
        base += noff
    G, T, A = np.concatenate(G), np.concatenate(T), np.concatenate(A)
    BG, BT, BA = np.concatenate(BG), np.concatenate(BT), np.concatenate(BA)
    seg = np.array(seg, dtype=[("off", "i8"), ("len", "i4"), ("opp", "i1"), ("uid", "i8"), ("over", "?"), ("fin", "f8"), ("bi", "i8")])
    _, e = load_rows(a.data)
    fin_of = dict(zip(e["euid"].tolist(), e["efin"].tolist()))
    seg_mc = np.array([fin_of.get(int(u), np.nan) for u in seg["uid"]])     # game outcome (known only for finished games)
    print("stored rows", len(G), "segments", len(seg), flush=True)
    train = np.nonzero(seg["uid"] % 2 == 0)[0]      # games split by uid parity: train / held-out
    test = np.nonzero(seg["uid"] % 2 == 1)[0]
    by_opp = lambda idx, o: idx[seg["opp"][idx] == o]
    rng = np.random.default_rng(a.seed)
    # ---- trainer: the critic update is PPOTrainer.update itself (critic_warmup=inf disables the policy update)
    cfg = PPOConfig(init=f"{R}/ppo_r2_reg/final.pt", opponents=[OPPS["p24c_v321"]], opp_weights=[1.0], games=2, critic_fp16=True,
                    update_compile=a.compile, update_fp16=True, critic_warmup=10**9, seed=a.seed)
    tr = PPOTrainer(cfg)
    tr.critic.load_state_dict(torch.load(a.critic, map_location=tr.dev, weights_only=True))
    tr.critic.eval()

    def values(g, t, ag):
        return tr._values_np(g, t, ag).astype(np.float64)

    def build(sel):
        rows = np.concatenate([np.arange(seg["off"][s], seg["off"][s] + seg["len"][s]) for s in sel])
        ln = seg["len"][sel]
        return rows, np.r_[0, np.cumsum(ln)[:-1]], ln, seg["bi"][sel]

    def adv_ret(sel, rows, off, ln, bi):
        v = values(G[rows], T[rows], A[rows])
        nxt = seg["fin"][sel].copy()
        live = bi >= 0
        if live.any():
            nxt[live] = values(BG[bi[live]], BT[bi[live]], BA[bi[live]])
        adv = gae_np(v, off, ln, nxt, cfg.gae_lambda)
        return v, adv, v + adv

    def evaluate(max_seg):
        res = {}
        for o, n in enumerate(names):
            sel = by_opp(test, o)
            if len(sel) > max_seg:
                sel = np.sort(np.random.default_rng(7).choice(sel, max_seg, replace=False))
            rows, off, ln, bi = build(sel)
            v, adv, ret = adv_ret(sel, rows, off, ln, bi)
            bkt = bucket_of((1 - T[rows][:, 2]) * EPISODE_SECONDS)
            mc = np.repeat(seg_mc[sel], ln)
            hm = ~np.isnan(mc)
            res[n] = dict(bias_gae=float(adv.mean()), ev_gae=ev_of(ret, v), abs_adv=float(np.abs(adv).mean()), adv_std=float(adv.std()),
                          bias_mc=float((mc[hm] - v[hm]).mean()), ev_mc=ev_of(mc[hm], v[hm]), v_mean=float(v.mean()),
                          bucket_bias_gae=[float(adv[bkt == j].mean()) if (bkt == j).any() else None for j in range(len(BUCKETS))],
                          bucket_bias_mc=[float((mc[hm & (bkt == j)] - v[hm & (bkt == j)]).mean()) if (hm & (bkt == j)).any() else None
                                          for j in range(len(BUCKETS))])
        return res

    def one_iter(shares):
        """critic-only update on a freshly drawn batch; shares = {opp index: number of segments}."""
        sel = np.concatenate([rng.choice(by_opp(train, o), k, replace=False) for o, k in shares.items()])
        rows, off, ln, bi = build(sel)
        v, adv, ret = adv_ret(sel, rows, off, ln, bi)
        batch = {"g": G[rows], "t": T[rows], "a": A[rows], "act": np.zeros(len(rows), np.int8), "adv": adv.astype(np.float32),
                 "ret": ret.astype(np.float32)}
        return tr.update(batch)["vf_loss"], float(adv.mean())

    def fresh_scaler():
        tr.cscaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10**9, enabled=tr.dev.type == "cuda")

    pool_idx = [names.index(n) for n in POOL]
    NS = a.segs
    log = {"names": names, "args": {k: v for k, v in vars(a).items() if k != "fn"}, "eval0": evaluate(a.eval_segs), "traj": {}}
    print("eval k=0 done", flush=True)
    # pre-roll: warm Adam on pool-only batches (the real critic has a long optimiser history), then snapshot
    tr.copt = torch.optim.AdamW(tr.critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    fresh_scaler()
    for _ in range(a.preroll):
        one_iter({o: NS // 3 for o in pool_idx})
    snap = {k: v.clone() for k, v in tr.critic.state_dict().items()}
    opt_state = copy.deepcopy(tr.copt.state_dict())
    log["eval_preroll"] = evaluate(a.eval_segs)
    ks = [1, 2, 3, 5, 10, 15, 20]
    jobs = [("pool_only", 0.0)] + [(nn, s) for nn in NEW for s in (0.05, 0.10, 0.20)]
    if a.only:
        jobs = [(nn_, float(s)) for nn_ in a.only.split(",") for s in a.shares.split(",")]
    for nn, s in jobs:
        key = f"{nn}@{int(s * 100)}"
        tr.critic.load_state_dict(snap)
        tr.copt.load_state_dict(copy.deepcopy(opt_state))
        for pg in tr.copt.param_groups:
            pg["lr"] = cfg.critic_lr * a.lr_mult      # warm-up variant: larger critic lr (the pre-roll used the base lr)
        fresh_scaler()
        if nn == "pool_only":
            shares = {o: NS // 3 for o in pool_idx}
        else:
            n_new = int(round(s * NS))
            shares = {o: (NS - n_new) // 3 for o in pool_idx}
            shares[names.index(nn)] = n_new
        traj = {}
        for k in range(1, max(ks) + 1):
            l, mb = one_iter(shares)
            traj.setdefault("loss", []).append(l)
            traj.setdefault("batch_bias_gae", []).append(mb)
            if k in ks:
                traj[f"k{k}"] = evaluate(a.eval_segs)
                tr.critic.eval()
        log["traj"][key] = traj
        json.dump(log, open(os.path.join(a.out, f"finetune{a.tag}.json"), "w"))
        print("done", key, flush=True)
    os._exit(0)


def cmd_ftreport(a):
    """summary table + figures of the critic-only fine-tune simulation (finetune*.json from ``finetune``)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for f in a.ft:
        d = json.load(open(f))
        tag = os.path.basename(f)[len("finetune"):-len(".json")] or "_lr1"
        runs[tag] = d
    names = runs[next(iter(runs))]["names"]
    pool = [n for n in POOL]
    ks = [0, 1, 2, 3, 5, 10, 15, 20]
    summ = {}

    def at(d, key, k):
        return d["eval0"] if k == 0 else d["traj"][key][f"k{k}"]

    for tag, d in runs.items():
        ctrl = d["traj"].get("pool_only@0")
        for key in d["traj"]:
            nn = key.split("@")[0]
            if nn == "pool_only":
                continue
            row = {}
            for k in ks:
                e = at(d, key, k)
                pl = [e[n] for n in pool]
                row[k] = dict(new_bias_mc=e[nn]["bias_mc"], new_ev_mc=e[nn]["ev_mc"], new_bias_gae=e[nn]["bias_gae"], new_ev_gae=e[nn]["ev_gae"],
                              new_bucket_bias_mc=e[nn]["bucket_bias_mc"], new_bucket_bias_gae=e[nn]["bucket_bias_gae"], new_abs_adv=e[nn]["abs_adv"],
                              pool_bias_mc=float(np.mean([x["bias_mc"] for x in pl])), pool_ev_mc=float(np.mean([x["ev_mc"] for x in pl])),
                              pool_ev_gae=float(np.mean([x["ev_gae"] for x in pl])), pool_abs_adv=float(np.mean([x["abs_adv"] for x in pl])),
                              pool_bucket_bias_mc=[float(np.nanmean([np.nan if x["bucket_bias_mc"][j] is None else x["bucket_bias_mc"][j] for x in pl]))
                                                   for j in range(len(BUCKETS))])
                if ctrl is not None and k > 0:
                    ec = ctrl[f"k{k}"]
                    row[k]["ctrl_new_bias_mc"] = ec[nn]["bias_mc"]
                    row[k]["ctrl_new_ev_mc"] = ec[nn]["ev_mc"]
                    row[k]["ctrl_pool_ev_mc"] = float(np.mean([ec[n]["ev_mc"] for n in pool]))
                    row[k]["ctrl_pool_bias_mc"] = float(np.mean([ec[n]["bias_mc"] for n in pool]))
            summ[f"{tag}:{key}"] = row
    json.dump(summ, open(os.path.join(a.out, "finetune_summary.json"), "w"), indent=1)
    print("run:key  (k -> new bias_MC / new EV_MC | pool bias_MC / pool EV_MC), control pool-only in [ ]")
    for rk, row in summ.items():
        print(rk)
        for k in ks:
            r = row[k]
            c = f" [ctrl new {r['ctrl_new_bias_mc']:+.3f}/{r['ctrl_new_ev_mc']:.3f} pool {r['ctrl_pool_bias_mc']:+.3f}/{r['ctrl_pool_ev_mc']:.3f}]" if "ctrl_new_bias_mc" in r else ""
            print(f"  k={k:2d} new {r['new_bias_mc']:+.3f}/{r['new_ev_mc']:.3f} gae {r['new_bias_gae']:+.4f}/{r['new_ev_gae']:.4f} |adv| {r['new_abs_adv']:.4f}"
                  f" | pool {r['pool_bias_mc']:+.3f}/{r['pool_ev_mc']:.3f} |adv| {r['pool_abs_adv']:.4f}{c}")
    # figures: one panel per new opponent (lr x1 runs) + the lr variants for the self opponent
    base = runs.get("_lr1") or runs[next(iter(runs))]
    new_list = [n for n in NEW if any(k.startswith(n) for k in base["traj"])]
    fig, ax = plt.subplots(3, len(new_list), figsize=(4.2 * len(new_list), 9), squeeze=False)
    cmap = {"5": "tab:blue", "10": "tab:orange", "20": "tab:red"}
    for j, nn in enumerate(new_list):
        for sh in ("5", "10", "20"):
            key = f"{nn}@{sh}"
            if key not in base["traj"]:
                continue
            row = summ[f"_lr1:{key}"] if "_lr1" in runs else summ[f"{next(iter(runs))}:{key}"]
            ax[0, j].plot(ks, [row[k]["new_bias_mc"] for k in ks], "-o", color=cmap[sh], label=f"{sh}% share")
            ax[1, j].plot(ks, [row[k]["new_ev_mc"] for k in ks], "-o", color=cmap[sh], label=f"{sh}% share")
            ax[2, j].plot(ks, [row[k]["pool_ev_mc"] for k in ks], "-o", color=cmap[sh], label=f"{sh}% share")
        ctrl = base["traj"].get("pool_only@0")
        if ctrl:
            ax[0, j].plot(ks[1:], [ctrl[f"k{k}"][nn]["bias_mc"] for k in ks[1:]], "k--", label="pool-only control")
            ax[1, j].plot(ks[1:], [ctrl[f"k{k}"][nn]["ev_mc"] for k in ks[1:]], "k--", label="pool-only control")
            ax[2, j].plot(ks[1:], [float(np.mean([ctrl[f"k{k}"][n]["ev_mc"] for n in pool])) for k in ks[1:]], "k--", label="pool-only control")
        ax[0, j].set_title(nn)
        ax[0, j].set_ylabel("new opp: mean(final reward - V)")
        ax[1, j].set_ylabel("new opp: MC explained variance")
        ax[2, j].set_ylabel("pool opps: MC explained variance")
        ax[2, j].set_xlabel("critic-only iterations k (lr 1e-5)")
        ax[0, j].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "critic_warmup_recovery.png"), dpi=110)
    plt.close()
    lr_runs = [t for t in runs if t != "_lr1"]
    if lr_runs and "_lr1" in runs:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for tag, col in zip(["_lr1"] + lr_runs, ("k", "tab:blue", "tab:red", "tab:green")):
            for sh, ls in (("10", "-"), ("20", "--")):
                key = f"self@{sh}"
                key = f"r2reg_final@{sh}"
                if f"{tag}:{key}" not in summ:
                    continue
                row = summ[f"{tag}:{key}"]
                lab = f"lr x{tag[3:]} {sh}%"
                ax[0].plot(ks, [row[k]["new_bias_mc"] for k in ks], ls, color=col, marker="o", label=lab)
                ax[1].plot(ks, [row[k]["new_ev_mc"] for k in ks], ls, color=col, marker="o", label=lab)
                ax[2].plot(ks, [row[k]["pool_ev_mc"] for k in ks], ls, color=col, marker="o", label=lab)
        ax[0].set_title("r2reg_final: MC bias")
        ax[1].set_title("r2reg_final: MC EV")
        ax[2].set_title("pool opps: MC EV")
        for x_ in ax:
            x_.set_xlabel("critic-only iterations k")
        ax[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "critic_warmup_lr.png"), dpi=110)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--critic", default=f"{R}/ppo_r2_reg/final_critic.pt")
    c.add_argument("--games", type=int, default=280)
    c.add_argument("--target", type=int, default=200, help="stop when every opponent has this many finished games")
    c.add_argument("--hours", type=float, default=8.0)
    c.add_argument("--max-iters", type=int, default=10**9)
    c.add_argument("--p-store", type=float, default=0.15, help="share of (game, iteration) segments whose views are kept")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--opps", default="", help="comma list of opponent names (default: the 7 of OPPS; EXTRA: p5a_v22, p16_v27)")
    c.set_defaults(fn=cmd_collect)
    an = sp.add_parser("analyze")
    an.add_argument("--data", required=True)
    an.add_argument("--out", required=True)
    an.add_argument("--boot", type=int, default=300)
    an.add_argument("--crushed", default="", help="collect output vs crushed opponents (p5a_v22, p16_v27): adds their advantage scales")
    an.set_defaults(fn=cmd_analyze)
    f = sp.add_parser("finetune")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--critic", default=f"{R}/ppo_r2_reg/final_critic.pt")
    f.add_argument("--segs", type=int, default=255, help="segments (= games) per simulated iteration (divisible by 3)")
    f.add_argument("--eval-segs", type=int, default=300, help="held-out segments per opponent per evaluation")
    f.add_argument("--preroll", type=int, default=5)
    f.add_argument("--compile", action="store_true")
    f.add_argument("--only", default="", help="comma list of new opponents (default: all four + a pool-only control)")
    f.add_argument("--shares", default="0.05,0.1,0.2")
    f.add_argument("--lr-mult", type=float, default=1.0)
    f.add_argument("--tag", default="")
    f.add_argument("--max-seg-per-opp", type=int, default=2400, help="memory cap: random subset of the stored segments per opponent")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=cmd_finetune)
    r = sp.add_parser("ftreport")
    r.add_argument("--ft", nargs="+", required=True, help="finetune*.json files (finetune.json = base lr; finetune_lr10.json ...)")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_ftreport)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
