"""Sample-efficiency analysis: eval jsons (reports/rl/sample_eff/*.json) x log.jsonl x-axes -> tables + PNGs.
Run: python reports/rl/sample_eff_analysis.py   (from repo root)"""
import json, glob, os, math, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "reports/rl/sample_eff"
RUNS = ["grpo_r3b", "ppo_r1", "ppo_r2_reg"]
LABEL = {"grpo_r3b": "GRPO r3b", "ppo_r1": "PPO r1", "ppo_r2_reg": "PPO r2 (reg)"}
COL = {"grpo_r3b": "#d95f02", "ppo_r1": "#1b9e77", "ppo_r2_reg": "#7570b3"}
OPPS = ["p24c_v321", "p22_v31", "v32_1"]
ONAME = {"p24c_v321": "vs p24c_v321", "p22_v31": "vs p22_v31", "v32_1": "vs V32.1", "avg": "average of 3"}

def load_log(r):
    return [json.loads(l) for l in open(f"{D}/runlogs/{r}_log.jsonl")]

def axes_for(r):
    """iter -> (cum decisions, cum views, wall_h)"""
    L = [x for x in load_log(r) if "views" in x]
    out = {0: (0.0, 0.0, 0.0)}
    cd = cv = 0.0
    for x in L:
        cd += x["decisions"]; cv += x["views"]
        out[x["iter"]] = (cd, cv, x["wall_h"])
    # PPO has decisions_total (authoritative cumulative)
    for x in L:
        if "decisions_total" in x:
            out[x["iter"]] = (x["decisions_total"], out[x["iter"]][1], x["wall_h"])
    return out

def stats(m0, m1):
    m = np.array(list(m0) + list(m1), float)
    w = (np.sign(m) + 1) / 2
    n = len(m)
    return dict(mean=m.mean(), se=m.std(ddof=1) / math.sqrt(n), win=w.mean(), wse=w.std(ddof=1) / math.sqrt(n), n=n, m=m, w=w)

def get(name, o):
    f = f"{D}/{name}__{o}.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return stats(j["margin_side0"], j["margin_side1"])

# ---- collect points
pts = {r: [] for r in RUNS}   # (iter, decisions, views, hours, {opp: stats})
step0 = {o: get("step0__p24c_v321", o) for o in OPPS}
for r in RUNS:
    ax = axes_for(r)
    final_it = max(ax)
    its = sorted(int(re.search(r"__(\d+)__", os.path.basename(f)).group(1)) for f in glob.glob(f"{D}/{r}__*__p24c_v321.json")
                 if re.search(r"__(\d+)__", os.path.basename(f)))
    cands = [(i, i) for i in its] + ([(final_it, "final")] if final_it not in its else [])
    pts[r].append((0, 0.0, 0.0, 0.0, step0))
    for it, tag in sorted(cands):
        s = {o: get(f"{r}__{tag}", o) for o in OPPS}
        if any(v is None for v in s.values()):
            continue
        d, v, h = ax[it]
        pts[r].append((it, d, v, h, s))

def avg_stats(s):
    """average over opponents: mean of per-game paired margins (same seeds/sides across opps -> games independent per opp)"""
    ms = np.mean([s[o]["mean"] for o in OPPS]); ws = np.mean([s[o]["win"] for o in OPPS])
    se = math.sqrt(sum(s[o]["se"] ** 2 for o in OPPS)) / len(OPPS)
    wse = math.sqrt(sum(s[o]["wse"] ** 2 for o in OPPS)) / len(OPPS)
    return dict(mean=ms, se=se, win=ws, wse=wse)

# ---- tables
lines = []
for r in RUNS:
    lines.append(f"\n#### {LABEL[r]}\n")
    lines.append("| iter | decisions (M) | views (M) | wall h | " + " | ".join(f"{ONAME[o]} margin (SE) / win" for o in OPPS) + " | avg margin (SE) | avg win |")
    lines.append("|" + "---|" * (4 + len(OPPS) + 2))
    for it, d, v, h, s in pts[r]:
        a = avg_stats(s)
        lines.append(f"| {it} | {d/1e6:.2f} | {v/1e6:.2f} | {h:.2f} | " + " | ".join(f"{s[o]['mean']:+.1f} ({s[o]['se']:.1f}) / {s[o]['win']:.2f}" for o in OPPS)
                     + f" | {a['mean']:+.1f} ({a['se']:.1f}) | {a['win']:.3f} |")
open(f"{D}/tables.md", "w").write("\n".join(lines))

# ---- budget-fraction table (interpolate average margin / win at 25/50/75/100% of decisions budget)
frac_lines = ["| run | metric | 25% | 50% | 75% | 100% |", "|---|---|---|---|---|---|"]
for r in RUNS:
    P = pts[r]
    xs = np.array([p[1] for p in P]); tot = xs.max()
    am = np.array([avg_stats(p[4])["mean"] for p in P]); aw = np.array([avg_stats(p[4])["win"] for p in P])
    cells_m, cells_w = [], []
    for f in (.25, .5, .75, 1.0):
        # nearest checkpoint (no interpolation) to the budget fraction
        k = int(np.argmin(np.abs(xs - f * tot)))
        cells_m.append(f"{am[k]:+.1f} (it {P[k][0]})"); cells_w.append(f"{aw[k]:.3f}")
    frac_lines.append(f"| {LABEL[r]} | avg margin | " + " | ".join(cells_m) + " |")
    frac_lines.append(f"| | avg win | " + " | ".join(cells_w) + " |")
open(f"{D}/budget_table.md", "w").write("\n".join(frac_lines))

# ---- main figure: rows = x axis, cols = opponent (+avg); margin; second figure for win rate
XS = [("Cumulative simulator decisions (M)", 1, 1e6), ("Cumulative learner views (M)", 2, 1e6), ("Wall-clock hours", 3, 1.0)]
def mainfig(metric, fname):
    fig, axs = plt.subplots(3, 4, figsize=(18, 11), sharey="col")
    for ri, (xl, xi, sc) in enumerate(XS):
        for ci, o in enumerate(OPPS + ["avg"]):
            ax = axs[ri, ci]
            for r in RUNS:
                P = pts[r]
                x = [p[xi] / sc for p in P]
                st = [avg_stats(p[4]) if o == "avg" else p[4][o] for p in P]
                if metric == "margin":
                    y = [s["mean"] for s in st]; e = [1.96 * s["se"] for s in st]
                else:
                    y = [s["win"] for s in st]; e = [1.96 * s["wse"] for s in st]
                ax.errorbar(x, y, yerr=e, color=COL[r], marker="o", ms=4, lw=1.4, capsize=2, alpha=.9, label=LABEL[r])
            if ri == 0: ax.set_title(ONAME[o])
            if ri == 2: ax.set_xlabel(xl)
            else: ax.set_xlabel(xl, fontsize=8)
            ax.grid(alpha=.3)
            if ci == 0: ax.set_ylabel("mean margin" if metric == "margin" else "win rate")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Sample efficiency: {metric} (64 seeds x 2 sides = 128 games/point, bars = 95% CI; x=0 is p24c_v321)")
    fig.tight_layout(); fig.savefig(f"{D}/{fname}", dpi=110); plt.close(fig)
mainfig("margin", "fig_margin.png")
mainfig("win", "fig_winrate.png")

# ---- heatmap: avg margin per checkpoint index per run, with n/SE annotations
fig, axs = plt.subplots(1, 3, figsize=(17, 6))
for ax, r in zip(axs, RUNS):
    P = pts[r]
    M = np.array([[p[4][o]["mean"] for o in OPPS] for p in P])
    im = ax.imshow(M, cmap="RdYlGn", vmin=-10, vmax=75, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([ONAME[o] for o in OPPS], fontsize=8)
    ax.set_yticks(range(len(P))); ax.set_yticklabels([f"it {p[0]}" for p in P], fontsize=8)
    for i, p in enumerate(P):
        for j, o in enumerate(OPPS):
            ax.text(j, i, f"{p[4][o]['mean']:+.0f}\n±{p[4][o]['se']:.0f} n={p[4][o]['n']}", ha="center", va="center", fontsize=7)
    ax.set_title(LABEL[r])
fig.colorbar(im, ax=axs, shrink=.7, label="mean margin")
fig.savefig(f"{D}/fig_heatmap.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---- paired diff ppo_r2_reg - ppo_r1 at matching iter (same seeds, same GPU; different runs though) at shared iterations
shared = sorted(set(p[0] for p in pts["ppo_r2_reg"]) & set(p[0] for p in pts["ppo_r1"]))
fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
diffrows = ["| iter | " + " | ".join(ONAME[o] for o in OPPS) + " | avg |", "|---|---|---|---|---|"]
for ci, o in enumerate(OPPS + ["avg"]):
    xs, ys, es = [], [], []
    for it in shared:
        a = {p[0]: p for p in pts["ppo_r2_reg"]}[it][4]; b = {p[0]: p for p in pts["ppo_r1"]}[it][4]
        if o == "avg":
            d = np.mean([a[q]["m"] - b[q]["m"] for q in OPPS], axis=0)
        else:
            d = a[o]["m"] - b[o]["m"]
        xs.append(it); ys.append(d.mean()); es.append(1.96 * d.std(ddof=1) / math.sqrt(len(d)))
    axs[ci].errorbar(xs, ys, yerr=es, marker="o", color="#444", capsize=2); axs[ci].axhline(0, color="k", lw=.6)
    axs[ci].set_title(f"r2_reg - r1 margin, {ONAME[o]}"); axs[ci].set_xlabel("iteration"); axs[ci].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig_paired_reg_minus_r1.png", dpi=110); plt.close(fig)
for it in shared:
    a = {p[0]: p for p in pts["ppo_r2_reg"]}[it][4]; b = {p[0]: p for p in pts["ppo_r1"]}[it][4]
    cells = []
    for o in OPPS:
        d = a[o]["m"] - b[o]["m"]; cells.append(f"{d.mean():+.1f} ({d.std(ddof=1)/math.sqrt(len(d)):.1f})")
    d = np.mean([a[q]["m"] - b[q]["m"] for q in OPPS], axis=0)
    cells.append(f"{d.mean():+.1f} ({d.std(ddof=1)/math.sqrt(len(d)):.1f})")
    diffrows.append(f"| {it} | " + " | ".join(cells) + " |")
open(f"{D}/paired_table.md", "w").write("\n".join(diffrows))

# ---- critic diagnostics r1 vs r2_reg
def series(r, k):
    L = [x for x in load_log(r) if k in x and x[k] is not None and not (isinstance(x[k], float) and math.isnan(x[k]))]
    return np.array([x["iter"] for x in L]), np.array([x[k] for x in L], float)
def smooth(y, w=15):
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")
keys = [("vf_loss", True), ("expl_var", False), ("abs_adv", True), ("ret_mean", False), ("ret_std", False), ("v_mean", False), ("adv_std", True), ("kl", False), ("ent", False)]
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
for ax, (k, lg) in zip(axs.ravel(), keys):
    for r in ["ppo_r1", "ppo_r2_reg"]:
        x, y = series(r, k)
        if len(x) == 0: continue
        ax.plot(x, y, color=COL[r], alpha=.2, lw=.6); ax.plot(x, smooth(y), color=COL[r], lw=1.6, label=LABEL[r])
    if lg: ax.set_yscale("log")
    ax.set_title(k); ax.set_xlabel("iteration"); ax.grid(alpha=.3)
axs[0, 0].legend()
fig.suptitle("Critic / policy diagnostics from log.jsonl (thin = raw, thick = 15-iter moving average); expl_var, abs_adv, ret_* only logged for r2_reg")
fig.tight_layout(); fig.savefig(f"{D}/fig_critic_diag.png", dpi=110); plt.close(fig)
print(open(f"{D}/budget_table.md").read()); print(open(f"{D}/paired_table.md").read())
