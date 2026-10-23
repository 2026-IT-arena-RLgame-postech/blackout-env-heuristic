"""Tables of the velocity x pooling value-net A/B from examples/value_vp_predict.py output
(reports/value/vel_pool_ab.md).

    python examples/value_vp_analyze.py --preds reports/value/vel_pool_ab/preds --out reports/value/vel_pool_ab

Arms: base, vel (+Delta-pos), attn (attention pool), velattn; seeds s0..: files ``<arm>_s<seed>.npz``.
Per split (heldout, unseen_policy, unseen_ood) and subset (all, elapsed-time buckets, "contested" =
the 12-run mean P(win) in [0.2, 0.8], fixed rows for every arm): final-lead MAE / RMSE / R2 (points),
3-class result log loss, win Brier / ECE (all rows).  Mean +- seed sd per arm; factorial contrasts
(Delta-pos main effect, pool main effect, interaction) with seed-level SE = sd / sqrt(n_seeds) of the
per-seed contrasts; worst of the two unseen splits per arm and seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from blackout_env.value.metrics import classification  # noqa: E402

ARMS = ("base", "vel", "attn", "velattn")
SPLITS = ("heldout", "unseen_policy", "unseen_ood")
BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 45), (45, 1000))
BUCKET_NAMES = ("0-10s", "10-20s", "20-30s", "30-45s", ">45s")
HIGHER_BETTER = {"r2"}
METRICS = ("mae", "rmse", "r2", "logloss")


def subset_metrics(p, final_pts, y, res, with_win=False) -> dict:
    err = final_pts.astype(np.float64) - y
    sst = ((y - y.mean()) ** 2).sum()
    cls = res.astype(int) + 1
    out = {"n": int(len(y)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": float(1 - (err ** 2).sum() / sst) if sst > 0 else float("nan"),
           "logloss": float(-np.log(np.maximum(p[np.arange(len(cls)), cls], 1e-9)).mean())}
    if with_win:
        c = classification(p[:, 2].astype(np.float64), (res == 1).astype(np.float64), with_table=False)
        out.update(brier=c["brier"], ece=c["ece"], p99_abs=float(np.percentile(np.abs(err), 99)),
                   frac_err_gt40=float((np.abs(err) > 40).mean()))
    return out


def load(preds: Path):
    rows = np.load(preds / "rows.npz")
    runs = {}
    for f in sorted(preds.glob("*_s*.npz")):
        arm, seed = f.stem.rsplit("_s", 1)
        if arm in ARMS:
            runs[(arm, int(seed))] = np.load(f)
    return rows, runs


def compute(rows, runs) -> dict:
    seeds = sorted({s for _, s in runs})
    res = {}
    for split in SPLITS:
        y, r, el = rows[f"{split}/final_pts"], rows[f"{split}/result"], rows[f"{split}/elapsed_s"]
        ens = np.mean([run[f"{split}/p"][:, 2] for run in runs.values()], axis=0)
        masks = {"all": np.ones(len(y), bool), "contested": (ens >= 0.2) & (ens <= 0.8)}
        masks.update({n: (el >= a) & (el < b) for n, (a, b) in zip(BUCKET_NAMES, BUCKETS)})
        for (arm, seed), run in runs.items():
            for sub, m in masks.items():
                res.setdefault(split, {}).setdefault(sub, {}).setdefault(arm, {})[seed] = subset_metrics(
                    run[f"{split}/p"][m], run[f"{split}/final_pts"][m], y[m], r[m], with_win=sub == "all")
    return {"seeds": seeds, "res": res}


def ms(vals) -> tuple[float, float]:
    v = np.asarray(vals, np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def arm_values(res, split, sub, metric, arm):
    return [res[split][sub][arm][s][metric] for s in sorted(res[split][sub][arm])]


def contrasts(per_arm: dict[str, np.ndarray]) -> dict:
    """per-seed arrays by arm -> main effects / interaction (mean, SE) of the 2x2 design."""
    b, v, a, va = (per_arm[k] for k in ARMS)
    eff = {"dpos": ((v + va) - (b + a)) / 2, "pool": ((a + va) - (b + v)) / 2, "interaction": (va - a) - (v - b),
           "vel_alone": v - b, "attn_alone": a - b, "both": va - b}
    return {k: (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))
            for k, x in eff.items()}


def fmt(m, s, d=2):
    return f"{m:.{d}f}+-{s:.{d}f}"


def tables(out: dict) -> str:
    res, seeds = out["res"], out["seeds"]
    L = [f"Seeds: {seeds}.  mean +- seed sd per arm; effects = mean +- SE over seeds (paired by seed), "
         "negative = lower; for R2 positive = better.\n"]
    for split in SPLITS:
        n = res[split]["all"]["base"][seeds[0]]["n"]
        L += [f"### {split} (n={n})", "", "| arm | final MAE | final RMSE | final R2 | result logloss | win Brier | win ECE | p99 abs err | frac err>40 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            cells = []
            for k, d in (("mae", 2), ("rmse", 2), ("r2", 3), ("logloss", 3), ("brier", 4), ("ece", 3), ("p99_abs", 1), ("frac_err_gt40", 4)):
                cells.append(fmt(*ms(arm_values(res, split, "all", k, arm)), d))
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L.append("")
        L += ["Effects (all rows):", "", "| metric | dpos main | pool main | interaction | vel alone | attn alone | both vs base |", "|---|---|---|---|---|---|---|"]
        for k in METRICS + ("brier", "ece", "p99_abs"):
            c = contrasts({a: np.array(arm_values(res, split, "all", k, a)) for a in ARMS})
            L.append(f"| {k} | " + " | ".join(fmt(*c[e], 3 if k != "p99_abs" else 2) for e in ("dpos", "pool", "interaction", "vel_alone", "attn_alone", "both")) + " |")
        L.append("")
        for sub_title, subs in (("By elapsed time", BUCKET_NAMES), ("Contested (ensemble P(win) 0.2-0.8)", ("contested",))):
            L += [f"{sub_title} -- final MAE / R2 / result logloss per arm; effect of Delta-pos (main) and pool (main), +-SE:", "",
                  "| subset | n | " + " | ".join(f"{a} MAE" for a in ARMS) + " | dpos MAE | pool MAE | dpos R2 | pool R2 | dpos logloss | pool logloss |",
                  "|---|---|" + "---|" * (len(ARMS) + 6)]
            for sub in subs:
                n = res[split][sub]["base"][seeds[0]]["n"]
                arms_mae = [fmt(*ms(arm_values(res, split, sub, "mae", a))) for a in ARMS]
                cs = {k: contrasts({a: np.array(arm_values(res, split, sub, k, a)) for a in ARMS}) for k in ("mae", "r2", "logloss")}
                L.append(f"| {sub} | {n} | " + " | ".join(arms_mae) + " | " + " | ".join(
                    fmt(*cs[k][e], 3 if k != "mae" else 2) for k, e in (("mae", "dpos"), ("mae", "pool"), ("r2", "dpos"), ("r2", "pool"),
                                                                  ("logloss", "dpos"), ("logloss", "pool"))) + " |")
            L.append("")
    # worst of the two unseen splits per seed
    L += ["### Worst of the two unseen splits (per seed, then mean +- sd)", "", "| arm | MAE | RMSE | R2 | result logloss | p99 abs err |", "|---|---|---|---|---|---|"]
    worst = {}
    for arm in ARMS:
        cells = []
        for k, d in (("mae", 2), ("rmse", 2), ("r2", 3), ("logloss", 3), ("p99_abs", 1)):
            per = np.array([arm_values(res, sp, "all", k, arm) for sp in ("unseen_policy", "unseen_ood")])
            w = per.min(0) if k in HIGHER_BETTER else per.max(0)
            worst[(arm, k)] = w
            cells.append(fmt(*ms(w), d))
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "Effects on the worst-of-unseen value:", "", "| metric | dpos main | pool main | interaction |", "|---|---|---|---|"]
    for k in ("mae", "rmse", "r2", "logloss", "p99_abs"):
        c = contrasts({a: worst[(a, k)] for a in ARMS})
        L.append(f"| {k} | " + " | ".join(fmt(*c[e], 3) for e in ("dpos", "pool", "interaction")) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows, runs = load(args.preds)
    out = compute(rows, runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=1))
    md = tables(out)
    (args.out / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
