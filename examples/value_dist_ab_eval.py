"""Held-out evaluation of regression-final vs HL-Gauss-final value nets on the S1 row cache
(reports/value/dist_value_ab.md).

    python examples/value_dist_ab_eval.py --cache datasets/value_s1_r64 --out reports/value/dist_ab \
        --ckpt reg_s0=runs/reg_s0/latest.pt hlg_s0=runs/hlg_s0/latest.pt ...

Per checkpoint and split (heldout = held-out matches; unseen_policy / unseen_ood = never trained on):
final-margin error of the mean, calibration / log-loss of the result head, NLL + CRPS of the
distribution (HL-Gauss nets; regression nets get a Gaussian with the split's RMSE as reference),
and the Jensen quantity  E[tanh(final/30)] - tanh(E[final]/30)  with its effect on the GRPO leaf
reward R = winness + 0.2 tanh(.), plus how each reading tracks the realised tanh(true final / 30).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from blackout_env.train.accel import pick_device  # noqa: E402
from blackout_env.value.baselines import split_masks  # noqa: E402
from blackout_env.value.data import open_cache  # noqa: E402
from blackout_env.value.metrics import TIME_BUCKETS, bucket_names, classification, time_bucket  # noqa: E402
from blackout_env.value.model import REG_SCALE, bin_centers, decode_final_dist, load_value_net  # noqa: E402

SCALE = 30.0
LAM = 0.2
BIN_W_PTS = None


@torch.no_grad()
def predict(net, cache: dict, idx: np.ndarray, device, batch: int = 1024) -> dict:
    parts = []
    for a in range(0, len(idx), batch):
        j = idx[a:a + batch]
        g = torch.from_numpy(np.asarray(cache["graphic"][j])).to(device)
        ts = torch.from_numpy(np.asarray(cache["team_state"][j])).to(device)
        ag = torch.from_numpy(np.asarray(cache["agent_states"][j])).to(device)
        out, logits = net.forward_full_u8(g, ts, ag)
        out = out.float()
        h = net.n_horizons
        lead = ts[:, 0] - ts[:, 1]
        r = {"p": torch.softmax(out[:, h + 1:], -1).cpu().numpy(),
             "mean": ((lead * 100) + out[:, h] * (100 / REG_SCALE)).cpu().numpy()}
        if logits is not None:
            d = decode_final_dist(logits, lead, net.n_bins, net.bin_range, SCALE)
            r.update(pmf=d["pmf"], tanh_mean=d["tanh_mean"], sd=d["sd"] * 100)
        parts.append(r)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def gauss_nll_crps(y, mu, sigma):
    z = (y - mu) / sigma
    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return float(nll.mean()), float(crps.mean())


def dist_nll_crps(pmf, lead_pts, y, n_bins, bin_range):
    """NLL (per point of margin: bin mass / bin width) and CRPS (points) of the HL-Gauss pmf over the final lead."""
    w = 2 * bin_range / n_bins * (100 / REG_SCALE)
    c = bin_centers(n_bins, bin_range).numpy() * (100 / REG_SCALE)
    k = np.clip(np.floor((y - lead_pts + bin_range * 10) / w).astype(int), 0, n_bins - 1)
    mass = pmf[np.arange(len(y)), k]
    nll = float((-np.log(np.maximum(mass, 1e-9) / w)).mean())
    x = lead_pts[:, None] + c[None]
    cdf = np.cumsum(pmf, axis=1)
    crps = float((((cdf - (x >= y[:, None])) ** 2) * w).sum(1).mean())
    return nll, crps


def summarize(v: np.ndarray) -> dict:
    a = np.abs(v)
    return {"mean": float(v.mean()), "mean_abs": float(a.mean()), "p50_abs": float(np.percentile(a, 50)),
            "p90_abs": float(np.percentile(a, 90)), "p99_abs": float(np.percentile(a, 99)),
            "max_abs": float(a.max()), "sd": float(v.std())}


def pair_stats(match, tom, gap, tm, rng, max_pairs=200_000):
    """Within-match pairs of similar states (|d tanh(E)| < 0.1): mean |d gap| and sign-flip rate of the ordering."""
    order = np.argsort(match, kind="stable")
    m = match[order]
    cuts = np.flatnonzero(np.diff(m)) + 1
    d_gap, flips, n_close = [], 0, 0
    close_tot = 0
    for g in np.split(order, cuts):
        if len(g) < 2:
            continue
        i, j = np.triu_indices(len(g), 1)
        i, j = g[i], g[j]
        dt = tom[i] - tom[j]
        close = np.abs(dt) < 0.1
        d_gap.append(np.abs(gap[i] - gap[j])[close])
        near = np.abs(dt) < 0.05
        flips += int((np.sign(tm[i] - tm[j])[near] != np.sign(dt[near])).sum())
        n_close += int(near.sum())
        close_tot += int(close.sum())
    d_gap = np.concatenate(d_gap) if d_gap else np.zeros(1)
    return {"pairs_close": close_tot, "mean_abs_d_gap": float(d_gap.mean()), "p90_abs_d_gap": float(np.percentile(d_gap, 90)),
            "flip_rate_near": flips / max(1, n_close), "n_near": n_close}


def evaluate_split(net, cache, idx, device) -> dict:
    p = predict(net, cache, idx, device)
    ts = np.asarray(cache["team_state"][idx])
    lead = (ts[:, 0] - ts[:, 1]).astype(np.float64) * 100
    y = np.asarray(cache["final"][idx]).astype(np.float64) * 100
    res = np.asarray(cache["result"][idx])
    el = np.asarray(cache["elapsed_s"][idx])
    match = np.asarray(cache["match"][idx])
    mean = p["mean"].astype(np.float64)
    err = mean - y
    out = {"rows": int(len(idx)), "matches": int(len(np.unique(match))),
           "final_mae": float(np.abs(err).mean()), "final_rmse": float(np.sqrt((err ** 2).mean())),
           "final_bias": float(err.mean()), "final_r2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "final_mae_median_row_by_match": None}
    # result head: 3-class log loss, win Brier/ECE
    pr = p["p"].astype(np.float64)
    cls = res.astype(int) + 1
    out["result_logloss"] = float(-np.log(np.maximum(pr[np.arange(len(cls)), cls], 1e-9)).mean())
    out["win"] = {k: v for k, v in classification(pr[:, 2], (res == 1).astype(np.float64)).items() if k != "calibration"}
    out["win_calibration"] = classification(pr[:, 2], (res == 1).astype(np.float64))["calibration"]
    ty = np.tanh(y / SCALE)
    tom = np.tanh(mean / SCALE)
    out["tanh_mse_of_mean"] = float(((tom - ty) ** 2).mean())
    out["tanh_target_var"] = float(ty.var())
    sigma = out["final_rmse"]
    out["gauss_ref_nll"], out["gauss_ref_crps"] = gauss_nll_crps(y, mean, sigma)
    if "pmf" in p:
        n_bins, bin_range = net.n_bins, net.bin_range
        out["dist_nll"], out["dist_crps"] = dist_nll_crps(p["pmf"].astype(np.float64), lead, y, n_bins, bin_range)
        tm = p["tanh_mean"].astype(np.float64)
        gap = tm - tom
        out["tanh_mse_of_dist"] = float(((tm - ty) ** 2).mean())
        out["gap"] = summarize(gap)
        out["gap_R"] = summarize(LAM * gap)  # difference in the GRPO leaf reward R
        out["sd_pts_mean"] = float(p["sd"].mean())
        # implied win prob from the pmf vs the result head
        c = bin_centers(n_bins, bin_range).numpy() * (100 / REG_SCALE)
        pw = (p["pmf"].astype(np.float64) * ((lead[:, None] + c[None]) > 0)).sum(1)
        out["pwin_pmf_vs_head_mae"] = float(np.abs(pw - pr[:, 2]).mean())
        out["pwin_pmf_logloss"] = classification(pw, (res == 1).astype(np.float64), with_table=False)["logloss"]
        out["pwin_head_logloss"] = classification(pr[:, 2], (res == 1).astype(np.float64), with_table=False)["logloss"]
        bk = time_bucket(el)
        out["gap_by_phase"] = {n: {**summarize(gap[bk == b]), "n": int((bk == b).sum())}
                               for b, n in enumerate(bucket_names()) if (bk == b).any()}
        qs = np.percentile(p["sd"], [0, 25, 50, 75, 100])
        sb = np.clip(np.searchsorted(qs, p["sd"], side="right") - 1, 0, 3)
        out["gap_by_sd_quartile"] = {f"Q{q + 1} sd {qs[q]:.1f}-{qs[q + 1]:.1f}": {**summarize(gap[sb == q]), "n": int((sb == q).sum())}
                                     for q in range(4)}
        pb = np.digitize(pr[:, 2], [0.2, 0.8])
        out["gap_by_pwin"] = {n: {**summarize(gap[pb == i]), "n": int((pb == i).sum())}
                              for i, n in enumerate(("pwin<0.2", "0.2-0.8", "pwin>0.8")) if (pb == i).any()}
        out["pairs"] = pair_stats(match, tom, gap, tm, np.random.default_rng(0))
        # empirical comparison: bucket by tanh(E[final]) and compare each reading's bucket mean with the realised tanh(true)
        edges = np.linspace(-1, 1, 11)
        b = np.clip(np.digitize(tom, edges) - 1, 0, 9)
        rows = []
        for i in range(10):
            s = b == i
            if s.sum() >= 20:
                rows.append({"bin": f"{edges[i]:+.1f}..{edges[i + 1]:+.1f}", "n": int(s.sum()), "realised": float(ty[s].mean()),
                             "tanh_of_mean": float(tom[s].mean()), "mean_of_tanh": float(tm[s].mean())})
        out["empirical_by_tanh_bin"] = rows
        n = np.array([r["n"] for r in rows])
        for key in ("tanh_of_mean", "mean_of_tanh"):
            out[f"empirical_bin_mae_{key}"] = float((n * np.abs([r[key] - r["realised"] for r in rows])).sum() / n.sum())
        # same, bucketed by the distribution's own E[tanh]
        b2 = np.clip(np.digitize(tm, edges) - 1, 0, 9)
        rows2 = []
        for i in range(10):
            s = b2 == i
            if s.sum() >= 20:
                rows2.append({"n": int(s.sum()), "realised": float(ty[s].mean()), "mean_of_tanh": float(tm[s].mean())})
        n2 = np.array([r["n"] for r in rows2])
        out["empirical_bin_mae_mean_of_tanh_own_bins"] = float((n2 * np.abs([r["mean_of_tanh"] - r["realised"] for r in rows2])).sum() / n2.sum())
    else:
        # regression net: calibration of tanh(E) by bins (same table for the comparison)
        edges = np.linspace(-1, 1, 11)
        b = np.clip(np.digitize(tom, edges) - 1, 0, 9)
        rows = [{"n": int((b == i).sum()), "realised": float(ty[b == i].mean()), "tanh_of_mean": float(tom[b == i].mean())}
                for i in range(10) if (b == i).sum() >= 20]
        n = np.array([r["n"] for r in rows])
        out["empirical_bin_mae_tanh_of_mean"] = float((n * np.abs([r["tanh_of_mean"] - r["realised"] for r in rows])).sum() / n.sum())
    bk = time_bucket(el)
    out["final_mae_by_phase"] = {n_: float(np.abs(err[bk == b]).mean()) for b, n_ in enumerate(bucket_names()) if (bk == b).any()}
    out.pop("final_mae_median_row_by_match")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=60_000, help="per split (heldout is smaller)")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    cache = open_cache(args.cache)
    matches = json.loads((args.cache / "matches.json").read_text())
    rng = np.random.default_rng(0)
    idxs = {}
    for name, mask in split_masks(cache, matches).items():
        if name == "unseen":
            continue
        idx = np.flatnonzero(mask)
        if len(idx) > args.max_rows:
            idx = np.sort(rng.choice(idx, args.max_rows, replace=False))
        idxs[name] = idx
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        net = load_value_net(path, device)
        res = {s: evaluate_split(net, cache, idx, device) for s, idx in idxs.items()}
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=1))
        h = res["heldout"]
        print(f"{name}: heldout final MAE {h['final_mae']:.2f} RMSE {h['final_rmse']:.2f} result logloss {h['result_logloss']:.3f}"
              + (f" dist NLL {h['dist_nll']:.3f} CRPS {h['dist_crps']:.2f} gap mean|.| {h['gap']['mean_abs']:.4f}" if "gap" in h else ""),
              flush=True)


if __name__ == "__main__":
    main()
