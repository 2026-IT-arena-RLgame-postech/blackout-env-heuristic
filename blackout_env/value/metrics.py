"""S1 metrics: lead-change regression per horizon and match-time bucket, win-probability quality.

``evaluate(pred, cache_rows)`` takes predictions for a set of cache rows:

  delta   [N, H]  predicted lead change per horizon (score / 100)
  final   [N]     predicted final lead (score / 100)
  p_win   [N]     predicted P(learner wins)
  p       [N, 3]  optional loss / draw / win probabilities

and returns a JSON-able dict: per horizon R² and MAE (in points) overall and per time bucket,
the same for the final lead and for its remaining change (final - lead now), and for the win
probability Brier score, accuracy (p_win > 0.5 vs won), the base-rate Brier and a calibration
table (10 bins), overall and per time bucket.
"""

from __future__ import annotations

import numpy as np

from blackout_env.value.targets import HORIZONS_S

TIME_BUCKETS = (0, 10, 20, 30, 45, 60, 120, 421)
CAL_BINS = np.linspace(0.0, 1.0, 11)


def bucket_names() -> list[str]:
    return [f"{a}-{min(b, 420)}s" for a, b in zip(TIME_BUCKETS[:-1], TIME_BUCKETS[1:])]


def time_bucket(elapsed_s: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(TIME_BUCKETS, elapsed_s, side="right") - 1, 0, len(TIME_BUCKETS) - 2)


def regression(pred: np.ndarray, target: np.ndarray) -> dict:
    """R² (vs the subset's own mean) and MAE in points (target units x 100)."""
    n = len(target)
    if n < 2:
        return {"n": int(n)}
    err = pred.astype(np.float64) - target
    sst = float(((target - target.mean()) ** 2).sum())
    return {"n": int(n), "r2": float(1.0 - (err ** 2).sum() / sst) if sst > 0 else float("nan"),
            "mae": float(np.abs(err).mean() * 100), "rmse": float(np.sqrt((err ** 2).mean()) * 100),
            "target_sd": float(target.std() * 100)}


def calibration(p: np.ndarray, won: np.ndarray) -> list[dict]:
    idx = np.clip(np.digitize(p, CAL_BINS) - 1, 0, len(CAL_BINS) - 2)
    out = []
    for b in range(len(CAL_BINS) - 1):
        sel = idx == b
        if sel.any():
            out.append({"bin": f"{CAL_BINS[b]:.1f}-{CAL_BINS[b + 1]:.1f}", "n": int(sel.sum()),
                        "pred": float(p[sel].mean()), "won": float(won[sel].mean())})
    return out


def classification(p_win: np.ndarray, won: np.ndarray, with_table: bool = True) -> dict:
    n = len(won)
    if n == 0:
        return {"n": 0}
    rate = won.mean()
    ece = sum(c["n"] * abs(c["pred"] - c["won"]) for c in calibration(p_win, won)) / n
    out = {"n": int(n), "brier": float(((p_win - won) ** 2).mean()), "base_brier": float(rate * (1 - rate)),
           "acc": float(((p_win > 0.5) == won).mean()), "base_acc": float(max(rate, 1 - rate)),
           "win_rate": float(rate), "ece": float(ece),
           "logloss": float(-np.mean(won * np.log(np.clip(p_win, 1e-6, 1)) + (1 - won) * np.log(np.clip(1 - p_win, 1e-6, 1))))}
    if with_table:
        out["calibration"] = calibration(p_win, won)
    return out


def evaluate(pred: dict, rows: dict, horizons_s=HORIZONS_S) -> dict:
    """``rows``: the cache arrays (or slices) for the predicted rows: delta, delta_mask, final,
    result, elapsed_s, team_state."""
    bucket = time_bucket(np.asarray(rows["elapsed_s"]))
    names = bucket_names()
    lead = np.asarray(rows["team_state"])[:, 0] - np.asarray(rows["team_state"])[:, 1]
    final = np.asarray(rows["final"], np.float64)
    won = (np.asarray(rows["result"]) == 1).astype(np.float64)
    report: dict = {"rows": int(len(bucket)), "delta": {}, "buckets": names}
    for h, seconds in enumerate(horizons_s):
        mask = np.asarray(rows["delta_mask"])[:, h]
        target = np.asarray(rows["delta"])[:, h].astype(np.float64)
        p = np.asarray(pred["delta"])[:, h]
        entry = {"all": regression(p[mask], target[mask])}
        for b, name in enumerate(names):
            sel = mask & (bucket == b)
            entry[name] = regression(p[sel], target[sel])
        report["delta"][f"{seconds}s"] = entry
    for key, target, p in (("final", final, np.asarray(pred["final"])),
                           ("final_remaining", final - lead, np.asarray(pred["final"]) - lead)):
        entry = {"all": regression(p, target)}
        for b, name in enumerate(names):
            entry[name] = regression(p[bucket == b], target[bucket == b])
        report[key] = entry
    p_win = np.asarray(pred["p_win"], np.float64)
    win = {"all": classification(p_win, won)}
    for b, name in enumerate(names):
        win[name] = classification(p_win[bucket == b], won[bucket == b])
    report["win"] = win
    return report


def format_report(report: dict) -> str:
    """Compact text tables: {model: {split: evaluate(...)}} or a single evaluate(...) result."""
    if "delta" in report:
        report = {"model": {"split": report}}
    lines = []
    for model, splits in report.items():
        for split, r in splits.items():
            if not isinstance(r, dict) or "delta" not in r:
                continue
            lines.append(f"== {model} / {split}  ({r['rows']} rows)")
            head = "  target        " + "".join(f"{'all':>14}" if i == 0 else f"{b:>14}" for i, b in enumerate(["all"] + r["buckets"]))
            lines.append(head)
            for key, entry in list(r["delta"].items()) + [("final", r["final"]), ("final_rem", r["final_remaining"])]:
                cells = []
                for b in ["all"] + r["buckets"]:
                    m = entry.get(b, {})
                    cells.append(f"{m['r2']:6.3f}/{m['mae']:5.2f}" if "r2" in m else f"{'-':>12}")
                lines.append(f"  {key:<12}" + "".join(f"{c:>14}" for c in cells) + "   (R²/MAE pts)")
            cells = []
            for b in ["all"] + r["buckets"]:
                m = r["win"].get(b, {})
                cells.append(f"{m['brier']:.3f}/{m['acc']:.3f}" if m.get("n") else f"{'-':>12}")
            lines.append(f"  {'win':<12}" + "".join(f"{c:>14}" for c in cells) + "   (Brier/acc)")
            w = r["win"]["all"]
            lines.append(f"  win: base Brier {w['base_brier']:.3f} base acc {w['base_acc']:.3f} ECE {w['ece']:.3f} "
                         f"logloss {w['logloss']:.3f}")
    return "\n".join(lines)
