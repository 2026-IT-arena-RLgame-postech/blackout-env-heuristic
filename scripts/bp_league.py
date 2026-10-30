"""Elo (Bradley-Terry) + win-rate / margin heatmaps of the post-patch heuristic pool from econ_stats results.

    PYTHONPATH=. .venv/bin/python scripts/bp_league.py --names V33bp V34a V34b [--extra B2=<desc> ...] \
        --dirs reports/bp/x11 reports/bp/league --out reports/bp/league

Rows are matched by descriptor (aliases resolved).  Each (row, col) cell pools both orientations (row as ours and as
opponent).  A draw scores 0.5.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from blackout_env.heuristics.params import resolve_descriptor  # noqa: E402
from blackout_env.rating import ELO, fit_bt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=[])
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=descriptor")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    named = {}
    for n in a.names:
        named[n] = n
    for e in a.extra:
        k, v = e.split("=", 1)
        named[k] = v
    labels = list(named)
    key = {}
    for n, d in named.items():
        key[d] = n
        try:
            key[resolve_descriptor(d)] = n
        except Exception:
            pass
    k = len(labels)
    idx = {n: i for i, n in enumerate(labels)}
    margin_sum = np.zeros((k, k))
    win_sum = np.zeros((k, k))
    games = np.zeros((k, k), int)
    I, J, S = [], [], []
    for d in a.dirs:
        for f in glob.glob(f"{d}/**/matches.jsonl", recursive=True):
            for line in open(f):
                r = json.loads(line)
                a_, b_ = key.get(r["ours"]), key.get(r["opponent"])
                if a_ is None or b_ is None or a_ == b_:
                    continue
                i, j = idx[a_], idx[b_]
                m = r["margin"]
                s = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
                for (p, q, mm, ss) in ((i, j, m, s), (j, i, -m, 1 - s)):
                    margin_sum[p, q] += mm
                    win_sum[p, q] += ss
                    games[p, q] += 1
                I.append(i)
                J.append(j)
                S.append(s)
    r, cov, _, _ = fit_bt(k, I, J, S)
    elo = r * ELO
    se = np.sqrt(np.clip(np.diag(cov), 0, None)) * ELO
    with np.errstate(invalid="ignore", divide="ignore"):
        mmat = margin_sum / games
        wmat = win_sum / games
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-elo)
    lines = ["| rank | policy | Elo | ±SE | mean margin | games |", "|---|---|---|---|---|---|"]
    for rank, i in enumerate(order, 1):
        lines.append(f"| {rank} | {labels[i]} | {elo[i]:+.0f} | {se[i]:.0f} | {np.nanmean(np.where(games[i] > 0, mmat[i], np.nan)):+.1f} | {games[i].sum()} |")
    txt = "\n".join(lines)
    print(txt)
    (out / "elo.md").write_text(txt + "\n")
    try:
        from elo24 import _heatmap
        lab = [labels[i] for i in order]
        sub = np.ix_(order, order)
        _heatmap(out / "winrate.png", lab, wmat[sub], games[sub], None, "post-patch pool: win rate")
        _heatmap(out / "margin.png", lab, mmat[sub], games[sub], None, "post-patch pool: mean margin", margin=True)
    except Exception as e:  # plotting is optional
        print("heatmap skipped:", e)


if __name__ == "__main__":
    main()
