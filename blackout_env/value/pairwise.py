"""Pairwise test: does V rank *alternative trajectories of the same start* correctly?

A search leaf evaluator compares states reached from one root by different plans, so what matters
is not R² across all states (dominated by between-match differences) but whether V orders two
states of the same situation by their outcome.  The held-out suites play many learner policies
on the same (seed, side, opponent): same map, same opponent, different play.  At aligned match
times t, for every pair of such matches still running, the test asks whether the sign of the
predicted difference matches the sign of the realised difference in

  * final lead (prediction: V's final lead, or P(win) when "win"), and
  * lead change over the next 20 s (prediction: V's 20 s head),

reported as concordance (ties in the prediction count 1/2; pairs with equal outcomes are
skipped), overall and on "lead-tied" pairs whose current lead is equal (where the score alone
cannot separate them).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.value.targets import DECISIONS_PER_S, HORIZONS_S, match_targets, shard_ticks

TIMES_S = (5, 10, 15, 20, 30, 45, 60, 90)
H20 = HORIZONS_S.index(20)


def aligned_rows(repo: Path, entries: list[dict], times_s=TIMES_S) -> dict:
    """Rows of every match at each time in ``times_s`` (the last row at or before it, within 1 s),
    with their match key and targets."""
    out = defaultdict(list)
    for m_idx, entry in enumerate(entries):
        with np.load(Path(repo) / entry["path"]) as shard:
            ts = shard["team_state"]
            ticks = shard_ticks(shard)
            if ticks[0] < 0:
                continue
            targets = match_targets(ts, ticks, entry)
            want = np.array(times_s) * DECISIONS_PER_S
            j = np.searchsorted(ticks, want, side="right") - 1
            ok = (j >= 0) & (want - ticks[np.clip(j, 0, None)] <= DECISIONS_PER_S) & (want <= ticks[-1])
            if not ok.any():
                continue
            sel = j[ok]
            out["graphic"].append(shard["graphic"][sel])
            out["agent_states"].append(shard["agent_states"][sel])
        out["team_state"].append(ts[sel])
        out["time_s"].append(np.array(times_s)[ok])
        out["match"].append(np.full(len(sel), m_idx))
        out["final"].append(targets["final"][sel])
        out["delta20"].append(targets["delta"][sel, H20])
        out["delta20_mask"].append(targets["delta_mask"][sel, H20])
        out["won"].append((targets["result"][sel] == 1).astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def concordance(rows: dict, pred: dict, entries: list[dict]) -> dict:
    """``pred``: {"final": [N], "p_win": [N], "delta20": [N]} for ``rows`` (aligned_rows)."""
    key_of = [(e.get("seed"), e.get("swap"), e.get("opponent")) for e in entries]
    groups = defaultdict(list)
    for i, (m, t) in enumerate(zip(rows["match"], rows["time_s"])):
        groups[(key_of[m], int(t))].append(i)
    lead = rows["team_state"][:, 0] - rows["team_state"][:, 1]
    tallies = defaultdict(lambda: [0.0, 0])

    def add(name, p_a, p_b, y_a, y_b):
        if abs(y_a - y_b) < 1e-6:
            return
        d = p_a - p_b
        tallies[name][0] += 0.5 if abs(d) < 1e-9 else float(np.sign(d) == np.sign(y_a - y_b))
        tallies[name][1] += 1

    for (_, t), idx in groups.items():
        for a, b in itertools.combinations(idx, 2):
            tied = abs(lead[a] - lead[b]) < 1e-4
            for target, y in (("final", rows["final"]), ("win", rows["won"])):
                for pname in ("final", "p_win"):
                    for scope in ("all", "lead_tied") if tied else ("all",):
                        for tag in (f"{target}|{pname}|{scope}", f"{target}|{pname}|{scope}|t{t}"):
                            add(tag, pred[pname][a], pred[pname][b], y[a], y[b])
            if rows["delta20_mask"][a] and rows["delta20_mask"][b]:
                for scope in ("all", "lead_tied") if tied else ("all",):
                    add(f"delta20|delta20|{scope}", pred["delta20"][a], pred["delta20"][b],
                        rows["delta20"][a], rows["delta20"][b])
    return {k: {"concordance": v[0] / v[1], "pairs": v[1]} for k, v in sorted(tallies.items()) if v[1]}


def smoothness(predict, repo: Path, entries: list[dict], max_ticks: int = 750) -> dict:
    """How much V moves from one decision tick to the next when nothing scored.

    ``predict(rows) -> {"final": [N], "delta": [N, H]}`` on consecutive rows of each match (the
    first ``max_ticks`` contiguous ticks).  Reports, in points, the median and 90th percentile of
    |V(t+1) - V(t)| for the final-lead prediction on ticks where the lead did not change, next to the
    spread of V over the next 1 and 2 s (std of V(t+k) - V(t), k = 25 / 50 ticks): a leaf search
    that compares states 1-2 s apart needs real differences larger than this jitter."""
    steps, spread1, spread2 = [], [], []
    for entry in entries:
        with np.load(Path(repo) / entry["path"]) as shard:
            if "tick" not in shard.files:
                continue
            n = min(len(shard["tick"]), max_ticks)
            ticks = shard["tick"][:n]
            if n < 60 or not np.all(np.diff(ticks) == 1):
                continue
            rows = {k: shard[k][:n] for k in ("graphic", "team_state", "agent_states")}
        v = predict(rows)["final"] * 100
        lead = (rows["team_state"][:, 0] - rows["team_state"][:, 1]) * 100
        quiet = np.abs(np.diff(lead)) < 1e-3
        steps.append(np.abs(np.diff(v - lead))[quiet])
        spread1.append(v[25:] - v[:-25])
        spread2.append(v[50:] - v[:-50])
    steps = np.concatenate(steps)
    return {"matches": len(spread1), "tick_jitter_median": float(np.median(steps)),
            "tick_jitter_p90": float(np.percentile(steps, 90)),
            "v_change_1s_sd": float(np.concatenate(spread1).std()), "v_change_2s_sd": float(np.concatenate(spread2).std())}
