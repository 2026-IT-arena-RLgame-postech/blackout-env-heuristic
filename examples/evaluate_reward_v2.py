"""
Unit-level evaluation of reward v2 against heuristic play (docs/reward_v2_design.md §5).

E1  one-step ranking: for a unit-tick in heuristic data, move that unit one decision's distance in
    each of the 8 directions (others and the map unchanged) and rank the heuristic's actual
    direction by the unit's own shaped credit. Reported by role, against the 4.5 average rank of a
    random direction; ticks where all 8 directions score the same are counted as "no signal".
E2  events: the unit's credit on the tick of pickups, deliveries, steals, deaths (empty/carrying,
    before/after the field runs dry), kills by own Hunters, transformations, and absorptions.

Both read an offline dataset (stream A, heuristic-vs-heuristic) and need no Unity.

Usage:
    python examples/evaluate_reward_v2.py --dataset-dir datasets/<name> [--e1-samples 1500]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.train.reward_v2 import (
    BATTERY_SCALE, CARGO_COL, CLASS_COLS, HUNTER, SPEED, H, W, Potentials, RewardV2Config,
    compute_potentials, geometry_for, score_reward, shaped_rewards, unit_positions,
)
from probe_boundary_chatter import npz_member_memmap

TICK_SECONDS = 1.0 / 25.0


def slice_potentials(p: Potentials, index) -> Potentials:
    return Potentials(p.unit[index], p.common[index], p.confirmed[index], p.hunter_of[index], p.own[index],
                      {k: v[index] for k, v in p.terms.items()})


def potentials_chunked(graphic, states, team, cfg, chunk=2048) -> Potentials:
    parts = [compute_potentials(graphic[i : i + chunk], states[i : i + chunk], team[i : i + chunk], cfg) for i in range(0, len(graphic), chunk)]
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
    return Potentials(cat("unit"), cat("common"), cat("confirmed"), cat("hunter_of"), cat("own"),
                      {k: np.concatenate([p.terms[k] for p in parts]) for k in parts[0].terms})


# ------------------------------------------------------------------------------------------ E1


def e1(path: Path, samples: int, cfg: RewardV2Config, rng: np.random.Generator) -> None:
    graphic_mm, states_mm = npz_member_memmap(path, "graphic"), npz_member_memmap(path, "agent_states")
    team_mm, actions_mm = npz_member_memmap(path, "team_state"), npz_member_memmap(path, "actions")
    # only ticks while the field still matters (the first ~60 s decide the match)
    pool = np.sort(rng.choice(len(states_mm), samples * 4, replace=False))
    econ_ok = np.asarray(team_mm[pool])[:, 2] > 1.0 - 60.0 / 420.0
    idx = pool[econ_ok][:samples]
    G, S, T, A = (np.asarray(graphic_mm[idx]), np.asarray(states_mm[idx]), np.asarray(team_mm[idx]), np.asarray(actions_mm[idx]))
    geo = geometry_for(G[0])
    base = potentials_chunked(G, S, T, cfg)

    angles = np.arange(8) * math.pi / 4
    dx, dy = np.cos(angles), np.sin(angles)
    stats = defaultdict(lambda: {"n": 0, "rank": 0.0, "top1": 0, "top3": 0, "flat": 0})
    for u in range(5):
        cls = S[:, u, CLASS_COLS].argmax(-1)
        step = SPEED[cls] * TICK_SECONDS
        row, col = unit_positions(S[:, u : u + 1])
        moved = np.repeat(S[:, None], 8, 1).copy()  # [B, 8, 10, 12]
        new_row = np.clip(row + (-dy[None, :] * step[:, None]), 0, H - 1)
        new_col = np.clip(col + (dx[None, :] * step[:, None]), 0, W - 1)
        blocked = ~geo.walkable[0][np.rint(new_row).astype(int), np.rint(new_col).astype(int)]
        new_row = np.where(blocked, row, new_row)
        new_col = np.where(blocked, col, new_col)
        moved[:, :, u, 0] = (new_col + 0.5) * 2.0 / W - 1.0
        moved[:, :, u, 1] = 1.0 - (new_row + 0.5) * 2.0 / H
        batch = len(idx)
        after = potentials_chunked(np.repeat(G, 8, 0), moved.reshape(batch * 8, 10, 12), np.repeat(T, 8, 0), cfg)
        before = slice_potentials(base, np.repeat(np.arange(batch), 8))
        credit = shaped_rewards(before, after, 1.0)[0][:, u].reshape(batch, 8)

        cargo = S[:, u, CARGO_COL] > 0
        fetching = base.terms["fetch"][:, u] > 0
        hunting = base.terms["hunt_exit"][:, u] > 0
        role = np.where(cargo, "carry", np.where(fetching, "fetch", np.where(cls == HUNTER, np.where(hunting, "hunt/exit", "hunter-idle"), "other")))
        chosen = A[:, u]
        chosen_credit = credit[np.arange(batch), chosen]
        higher = (credit > chosen_credit[:, None] + 1e-6).sum(1)
        ties = (np.abs(credit - chosen_credit[:, None]) <= 1e-6).sum(1)
        rank = 1 + higher + (ties - 1) / 2.0
        flat = (credit.max(1) - credit.min(1)) <= 1e-6
        for r in np.unique(role):
            m = role == r
            s = stats[r]
            s["n"] += int(m.sum()); s["flat"] += int((m & flat).sum())
            informative = m & ~flat
            s["rank"] += float(rank[informative].sum())
            s["top1"] += int((informative & (higher == 0)).sum())
            s["top3"] += int((informative & (higher <= 2)).sum())

    print("\nE1 one-step ranking of the heuristic's direction (1 = best of 8; random = 4.5, top-3 random = 37.5%)")
    print(f"{'role':12s} {'unit-ticks':>10s} {'no signal':>9s} {'mean rank':>9s} {'top-1':>6s} {'top-3':>6s}")
    for r, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        informative = s["n"] - s["flat"]
        if informative == 0:
            print(f"{r:12s} {s['n']:10d} {s['flat'] / max(1, s['n']):9.0%}")
            continue
        print(f"{r:12s} {s['n']:10d} {s['flat'] / s['n']:9.0%} {s['rank'] / informative:9.2f} {s['top1'] / informative:6.0%} {s['top3'] / informative:6.0%}")


# ------------------------------------------------------------------------------------------ E2


def e2(path: Path, ticks: int, cfg: RewardV2Config, start: int) -> None:
    graphic_mm, states_mm = npz_member_memmap(path, "graphic"), npz_member_memmap(path, "agent_states")
    team_mm, done_mm = npz_member_memmap(path, "team_state"), npz_member_memmap(path, "done")
    sl = slice(start, start + ticks)
    G, S, T = np.asarray(graphic_mm[sl]), np.asarray(states_mm[sl]), np.asarray(team_mm[sl])
    P = potentials_chunked(G, S, T, cfg)
    before, after = slice_potentials(P, slice(0, ticks - 1)), slice_potentials(P, slice(1, ticks))
    per_unit, team = shaped_rewards(before, after, 1.0)
    score = score_reward(before, after)
    reset = T[1:, 2] > T[:-1, 2] + 1e-4  # any rise: early-ended matches reset from wherever they were
    per_unit[reset] = 0
    team = np.where(reset, 0, team)
    score = np.where(reset, 0, score)

    geo = geometry_for(G[0])
    row, col = unit_positions(S)
    cls = S[..., CLASS_COLS].argmax(-1)
    cargo = np.rint(np.maximum(S[..., CARGO_COL], 0) * BATTERY_SCALE)
    storage_own, storage_enemy = G[..., 6] > 0.5, G[..., 7] > 0.5
    econ = P.terms["econ"][:, 0]
    jump = np.hypot(row[1:] - row[:-1], col[1:] - col[:-1])
    spawn_r, spawn_c = geo.spawn[0] // W, geo.spawn[0] % W
    enemy_spawn_r, enemy_spawn_c = geo.spawn[1] // W, geo.spawn[1] % W

    events = defaultdict(list)
    for t in range(ticks - 1):
        if reset[t]:
            continue
        late = "field empty" if econ[t] < 0.05 else "field live"
        for u in range(5):
            r0, c0 = int(round(row[t, u])), int(round(col[t, u]))
            died = jump[t, u] > 1.5 and math.hypot(row[t + 1, u] - spawn_r, col[t + 1, u] - spawn_c) < 1.5
            credit = float(per_unit[t, u])
            if died:
                kind = f"death carrying ({late})" if cargo[t, u] > 0 else f"death empty-handed {'Hunter' if cls[t, u] == HUNTER else 'non-Hunter'} ({late})"
                events[kind].append(credit)
                continue
            if cargo[t, u] == 0 and cargo[t + 1, u] > 0:
                events["steal (pickup on enemy storage)" if storage_enemy[t, r0, c0] else "pickup"].append(credit)
            elif cargo[t, u] > 0 and cargo[t + 1, u] == 0:
                events["delivery" if storage_own[t, r0, c0] or storage_own[t + 1, r0, c0] else "cargo gone (not delivered)"].append(credit)
            if cls[t + 1, u] != cls[t, u] and not died:
                events[f"transform -> {['Collector', 'Hunter', 'Carrier'][cls[t + 1, u]]} ({late})"].append(credit)
        for j in range(5, 10):
            if jump[t, j] > 1.5 and math.hypot(row[t + 1, j] - enemy_spawn_r, col[t + 1, j] - enemy_spawn_c) < 1.5:
                hunter = int(before.hunter_of[t, j])
                label = f"enemy {'carrier' if cargo[t, j] > 0 else 'empty unit'} killed ({late})"
                events[label + " -> assigned Hunter"].append(float(per_unit[t, hunter]) if hunter >= 0 else float("nan"))
                events[label + " -> team"].append(float(team[t]))
        if T[t + 1, 3] > T[t, 3] + 0.5:
            events["absorption -> team shaping + score"].append(float(team[t] + score[t]))

    print(f"\nE2 unit credit on the event tick (points, gamma = 1), ticks {start}..{start + ticks}")
    print(f"{'event':52s} {'n':>5s} {'mean':>7s} {'median':>7s} {'>0':>5s}")
    for kind, values in sorted(events.items()):
        v = np.array([x for x in values if np.isfinite(x)])
        if len(v) == 0:
            print(f"{kind:52s} {len(values):5d}   (no assigned Hunter)")
            continue
        print(f"{kind:52s} {len(v):5d} {v.mean():7.2f} {np.median(v):7.2f} {(v > 0).mean():5.0%}")
    print(f"\nper-tick team shaping: mean {team.mean():+.4f}, std {team.std():.3f}; score reward total {score.sum():+.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--e1-samples", type=int, default=1500)
    parser.add_argument("--e2-ticks", type=int, default=60000)
    parser.add_argument("--e2-start", type=int, default=0)
    parser.add_argument("--only", choices=["e1", "e2"])
    parser.add_argument("--config", type=Path, help="RewardV2Config json (e.g. from fit_reward_v2.py --save)")
    args = parser.parse_args()
    cfg = RewardV2Config()
    if args.config:
        raw = json.loads(args.config.read_text())
        cfg = RewardV2Config(**{**raw, "class_value": tuple(raw["class_value"])})
    path = args.dataset_dir / "buffer_a.npz"
    if args.only in (None, "e1"):
        e1(path, args.e1_samples, cfg, np.random.default_rng(0))
    if args.only in (None, "e2"):
        e2(path, args.e2_ticks, cfg, args.e2_start)


if __name__ == "__main__":
    main()
