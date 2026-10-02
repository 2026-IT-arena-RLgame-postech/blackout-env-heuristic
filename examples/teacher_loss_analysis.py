"""Where does a side lose?  Event tallies from distill_heldout_eval.py shards, both teams.

Every shard row is one learner tick in the canonical frame (own units rows 0-4, enemies 5-9),
so both teams' events can be read from it.  Per match and side:

  score@30/60/120/end   own score (x100) at that many seconds of play
  deliv, deliv_pts      cargo -> empty while alive and still a Collector/Carrier (deposits)
  lost_cargo            battery points destroyed when a holder died
  stolen                own-score drops between ticks without an absorption (raids on our storage)
  deaths/<cls>          units that respawned (position jump), by class at death; killer class
                        = nearest enemy at the previous tick
  near_base             deaths within 4 tiles of the own base
  kills_by_H            enemy deaths whose nearest own unit was a Hunter
  H(t)                  mean own Hunters in 0-30/30-60/60-120 s
  camped                share of ticks an enemy Hunter is within 3 tiles of the own base
  idle                  share of own non-Hunter unit-ticks that did not move (> 0.02 tiles)

  python examples/teacher_loss_analysis.py reports/teacher_v2/base_students [--by opponent]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.train.distill.dataset import decode_graphic

H = W = 24
CLASS_COLS = slice(9, 12)
CARGO_COL = 4
SPECIAL_COLS = slice(5, 9)
STORAGE_OWN, STORAGE_ENEMY, SPAWN_OWN, SPAWN_ENEMY = 6, 7, 4, 5
JUMP_TILES = 1.8  # RESPAWN_JUMP 0.15 normalised x 12
NAMES = ("C", "H", "Ca")


def base_cells(spawn: tuple[int, int]) -> np.ndarray:
    sy, sx = spawn
    rows = range(sy - 3, sy + 1) if sy > H // 2 else range(sy, sy + 4)
    cols = range(sx - 3, sx + 1) if sx > W // 2 else range(sx, sx + 4)
    return np.array([(y, x) for y in rows for x in cols], dtype=np.float64)


def side_events(pos, cls, cargo, special, score, absorb, t, mine, theirs, own_base) -> dict:
    """Events of team ``mine`` (row indices) against ``theirs``."""
    out = defaultdict(float)
    T = len(t)
    steps = np.linalg.norm(pos[1:, mine] - pos[:-1, mine], axis=-1)  # [T-1, 5]
    died = steps > JUMP_TILES
    for k in range(T - 1):
        for j, r in enumerate(mine):
            if died[k, j]:
                c = cls[k, r]
                out[f"deaths/{NAMES[c]}"] += 1
                gap = np.linalg.norm(pos[k, theirs] - pos[k, r], axis=-1)
                killer = cls[k, theirs[int(np.argmin(gap))]]
                out[f"killed_by/{NAMES[killer]}"] += 1
                out[f"deaths/{NAMES[c]}_by_{NAMES[killer]}"] += 1
                if np.sqrt(((own_base - pos[k, r]) ** 2).sum(-1)).min() <= 4.0:
                    out["near_base"] += 1
                if cargo[k, r] > 0:
                    out["lost_cargo"] += cargo[k, r]
                if special[k, r]:
                    out["lost_special"] += 1
                if t[k] < 60:
                    out["deaths<60s"] += 1
            elif cls[k, r] != 1 and cls[k + 1, r] == cls[k, r]:
                if cargo[k, r] > 0 and cargo[k + 1, r] == 0:
                    out["deliv"] += 1
                    out["deliv_pts"] += cargo[k, r]
                if special[k, r] and not special[k + 1, r]:
                    out["deliv_special"] += 1
                if cargo[k, r] == 0 and cargo[k + 1, r] > 0:
                    out["pickups"] += 1
    # own score drops without absorption = stolen from our storages
    ds = np.diff(score)
    fired = np.diff(absorb) > 1e-6
    out["stolen"] = float(-ds[(ds < 0) & ~fired].sum())
    # composition
    for lo, hi in ((0, 30), (30, 60), (60, 120)):
        sel = (t >= lo) & (t < hi)
        if sel.any():
            out[f"H{lo}_{hi}"] = float((cls[sel][:, mine] == 1).sum(1).mean())
            out[f"Ca{lo}_{hi}"] = float((cls[sel][:, mine] == 2).sum(1).mean())
    # camping of our base exit by their Hunters
    enemy_h = cls[:, theirs] == 1
    d_base = np.sqrt(((pos[:, theirs, None, :] - own_base[None, None]) ** 2).sum(-1)).min(-1)
    out["camped"] = float(((d_base <= 3.0) & enemy_h & (d_base > 0.01)).any(1).mean())
    # idle economy units
    econ = cls[:-1, mine] != 1
    still = (steps < 0.02) & ~died
    out["idle"] = float((still & econ).sum() / max(1, econ.sum()))
    for at in (30, 60, 120):
        k = np.searchsorted(t, at)
        out[f"score@{at}"] = float(score[min(k, T - 1)])
    out["score_end"] = float(score[-1])
    out["seconds"] = float(t[-1])
    return out


def kills(out_mine: dict, out_theirs: dict) -> None:
    out_mine["kills"] = sum(v for k, v in out_theirs.items() if k.startswith("deaths/") and "_by_" not in k)
    out_mine["kills_by_H"] = out_theirs.get("killed_by/H", 0.0)


def analyse(path: Path) -> tuple[dict, dict, dict]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    s = z["agent_states"].astype(np.float64)
    ts = z["team_state"].astype(np.float64)
    g0 = decode_graphic(z["graphic"][0])
    t = (1.0 - ts[:, 2]) * 420.0
    pos = np.stack([(1.0 - s[..., 1]) * 0.5 * H - 0.5, (s[..., 0] + 1.0) * 0.5 * W - 0.5], -1)
    cls = s[..., CLASS_COLS].argmax(-1)
    cargo = np.rint(np.maximum(s[..., CARGO_COL], 0) * 15.0)
    special = (s[..., SPECIAL_COLS] > 0.5).any(-1)
    own_spawn = tuple(int(v) for v in np.argwhere(g0[..., SPAWN_OWN] > 0.5)[0])
    enemy_spawn = tuple(int(v) for v in np.argwhere(g0[..., SPAWN_ENEMY] > 0.5)[0])
    mine, theirs = np.arange(5), np.arange(5, 10)
    a = side_events(pos, cls, cargo, special, ts[:, 0] * 100, ts[:, 3], t, mine, theirs, base_cells(own_spawn))
    b = side_events(pos, cls, cargo, special, ts[:, 1] * 100, ts[:, 3], t, theirs, mine, base_cells(enemy_spawn))
    kills(a, b)
    kills(b, a)
    return meta, a, b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--keys", nargs="*", default=None)
    p.add_argument("--per-match", action="store_true")
    args = p.parse_args()
    groups = defaultdict(lambda: ([], []))
    for d in args.dirs:
        for f in sorted((d / "shards").glob("*.npz")):
            meta, a, b = analyse(f)
            key = (meta.get("student") or meta["teacher"], meta["opponent"])
            groups[key][0].append(a)
            groups[key][1].append(b)
            if args.per_match:
                print(f"{f.name:80s} margin {meta['learner_score'] - meta['opponent_score']:+6.1f} "
                      f"deaths {sum(v for k, v in a.items() if k.startswith('deaths/') and '_by_' not in k):3.0f}/"
                      f"{sum(v for k, v in b.items() if k.startswith('deaths/') and '_by_' not in k):3.0f} "
                      f"s30 {a['score@30']:.0f}/{b['score@30']:.0f} s60 {a['score@60']:.0f}/{b['score@60']:.0f} "
                      f"end {a['score_end']:.0f}/{b['score_end']:.0f} stolen {a['stolen']:.0f}/{b['stolen']:.0f} "
                      f"lost {a.get('lost_cargo', 0):.0f}/{b.get('lost_cargo', 0):.0f}")
    for (cand, opp), (mine, theirs) in sorted(groups.items()):
        keys = args.keys or sorted(set().union(*mine, *theirs))
        print(f"\n== {cand} vs {opp}  ({len(mine)} matches)   own | opponent")
        for k in keys:
            va = np.array([m.get(k, 0.0) for m in mine])
            vb = np.array([m.get(k, 0.0) for m in theirs])
            print(f"  {k:24s} {va.mean():8.2f} | {vb.mean():8.2f}")


if __name__ == "__main__":
    main()
