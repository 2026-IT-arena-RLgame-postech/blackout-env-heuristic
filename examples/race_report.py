"""The opening race to the central Hunter sanctuary, per candidate and opponent, from saved shards.

The match is decided by who first gets a Hunter (docs/experiments_index_ko.md, P10 analysis):
the sanctuary is taken by single 0.04 s decision ticks, a tie is the normal outcome between two
precise players, and losing it by one tick costs ~40-90 points.  For every shard (one match, e.g.
``examples/distill_heldout_eval.py --save-shards``) this finds the first decision tick with a
Hunter on each side -- shard rows are in the learner's canonical frame, own units in rows 0-4,
the opponent's in 5-9; class one-hot ``agent_states[..., 9:12]``, Hunter = index 1 -- and calls
the race won (ours first), tie (same tick), lost (theirs first, or ours never) or none (neither
side ever has one).  Printed per (candidate, opponent): counts, mean margin per outcome, and the
mean tick gap ours - theirs where both raced.

  python examples/race_report.py reports/p11_heldout/shards
  python examples/race_report.py 'reports/p10_vs_v22/shards/*strategic_v22*.npz' --by-side
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from blackout_env.train.distill.dataset import N_TEAM, reconstruct_ticks, shard_meta

CLASS_COLS = slice(9, 12)
HUNTER = 1
OUTCOMES = ("won", "tie", "lost", "none")


def first_hunter_ticks(agent_states: np.ndarray, ticks: np.ndarray) -> tuple[int | None, int | None]:
    """(own, opponent) first decision tick with a Hunter, None = never in this shard."""
    classes = agent_states[..., CLASS_COLS]
    hunter = (classes.argmax(-1) == HUNTER) & (classes.max(-1) > 0.5)  # [T, 10]
    out = []
    for side in (slice(0, N_TEAM), slice(N_TEAM, 2 * N_TEAM)):
        rows = np.flatnonzero(hunter[:, side].any(axis=1))
        out.append(int(ticks[rows[0]]) if len(rows) else None)
    return out[0], out[1]


def race_outcome(own: int | None, opponent: int | None) -> str:
    if own is None and opponent is None:
        return "none"
    if opponent is None or (own is not None and own < opponent):
        return "won"
    if own == opponent:
        return "tie"
    return "lost"


def shard_race(path: Path) -> dict:
    with np.load(path) as shard:
        meta = shard_meta(shard)
        ticks = shard["tick"] if "tick" in shard else reconstruct_ticks(shard["team_state"])
        own, opponent = first_hunter_ticks(shard["agent_states"], ticks)
    student = meta.get("student")
    candidate = student if student else f"teacher:{meta.get('teacher', '?')}"
    return {
        "candidate": f"{'/'.join(Path(candidate).parts[-2:])}" + (f" [{meta.get('inference')}]" if meta.get("inference", "iqn") != "iqn" else "")
        if student else candidate,
        "opponent": meta.get("opponent", "?"), "side": "B" if meta.get("swap") else "A",
        "own": own, "opponent_tick": opponent, "outcome": race_outcome(own, opponent),
        "margin": float(meta.get("learner_score", np.nan)) - float(meta.get("opponent_score", np.nan)),
    }


def _paths(entries: list[str]) -> list[Path]:
    out = []
    for entry in entries:
        path = Path(entry)
        out += sorted(path.glob("*.npz")) if path.is_dir() else [Path(p) for p in sorted(glob.glob(entry))]
    return [p for p in out if not p.name.endswith(".tmp.npz")]


def report(races: list[dict], by_side: bool = False) -> str:
    groups = defaultdict(list)
    for race in races:
        groups[(race["candidate"], race["opponent"]) + ((race["side"],) if by_side else ())].append(race)
    head = f"{'candidate':34s} {'opponent':34s}" + (" side" if by_side else "")
    lines = [head + f" {'n':>3s} {'W/T/L/-':>11s} {'margin':>7s}" + "".join(f" {'m_' + o:>7s}" for o in OUTCOMES[:3])
             + f" {'gap':>6s}"]
    for key, group in sorted(groups.items()):
        counts = {o: sum(r["outcome"] == o for r in group) for o in OUTCOMES}
        margins = {o: [r["margin"] for r in group if r["outcome"] == o] for o in OUTCOMES}
        gaps = [r["own"] - r["opponent_tick"] for r in group if r["own"] is not None and r["opponent_tick"] is not None]
        name = f"{key[0]:34s} {key[1]:34s}" + (f" {key[2]:>4s}" if by_side else "")
        wtl = "/".join(str(counts[o]) for o in OUTCOMES)
        line = f"{name} {len(group):3d} {wtl:>11s} {np.mean([r['margin'] for r in group]):+7.1f}"
        line += "".join(f" {np.mean(margins[o]):+7.1f}" if margins[o] else f" {'-':>7s}" for o in OUTCOMES[:3])
        line += f" {np.mean(gaps):+6.1f}" if gaps else f" {'-':>6s}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("shards", nargs="+", help="shard directories or globs")
    p.add_argument("--by-side", action="store_true", help="split each row by the learner's side (A/B)")
    p.add_argument("--json", type=Path, default=None, help="also write every match's race as JSON lines here")
    args = p.parse_args()
    paths = _paths(args.shards)
    if not paths:
        raise SystemExit("no shards found")
    races = [shard_race(path) for path in paths]
    print(report(races, args.by_side))
    if args.json is not None:
        args.json.write_text("".join(json.dumps(r) + "\n" for r in races))


if __name__ == "__main__":
    main()
