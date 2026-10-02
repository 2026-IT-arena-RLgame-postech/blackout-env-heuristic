"""S1 value-net data: an index of every distill shard, a by-match split, and a row cache.

The distill shards hold ~100M rows (one match per file, canonical team frame; dataset.py).  Rows
of one match are strongly correlated and the full set is ~750 GB of decoded maps, so the cache
keeps ``rows_per_match`` rows drawn uniformly from each match (seeded by the shard name), with
their targets (targets.py) computed from the *whole* shard, in one memory-mapped directory:

  graphic       uint8 [N, 24, 24, 13]  as stored (x15)
  team_state    f32   [N, 4]
  agent_states  f32   [N, 10, 12]
  delta         f32   [N, H]   lead change over HORIZONS_S
  delta_mask    bool  [N, H]
  past_end      bool  [N, H]   that horizon reaches past the match end (lead = final lead)
  final         f32   [N]      final lead (score / 100)
  result        int8  [N]      1 / 0 / -1
  elapsed_s     f32   [N]
  match         int32 [N]      index into matches.json
  split         int8  [N]      SPLITS index

Splits (by match, ``assign_split``): ``unseen`` = every match where P5a, D9 or P4a played
(learner side or opponent, by run directory or checkpoint name) or that comes from the OOD
suite (reports/distill_ood, or an opponent only that suite uses); of the rest, a hash-random
5% are ``heldout`` and the remainder ``train``.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blackout_env.train.distill.dataset import shard_meta
from blackout_env.value.targets import HORIZONS_S, match_targets, shard_ticks

SHARD_ROOTS = (
    "checkpoints/distill", "reports/distill_heldout", "reports/distill_ood", "reports/p4_heldout",
    "reports/p1_heldout", "reports/anticamp", "reports/teacher_v2", "reports/v19_analysis",
)
SPLITS = ("train", "heldout", "unseen")
HELDOUT_FRACTION = 0.05
UNSEEN_POLICIES = re.compile(r"(^|[/:_])(p5a|d9|p4a)([/_.]|$)")
OOD_DIR = "reports/distill_ood"
# opponents that only the OOD suite plays (examples/distill_heldout_eval.py --opponents ood);
# plain strategic_v21 is also a regular held-out opponent, so it does not mark a match OOD
OOD_ONLY_OPPONENTS = re.compile(
    r"20260919-010456_42178|^idle$|^random$|^self$|^teacher:v21-teach$|^strategic_v20$"
    r"|^strategic_v17:steal_weight=3\.0,exposure_cost=0\.3$|^strategic_v19:hunter_quota=4,hunter_mode=camp$"
)
META_KEYS = ("seed", "swap", "opponent", "beta", "teacher", "student", "tag", "rows", "learner_score",
             "opponent_score", "result", "elapsed_s", "stop_when_exhausted", "opponent_played")
ARRAYS = {  # name -> (dtype, per-row shape)
    "graphic": (np.uint8, (24, 24, 13)),
    "team_state": (np.float32, (4,)),
    "agent_states": (np.float32, (10, 12)),
    "delta": (np.float32, (len(HORIZONS_S),)),
    "delta_mask": (np.bool_, (len(HORIZONS_S),)),
    "past_end": (np.bool_, (len(HORIZONS_S),)),
    "final": (np.float32, ()),
    "result": (np.int8, ()),
    "elapsed_s": (np.float32, ()),
    "match": (np.int32, ()),
    "split": (np.int8, ()),
}


# ------------------------------------------------------------------------------ index

def shard_paths(repo: Path, roots=SHARD_ROOTS) -> list[str]:
    """Every complete shard under ``roots`` (relative to ``repo``), sorted."""
    out = []
    for root in roots:
        for dirpath, _, names in os.walk(repo / root):
            if "shards" not in Path(dirpath).name:
                continue
            out += [os.path.relpath(os.path.join(dirpath, n), repo) for n in names
                    if n.endswith(".npz") and not n.endswith(".tmp.npz")]
    return sorted(out)


def _read_meta(args) -> dict:
    repo, path = args
    with np.load(Path(repo) / path) as shard:
        meta = shard_meta(shard)
        has_tick = "tick" in shard.files
    return {"path": path, "has_tick": has_tick, **{k: meta.get(k) for k in META_KEYS}}


def build_index(repo: Path, workers: int = 8, roots=SHARD_ROOTS) -> list[dict]:
    paths = shard_paths(repo, roots)
    with ProcessPoolExecutor(workers) as pool:
        index = list(pool.map(_read_meta, [(str(repo), p) for p in paths], chunksize=64))
    for entry in index:
        entry["split"] = assign_split(entry)
    return index


def is_unseen(entry: dict) -> bool:
    if entry["path"].startswith(OOD_DIR):
        return True
    if OOD_ONLY_OPPONENTS.search(str(entry.get("opponent") or "")):
        return True
    fields = (entry["path"], entry.get("student"), entry.get("opponent"), entry.get("teacher"),
              entry.get("opponent_played"))
    return any(UNSEEN_POLICIES.search(str(f).lower()) for f in fields if f)


def assign_split(entry: dict) -> str:
    if is_unseen(entry):
        return "unseen"
    bucket = zlib.crc32(entry["path"].encode()) % 10_000
    return "heldout" if bucket < HELDOUT_FRACTION * 10_000 else "train"


def unseen_group(entry: dict) -> str:
    """Which unseen test a match belongs to: "ood" or "policy" (P5a/D9/P4a)."""
    if entry["path"].startswith(OOD_DIR) or OOD_ONLY_OPPONENTS.search(str(entry.get("opponent") or "")):
        return "ood"
    return "policy"


# ------------------------------------------------------------------------------ cache

def sample_rows(n: int, k: int, path: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def _match_rows(args):
    repo, entry, k = args
    with np.load(Path(repo) / entry["path"]) as shard:
        team_state = shard["team_state"]
        ticks = shard_ticks(shard)
        targets = match_targets(team_state, ticks, entry)
        idx = sample_rows(len(team_state), k, entry["path"])
        out = {
            "graphic": shard["graphic"][idx], "team_state": team_state[idx],
            "agent_states": shard["agent_states"][idx],
        }
    for name in ("delta", "delta_mask", "past_end", "final", "result", "elapsed_s"):
        out[name] = targets[name][idx]
    return out


def open_cache(directory: Path, mode: str = "r", n: int | None = None) -> dict[str, np.ndarray]:
    directory = Path(directory)
    if n is None:
        n = json.loads((directory / "cache.json").read_text())["rows"]
    return {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode=mode, dtype=dtype, shape=(n, *shape))
            if mode == "w+" else np.load(directory / f"{name}.npy", mmap_mode=mode)
            for name, (dtype, shape) in ARRAYS.items()}


def build_cache(repo: Path, index: list[dict], directory: Path, rows_per_match: int = 32, workers: int = 8,
                log=print) -> int:
    """Writes the row cache (module docstring) for every match in ``index``; returns its rows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = [e for e in index if (e.get("rows") or 0) > 0 and e.get("result") is not None]
    counts = np.array([min(e["rows"], rows_per_match) for e in entries])
    starts = np.concatenate(([0], np.cumsum(counts)))
    n = int(starts[-1])
    arrays = open_cache(directory, "w+", n)
    with ProcessPoolExecutor(workers) as pool:
        jobs = pool.map(_match_rows, [(str(repo), e, rows_per_match) for e in entries], chunksize=16)
        for i, (entry, rows) in enumerate(zip(entries, jobs)):
            a, b = starts[i], starts[i + 1]
            if len(rows["team_state"]) != b - a:
                raise RuntimeError(f"{entry['path']}: meta rows {entry['rows']} != shard rows")
            for name, values in rows.items():
                arrays[name][a:b] = values
            arrays["match"][a:b] = i
            arrays["split"][a:b] = SPLITS.index(entry["split"])
            if i % 5000 == 0:
                log(f"cache: {i}/{len(entries)} matches, {a} rows")
    for values in arrays.values():
        values.flush()
    (directory / "matches.json").write_text(json.dumps(entries))
    (directory / "cache.json").write_text(json.dumps({"rows": n, "rows_per_match": rows_per_match,
                                                       "horizons_s": list(HORIZONS_S), "matches": len(entries)}))
    return n
