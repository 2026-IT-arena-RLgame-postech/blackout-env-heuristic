---
name: project-heuristic-numba-perf
description: "Numba-JIT speedup applied to blackout_env heuristics' Dijkstra distance map, and where the next bottleneck is"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a52de8d-09c8-4cf3-9dd4-e524bdbef845
  modified: 2026-09-15T14:30:54.183Z
---

Profiled the Python heuristic policies (`blackout-env/blackout_env/heuristics/`) with cProfile on a synthetic 24x24-map benchmark and found `StrategicHeuristicV2._distance_map` (`advanced.py`, an 8-neighbour Dijkstra over the grid) dominated wall-clock time — ~47% of total for `StrategicHeuristicV4`. It's inherited unchanged by V3/V4/V5/V7/V8/V10/V11/V12, so it's the single highest-leverage hot path across almost the whole "advanced" heuristic family.

Added a Numba-JIT reimplementation in `blackout_env/heuristics/_native.py` (manual array-backed binary heap, since `heapq` isn't supported in Numba nopython mode) and wired it into `advanced.py:_distance_map` behind a `NUMBA_AVAILABLE` check, falling back to the original pure-Python version when numba isn't installed. Verified bit-exact output against the pure-Python reference across 300 random grids before trusting it. Added `numba` as an optional `fast` extra in `pyproject.toml` (not a hard dependency) and ran `uv lock`.

**Why:** user asked to speed up heuristic algorithm execution ("네이티브로 컴파일 못하나"). Initial instinct was mypyc/Cython for the whole module (it's OOP-heavy: dataclasses, `OrderedDict` caches, `heapq`-of-tuples — a bad Numba fit in general), but profiling first showed one isolated, purely-numeric function was the actual bottleneck, which flipped the recommendation to a targeted Numba kernel instead of a broad recompile.

**How to apply:** first pass (`_distance_map` only) was ~2.2x on `StrategicHeuristicV4`/V10/V12/`HeuristicPolicyMixture` (2454→5386 ticks/s). Followed up by also JIT'ing `strategic.py:_astar` (the 8-neighbour A* used by `_navigate`/`_cached_astar`, shared by every heuristic, not just the V2+ family) — combined effect is ~2.7x over the original baseline (2454→6604 ticks/s for V4). The A* port needed a manual tie-break comparator matching Python's `heapq` tuple order on `(f, g, y, x)` exactly, or it would pick a different equal-cost path than the reference, which would've silently changed recorded behavior in BC/offline-RL datasets. Both kernels verified by direct comparison against the pure-Python reference (bit-exact for `_distance_map` over 300 random grids; exact-path-match for `_astar` over 383 reachable random grids) before trusting them — profile + verify-against-reference is the pattern to repeat for any further heuristic speedup work here.

Committed as two separate commits in `blackout-env` (`_distance_map` kernel, then `_astar` kernel) since the repo had unrelated concurrent WIP (other modified/untracked files from other work) that must not be swept into these commits — always check `git diff --stat` on staged files before committing here.

After both kernels, no single function dominates `_navigate` anymore (fragmented across numpy calls, stuck/escape-tracking loops, generator expressions) — diminishing returns from further Numba porting. If asked to speed heuristics up again, profile first rather than assuming there's another big win sitting there. Related: [[feedback_use_gui_for_eval_runs]] — this speedup should make GUI-visible eval matches feel snappier without changing behavior.

**2026-09-15 end-to-end profile of the real collection path** (`QMIXTrainer.collect_step` + heuristic mixture, 3000 steps, one Unity process): heuristic action selection is only ~10% of wall time; `env.step` is ~90%. Inside it: waiting on Unity, and ~20% spent in pure-Python protobuf decoding (mlagents_envs 1.1.0 pins `protobuf<3.21`, and PyPI has no arm64 C-extension wheel for 3.20.3, so the installed wheel is pure Python). Fixed by building protobuf 3.20.3's C++ extension from the official source tarball and installing that wheel into `.venv` (bit-identical obs/rewards vs the python backend). **`uv sync` silently reinstalls the pure-Python wheel** — check `api_implementation.Type()`; rebuilt wheel lives in `blackout-env/build/wheels/`, steps in `blackout-env/docs/perf_experiments_20260915.md`. Unity-side reward shaping is also a real cost: a first team-assigned nav-potential version (Dictionary per flood + `IsWalkable` per neighbor, every physics tick) tripled Unity CPU/step (1.34→3.26 ms) and cut collection 170→127 steps/s — a single-match wall-clock check had shown only ~7% and hid it. The allocation-free rewrite (cached walkability, reused arrays) got 0.83 ms/step, 190 steps/s; with the C++ protobuf too, 495 steps/s (from 170 that morning). **How to apply:** measure Unity CPU per step (`ps` cputime of the RLGame2026 child) on the actual collection loop when changing C# reward code, not just single-match wall time.

**2026-09-16 V5/V6 speedup (behaviour-identical):** V5 273→207 µs/tick (skip intercept work
when no enemy holds cargo; use the episode caches for distance map / components / protected
mask / A*), V6 255→199 (per-tick risk-field cache + `_native.risk_astar_numba`, V4 ≈175).
The risk kernel runs *without* fastmath and takes a table of CPython `math.hypot` values and
a float32 `0.15` threshold (NumPy 2 NEP 50 compares a float32 cell with a Python float in
float32) — both needed for bit-exact paths. **Verification method worth reusing:** stream the
first N records out of `datasets/heuristic_mixv3_20260915/buffer_{a,b}.npz` (uncompressed
zip, 30GB — read the .npy header via zipfile, don't np.load) to get real team-A/team-B
observation sequences, replay them through each `make_heuristic(id)` and compare SHA-256
of the actions before/after a change.

**2026-09-16 V17-V19 speedup (behaviour-identical, same hash method):** per-tick memo
(`_per_tick`, keyed on tick + graphic identity) for special/field-item/sanctuary masks, one
batched gather + `np.add.reduceat` for the storage scan, per-task precompute in
`_assign_economy`, cached item-free distance maps (invalidated by field-item bytes), cached
base mask / sentry cell. Live V17-V19 match replay: V17 339→270, V18 358→270, V19 395→285
µs/tick (V6 ≈275 under the same load). What remains is shared V1 `_navigate` + V6
`_risk_field`, left untouched to keep V1-V16 files unchanged. The mixv3 dataset trace has
few 3-4-Hunter teams, so also record live V17-V19 matches (scratch `record_live.py`) and
replay them against a `git archive HEAD` copy; the HEAD copy's first run includes numba JIT
compile time — time it on a second run.

**2026-09-16 shared V1 `_navigate` pass (branch `perf/navigate`, commits 50e16df, c6a633c):**
exact rewrites only — min/max instead of np.clip, float32 array diffs → tolist for hypot,
`np.sqrt(v.dot(v))` instead of `np.linalg.norm`, cached per-step distances for the 15-step
oscillation sum (newest step = the stuck-check distance), float32 scalar pixel coords.
Live trace: V1 188→136, V4 247→195, V6 269→219, V17 269→216, V19 268→223 µs/tick.
Rejected: batching V6 `_risk_field` exp over all enemies (bit-identical on this Mac but
float32 exp may differ by batch/alignment on x86 SIMD builds; only ~2-3%/tick).
Remaining V1 navigate cost is diffuse (~8 µs/unit of numpy scalar ops).
