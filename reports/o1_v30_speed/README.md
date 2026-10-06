# O1: V30 speed-up, bit-identical (2026-10-06, branch `feat/o1-v30-speed`)

The design write-up is in `docs/policy_v2_design_ko.md`, section "O1". All timings below were taken on the MacBook, single-threaded at `nice 10`, against the original code at 78e1f72. Other agents were loading the machine at the same time, so absolute µs vary by about ±5%, but each orig/new pair was run back-to-back.

## Result
| measure | orig | new | speed-up |
|---|---|---|---|
| V30 µs/decision, corpus replay (80 sequences, 162k decisions) | 539 | 297 | **1.81x** |
| same, process caches cleared before every game (cold) | 539 | 312 | 1.73x |
| V30 vs P20, V30 µs/decision (16 closed-loop games) | 856 | 502 | 1.71x |
| V30 vs P20, game wall s (P20 net ≈1.9 ms/decision dominates) | 3.03 | 2.61 | 1.16x |
| V30 vs V28, V30 µs/decision (16 games) | 490 | 302 | 1.62x |
| V30 vs V28, game wall s (both sides heuristic) | 3.75 | 2.25 | 1.66x |
| V25 vs V28, V25 µs/decision (8 games) | 939 | 486 | 1.93x |
| V27(1.5) vs V28, V27 µs/decision (8 games) | 888 | 440 | 2.02x |

See `games_summary.md` for every matchup. The per-opponent-family corpus numbers are in `equivalence_v30_corpus.txt`.

## Equivalence
- **V30 corpus** (`equivalence_v30_corpus.txt`): 72 sim games, 80 V30 sequences, 161,928 decisions, 0 divergences. Each decision hashes the actions together with V30's inspection state: `staged`, `_staged_xy`, `joining`, `trade_held`, `_held_xy`, `sentries`, `committed`, `last_step`, `waiting`, `campers`, `_race_now`, `sizes` and `anyangle_stats`. The 72 games are:
  - V30 vs P20: 16 games
  - V30 vs V28: 16 games
  - V30 vs V29, V27(1.5), V25, P17 and self-play: 8 games each
  - every matchup played from both sides
- **All heuristics** (`equivalence_all_heuristics.txt`): every registered heuristic (38), plus the V27(1.5), V28, V29 and V30 descriptors and `mixture`. Each was replayed on both sides of 14 corpus games (56 sequences, 65k decisions): 0 divergences for all 43.
- **Closed-loop games** (`games_orig.jsonl` / `games_new.jsonl`): 88 full matches, of which 72 have V30 on one side (vs P20, V28, V29, V27, V25, P17 and itself). All 88 have identical move hashes, final scores and lengths.
- **Tests**: `tests/test_o1_speed.py`.
  - Every fast path is checked against the reference implementation, which is kept in the code.
  - Golden move hashes recorded with 78e1f72 cover 4 matches of 700 decisions.
  - Cold and warm process caches play identically.

## Profile
- `phases_{orig,new}.txt`: wall time per phase of `teach`, on the 22,935-decision A/B subset.
- `profile_cprofile_{orig,new}.txt`: cProfile tottime top 40 on the same subset.

Before O1 the cost was spread out:
- labels 54% (7.3 `_label` calls per decision, from relabelling across the V20–V22 layers)
- composition 19%
- context 14%
- economy 8%

The largest single native items were the new-mask any-angle fields during the race and the `RacePlanner` queries.

## Scripts (`scripts/`; paths point into the session scratchpad)
| script | what it does |
|---|---|
| `record.py` | Records the corpus (raw_state/raw_map per decision). |
| `replay.py`, `replay_any.py`, `cmp.py` | Per-decision hash and timing replay, and the comparison. |
| `allpol.sh` | Runs the all-heuristic sweep. |
| `bench_games.py` | Closed-loop games for timing and margins. |
| `phases*.py` | Phase timers. |
| `sample.py` | Signal-based sampling profiler. |
| `golden.py` | Golden hashes. |

The corpus itself (13 MB) is not committed.
