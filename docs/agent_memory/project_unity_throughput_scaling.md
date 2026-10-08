---
name: project-unity-throughput-scaling
description: "Unity/ML-Agents throughput findings — process-parallel scaling ceiling, and why multi-arena batching gave far less real gain than a naive metric suggested"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a52de8d-09c8-4cf3-9dd4-e524bdbef845
  modified: 2026-09-13T08:45:55.894Z
---

Investigated how to maximize BlackOut's ML-Agents data-collection throughput on this Mac (M5 Pro, 18 cores: 6 "Super" + 12 "Performance", 48GB RAM), in `blackout-env`'s `collect_heuristic_dataset*.py` heuristic-vs-heuristic rollout collection.

**time_scale is irrelevant above ~20.** Raising Unity's `Time.timeScale` from 20 to 1000 changes nothing (measured both isolated and at the 18-process contention peak) — confirmed the real bottleneck is the fixed per-`env.step()` ML-Agents communicator round-trip (~6ms), not simulation speed. Payload size is already minimized on the Unity side (bit-packed single-channel semantic map texture, tiny shared vector obs) so that's not it either.

**Process-parallel scaling (single-arena, `build/mac/BlackOut.app`)**: aggregate steps/s peaks at **workers=18** (matches physical core count exactly) — 1592.8 aggregate steps/s, ~10x a single worker's 160.3. Past 18, oversubscription hurts (24→1548, 32→1448). No knee specifically at 6 (the Super-core count) — just a smooth curve peaking at the logical core count. This is the current recommended production config: `collect_heuristic_dataset_parallel.py --workers 18` at default `--time-scale 20`.

**Multi-arena batching (built, then set aside — see [[feedback_metric_choice_matters]])**: built a full pipeline — `blackout` repo's `ArenaDuplicator.cs` (headlessly clones `Prototype.unity` into `PrototypeMultiArena.unity` with N side-by-side arenas), an `arenaIndex` wire-protocol addition to `BlackOutAgent.cs`/`MapObsAgent.cs` (backward-compatible, default -1 = single-arena unaffected), and `blackout-env`'s `MultiArenaBlackOutEnv` + `collect_heuristic_dataset_multiarena.py` to actually consume it. All committed and functional (`blackout@e7ed3ab/8127508`, `blackout-env@d07027a`).

**Why it was set aside**: initial benchmarking used "agent-decisions/s" (every unit counted) as the throughput metric, which showed a dramatic ~4x win (single-arena 1650 dec/s → 4-arena 5648 dec/s per process). But that metric inflates trivially with arena count regardless of whether round-trip overhead was actually amortized. The metric that actually matters for data collection — transitions/s reaching the replay buffer — barely moved: single-arena `collect_heuristic_dataset.py` 164.8 steps/s vs 4-arena `collect_heuristic_dataset_multiarena.py` 176.9 steps/s, only **~1.07x**. Root cause: Unity's own per-arena `FixedUpdate` game-logic/physics cost (`EpisodeUpdate`/`Render`/reward-shaping) scales with arena count too, not just the gRPC round-trip, so raw tick rate dropped ~3.2x (407.6→128.4 ticks/s) when going 1→4 arenas in one process — almost fully cancelling the round-trip-amortization benefit.

**Current recommendation**: single-arena build × 18 parallel processes, `--time-scale 20`. The multi-arena code is kept working (not deleted) in case a future profiling pass into Unity's per-arena `FixedUpdate` cost (not yet done) changes this tradeoff — user explicitly said to conclude here provisionally ("잠정 종료"), not to abandon the approach permanently.

**How to apply**: if asked to optimize this pipeline's throughput again, don't reach for "agent-decisions/s" or any per-unit-scaled metric as the success criterion — measure actual transitions/s through the real collection script end-to-end. See [[project_heuristic_numba_perf]] for the separate (and genuinely effective) Python-side heuristic speedup work.
