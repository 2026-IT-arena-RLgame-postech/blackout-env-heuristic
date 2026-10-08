---
name: project-distill-step-perf
description: Distillation learner speed on MPS (2026-09-30) — what speeds it up, the inductor embedding-grad bug, and the pgrep self-match gotcha in waiter scripts
metadata:
  type: project
---

Learner step speed on MPS, measured 2026-09-30 (M5 Pro, batch 256, commit 696e500). The default is 218 ms/step.
- Adding `--train-quantiles 8 --compile` plus `--amp bf16` gives 73 ms; with `--amp fp16` instead it gives 84 ms.
- Q head on own units only (`q_units=5`) is always on. It is exact.
- A 3000-step offline A/B showed the same held-out fit for all three settings.
- The user prefers fp16 over bf16 for precision. fp16 runs with a GradScaler; logits stay around |10| under z-loss, and no steps were skipped.
- The GPU is compute-bound, so a bigger batch doesn't raise rows/s.

Two gotchas.
- **Inductor on MPS (torch 2.14):** it multiplies the gradient of batch-broadcast embeddings by B, so only `COMPILED_MODULES` are compiled. It also can't compile a module that turns autocast off.
- **Waiter scripts:** `pgrep -f "<pattern>"` inside `zsh -c '...'` matches the waiter's own command line, so it waits forever. That is how the D4 held-out waiter hung.

**Why:** a faster learner at replay ratio 2 needs about 3× more collection, roughly 12–18 Unity workers (see [[project-unity-throughput-scaling]]).

**How to apply:** use the fast flags on the next distill run and raise `--workers`. When waiting on a process, match a pattern that the waiter itself cannot contain.

Worker tick on an idle machine, D4 student vs V17: 3.8 ms, i.e. 10.6× real time. This is after three changes.
- Traced own-rows student inference.
- A Unity build that broadcasts the map only on decision ticks. The change is on branch `perf/mapobs-on-decision` in blackout (ca27b98). `build/mac` is now this build; the previous build is `build/mac_pre_mapobs_20260930`. Verified identical over 11,494 decisions.
- Faster monitors.

Gotcha: the reversal metric must keep `np.linalg.norm`. Exact 135° compass turns sit right on REVERSAL_COS, so a different norm's rounding changes the counts.
