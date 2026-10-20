# GRPO iteration speed-up (perf/grpo)

Measured on poscat_gpu2 GPU 1 (2080 Ti), production config (groups=48 k=16 sample_steps=75 drop_flat=0.005, 9-opponent pool,
PFSP hard), same machine/contention (two live runs in the background), NUMBA_NUM_THREADS=6, steady-state iterations (1-2).
Tool: `examples/grpo_profile.py` (per-phase/per-policy timers; `--hash`, `--det`, `--save-batch`).

## Baseline profile (7474080), ~120-128 s/iter
| part | s | note |
|---|---|---|
| trunk rollout | 40 | 12k decisions: policy 40 s, sim step 0.14, render 0.06 |
| branch rollout | 78-80 | 153k decisions: policy 78, sim step 0.8, render 0.25 |
| of which V32.1 + mixture (Python, main thread) | 42 | 35k decisions x 1.2 ms |
| of which learner forward+sampling | 15 | ~400 calls of ~380 rows |
| of which 7 net opponents | ~20 (branch) + ~35 (trunk) | ~7000 tiny calls, ~9 ms each |
| leaf value | 0.15 | |
| update | 5-8 (up to 22 early) | fp32 GEMM-bound, ~42k rows early |

Conclusion: sim stepping and rendering are negligible (<1 s). The iteration is launch/latency-bound (24x24 maps, GPU idle)
plus single-threaded Python heuristics.

## Changes
| # | change | s/iter (prod config) | correctness |
|---|---|---|---|
| 0 | baseline | 117-126 | |
| 1 | `sim/graphfwd.py`: CUDA-graph replay of the eager fp16-autocast forward per exact batch size (learner + net opponents; `graphs=True`, shared mem pool, 120 MB); LearnerPolicy: argmax/sampling on device, 1-2 D2H instead of ~5 | -> ~70 | graph == eager **bitwise** at every size tested (scripts/perf_numerics_check.py); collect() hashes identical to baseline (see below) |
| 2 | `rl/hpool.py`: V32.1/mixture instances in 4 spawn workers (game b -> worker b%n, same per-(game,team) seeds and call order), started before and collected after the GPU forwards; `BatchSim.raw_batch` numba kernel (`heur_workers`, -1 = 4 on CUDA) | 57-60 (1+2) | collect() hash identical (graphs+workers vs baseline) |
| 3 | `JointSeats` (`joint=True`, opt-in): one policy call per step for both teams -> one forward per policy | 32-35 (1+2+3) | **not bitwise**: fp16 logits change with batch composition (max abs dlogit ~0.1-0.2 between batch 40 and 128, argmax agreement 1.0 in the check; this noise already exists between any two batch layouts). RNG stream and record order are kept. R_mean in the same range as run-to-run noise. Optional. |
| 4 | `update_fp16=True` (opt-in): autocast in update() | update 22.7 -> 13.9 s (42k rows) | dlogp mean 0.011 / max 0.26, argmax agreement 0.9998, KL(fp32||fp16) 5.6e-5; optional |

Final (steady state, same conditions, back-to-back): baseline 117-126 s -> graphs+workers 57-60 s (2.1x, bit-identical)
-> + joint 32-35 s (~3.7x, opt-in). decisions/s 1.3k -> 2.6-2.9k -> 4.8k. The remaining time at joint: trunk 6, branch 18,
update 5-10 (GPU fp32 GEMM-bound).

## Correctness gate
Baseline GRPO collect() is not reproducible across runs when the pool contains `d6_v42` (that net has a non-deterministic kernel;
every other net, V32.1 and mixture are deterministic). On the pool without d6 (groups=12 k=4, scripts/perf_det_probe.py,
`detcheck`): iteration-0 batch hash is identical for baseline x2, graphs off, graphs on, graphs + 3 heuristic workers
(ce6cb867380ed7fc); joint differs (expected). update() is itself non-deterministic (atomics in backward) so iteration 1 differs
in every variant, baseline included.
Tests: tests/test_grpo.py + tests/test_sim_batch.py, 24 passed (CPU; includes the PPO mixed-pool test from 516d9a9).

## Usage
Defaults now: `graphs=True`, `heur_workers=-1` (4 workers on CUDA), `joint=False`, `update_fp16=False`.
Opt in: `--set joint=true update_fp16=true`; revert everything: `--set graphs=false heur_workers=0`.
Warm-up: the first iteration captures ~100-400 graphs (+~20-40 s once). Workers use ~3-4 cores in total during branches.
