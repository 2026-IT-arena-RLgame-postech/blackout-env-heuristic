# PPO iteration speed-up (perf/ppo)

Measured on poscat_gpu2 GPU 2 (2080 Ti), PPO r1 config (`games=256 segment=128`, the 9-opponent r3b pool, `pfsp=hard`,
32.8k views/iter, `critic_warmup=0` so the policy update is included), steady-state iterations (1-3). The server was loaded
(load average ~23 on 28 cores: the 10 h GRPO job on GPU 0), so the absolute CPU-bound times (rollout) are ~1.2x slower than
in the r1 production log; ratios are what matters. Tools: `examples/ppo_profile.py` (per-phase/per-policy timers, `--hash`, `--det`),
`scripts/ppo_update_bench.py` (update on a saved batch), `scripts/ppo_fp16_check.py` (fp16 gradient/value agreement).

## Baseline profile (2521ff6), 57.4 s/iter
| phase | s | note |
|---|---|---|
| trunk rollout | 24.0 | policy 22.0 of it: learner 4.3 (**eager, no CUDA graph**: PPO's `collect` never passed `graphs=`), 7 net opponents 0.8 each = 6, V32.1 6.6 + mixture 4.8 (worker processes, but called synchronously per team, nothing overlapped); render 1.3, sim step 0.9 |
| leaf | 2.2 | all of it the fp32 critic forward over 32.8k views + bootstrap; the python GAE loop is 0.08 s (vectorising it is not worth it) |
| update | 31.2 | critic epochs 13.4 + policy (old/ref pass + 2 epochs) 17.8; fp32 GEMM / conv bound, then elementwise (copies, mul) |
| total | 57.4 | (r1 production log, less loaded: 16.8 / 2.2 / 29.7 = 49 s) |

Which GRPO perf work PPO already had: heuristic workers (`heur_workers=-1` -> 4) and CUDA graphs for the *net opponents*; it did
**not** get graphs for the learner, nor the start/finish overlap of heuristic workers with the GPU forwards (that only existed in
`JointSeats`, i.e. `joint=True`, which is not bit-identical), nor fp16 updates.

## Changes
| # | change | default? | effect | equivalence |
|---|---|---|---|---|
| 1 | learner CUDA graphs in PPO's trunk rollout (`LearnerPolicy(graphs=self.learner_gf)`, same `graphs` flag as GRPO) | default | learner 4.3 -> 2.1 s | bitwise (graph == eager fp16 forward) |
| 2 | `overlap_heur` (PPOConfig default True; GRPOConfig default False): `JointSeats(split=True)` = what the two per-team `Seats` calls compute, but both teams' heuristic opponents are `start`ed on the workers before any net forward and `finish`ed after; every learner / net forward keeps its per-team batch composition, RNG stream and record order | default (PPO) | trunk 24.0 -> ~12.7 s | bitwise: batch hash, policy and critic parameter hashes identical to baseline over 4 full iterations (collect + update, `--det`, pool without d6_v42); unit test `test_ppo_overlap_heur_is_bit_identical_cpu` (in-process and worker heuristics) |
| 3 | `critic_fp16` (PPO, **opt-in**): autocast + GradScaler for critic epochs and the leaf/bootstrap value reads | opt-in | leaf 2.2 -> 1.4, critic epochs ~13.4 -> ~6 (estimated from the combined numbers) | not bitwise: critic gradient cos 0.99999 / rel. err 0.5 % vs fp32, value mean abs diff 8e-4 (std of V 0.38) |
| 4 | `update_fp16` (existing, opt-in) now with loss scaling (`GradScaler(init_scale=1024, no growth)`; applies to GRPO too, still off by default) | opt-in | policy update 17.8 -> ~10 | without scaling ~10 % of fp16 policy gradients underflowed (rel. err 1.8 %, cos 0.99984); with scaling rel. err 0.07 %, cos 1.00000; logp mean abs diff 0.004, argmax agreement 1.0 |
| 5 | `update_compile` (opt-in, GRPO+PPO): torch.compile (static shape, full-size minibatches only, grad-mode forwards only; autocast entered inside the compiled fn) | opt-in | fp32: update 31.4 -> 24.3; with fp16: 19 -> 10.6 | fp32 compile: logits within 2e-5 of eager, losses equal to ~4 digits. **One-off compile ~60-90 s** at the first update. Inductor on torch 2.11 gives wrong/NaN fp16 outputs for no-grad forwards and crashes with `dynamic=True`, hence the restrictions (old/ref log-probs stay eager) |

Default path bit-identity: `scripts/ppo_update_bench.py --det --n 8192` (3 consecutive updates, loss/kl/param hashes) is identical
between 2521ff6 and perf/ppo; `ppo_profile.py --det --hash` (8-opponent pool without the non-deterministic d6_v42, 4 iterations
with updates): batch hash, policy hash and critic hash identical at every iteration. Without `--det` the update is not reproducible
run-to-run (atomics), as before. Tests: `tests/test_grpo.py` + `tests/test_sim_batch.py` 24 passed + the new test.

## Result (s/iter, same machine and conditions)
| config | trunk | leaf | update | total | vs baseline |
|---|---|---|---|---|---|
| baseline 2521ff6 | 24.0 | 2.2 | 31.2 | 57.4 | 1.0x |
| new defaults (bit-identical) | 12.7 | 2.2 | 31.4 | 46.4 | 1.24x |
| + `update_compile=true` (fp32; update measured on a saved batch, total estimated) | 12.7 | 2.2 | ~24.3 | ~39 | 1.5x |
| + `update_fp16 critic_fp16` (update measured on a saved batch, total estimated) | 12.7 | 1.4 | ~19 | ~33 | 1.7x |
| + all three (recommended) | 13.0 | 1.4 | 10.7 | 25.1 | 2.3x |

Not worth it / rejected (measured): `minibatch=2048` with fp16+compile 10.2 vs 10.6 s (GPU compute bound; and it would change the
algorithm), fused critic+policy passes (two different networks, nothing to share; H2D happens once per update already),
vectorised GAE (0.08 s), pinned memory (transfers are a single up-front copy), overlapping the critic update with the next rollout
(possible in principle since the rollout does not use the critic, but GIL/GPU contention with the latency-bound rollout makes the gain
<= ~4 s of 25; not implemented), `joint=true` (not bitwise; the overlap now captures most of its benefit for PPO).

## Recommended config for the next PPO run
`--set update_fp16=true critic_fp16=true update_compile=true` on top of the r1 command (new defaults already include graphs +
overlap). Expected ~25 s/iter vs ~57 s here (49 s in the r1 log) -> r1's 1047 iterations: ~16.9 h -> ~7.5 h (2.0-2.3x).
Learning should not change beyond noise: the fp16 gradients agree with fp32 to cos >= 0.99999 and the compiled fp32 path to float
rounding, but no training run was done with the opt-in flags. For an exact continuation of r1 use the defaults only (1.24x, bitwise).
Also note the first iteration pays numba/graph warm-up (+~20-40 s) and `update_compile` its ~60-90 s one-off compile.
