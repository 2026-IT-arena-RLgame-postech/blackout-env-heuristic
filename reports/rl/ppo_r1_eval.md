# PPO r1 vs GRPO r3b / r3c: 256-seed evaluation (2026-10-22)

Run: `checkpoints/rl/ppo_r1` (`reports/rl/ppo_r1.sh`, `blackout_env/rl/ppo.py`): init p24c_v321, same opponent pool / weights / PFSP-hard as r3b,
256 on-policy games x 128-decision segments, separate critic (copy of the S2 value net, GAE lambda 0.95, Huber), stopped at the r3b simulator
budget. Eval exactly as `grpo_r3bc_eval.md`: `examples/eval_grpo_ckpt.py`, greedy, both sides, seeds 970000..970255 (512 games/cell, the same
games as every other candidate), poscat_gpu2 GPU 0/2 (`reports/rl/ppo_r1_eval.sh`). Paired diff = per-game difference on the same seed/side.

## Budget

| Run | Iters | Sim decisions | Learner views (trained on) | Wall h |
|---|---|---|---|---|
| GRPO r3b | 210 | 33.33M | 11.85M | 10.4 |
| GRPO r3c (value retrain) | 201 | 31.86M | 11.33M | 10.4 |
| PPO r1 | 1047 | 33.35M | 33.35M (every decision) | 16.9 |

Equal simulator decisions; PPO trains on ~2.8x more learner views (GRPO only trains on the 75-decision sample window of each branch, the
greedy leaf rollouts and trunk are not trained on). Wall-clock is not comparable as-is (PPO: 256-game batched rollouts + critic updates,
different code path; r3b/r3c ran before the perf merge).

## Results (win / margin +- SE; paired diffs margin +- SE ; win +- SE)

| Opponent | PPO win | PPO margin | PPO - p24c | PPO - r3b | PPO - r3c |
|---|---|---|---|---|---|
| p24c_v321 (pool) | 0.818 | +34.3 +- 1.7 | +33.6 +- 2.9 ; +0.296 +- 0.027 | +7.3 +- 2.4 ; +0.104 +- 0.024 | +3.8 +- 2.5 ; +0.060 +- 0.024 |
| p22_v31 (pool) | 0.864 | +42.1 +- 1.7 | +20.8 +- 2.5 ; +0.214 +- 0.024 | +9.2 +- 2.2 ; +0.107 +- 0.022 | +12.0 +- 2.4 ; +0.146 +- 0.024 |
| V32.1 (pool) | 0.857 | +40.7 +- 1.7 | +15.4 +- 2.7 ; +0.162 +- 0.024 | -2.4 +- 2.2 ; +0.010 +- 0.020 | -3.3 +- 2.2 ; +0.004 +- 0.019 |
| p5a_v22 (held out) | 0.992 | +66.8 +- 0.8 | +3.3 +- 1.0 ; +0.010 +- 0.007 | +2.5 +- 1.0 ; +0.008 +- 0.007 | +2.6 +- 1.1 ; +0.016 +- 0.007 |
| p16_v27 (held out) | 0.993 | +72.1 +- 0.7 | +3.1 +- 1.0 ; +0.015 +- 0.007 | +2.3 +- 1.0 ; +0.013 +- 0.007 | +2.2 +- 1.0 ; +0.003 +- 0.006 |

## Summary

- At equal simulator decisions PPO r1 is the strongest checkpoint so far: vs r3b +7.3 (3 sigma) vs p24c and +9.2 (4 sigma) vs p22, win +0.10;
  equal vs V32.1 (-2.4 +- 2.2, win +0.01). vs r3c: +3.8 (1.5 sigma) vs p24c, +12.0 vs p22.
- Held-out (saturated) opponents: PPO +3.1 / +3.3 over p24c (3 sigma), +2.3 to +2.6 over r3b/r3c (~2.4 sigma) -- the first measurable held-out
  gain of any RL run, though the probes remain near-saturated (99% win).
- Caveat on "sample efficiency": per simulator decision PPO wins, but per learner view (gradient data) GRPO used ~2.8x less, and PPO took
  ~1.6x the wall-clock here. The full curve comparison (all checkpoints vs decisions / views / wall-clock, plus value-free GRPO r3d) is the
  next study (`rl_sample_efficiency.md`).

Raw per-game margins: `reports/rl/grpo_r3bc_eval/ppo_final__<opp>__<lo>.json`; driver log `ppo_driver.log`.
