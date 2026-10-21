# GRPO r3b / r3c: 256-seed evaluation (2026-10-21)

Runs: `checkpoints/rl/grpo_r3b` (value frozen) and `grpo_r3c` (value retrained), both init p24c_v321. Method as in `grpo_r3a_eval.md`
(`examples/eval_grpo_ckpt.py`, greedy, both sides, horizon 11000), seeds 970000..970255 (256 seeds x 2 sides = 512 games/cell, same for every
candidate; run in 64-seed chunks), poscat_gpu2 GPU 2 only. Win counts draws as 1/2. Baseline = p24c_v321.pt. Paired diff = per-game difference on same seed/side.
In-pool opponents: p24c, p22, V32.1. Held out of training: p5a_v22, p16_v27. Candidates r3b_iter90 / r3c_iter150 = mid-run peaks of the 16-seed eval.

## Results (win / margin +- SE; paired diff vs p24c baseline)

| Opp | Cand | n | Win | Margin ± SE | dWin vs p24c | dMargin vs p24c |
|---|---|---|---|---|---|---|
| p24c_v321 | p24c_v321 | 512 | 0.522 | +0.6 ± 2.7 | | |
| p24c_v321 | r3b_final | 512 | 0.715 | +26.9 ± 2.1 | +0.192 ± 0.026 | +26.3 ± 2.8 |
| p24c_v321 | r3c_final | 512 | 0.759 | +30.5 ± 2.0 | +0.236 ± 0.026 | +29.9 ± 2.8 |
| p24c_v321 | r3b_iter90 | 512 | 0.723 | +25.1 ± 2.2 | +0.200 ± 0.025 | +24.5 ± 2.7 |
| p24c_v321 | r3c_iter150 | 512 | 0.752 | +28.8 ± 2.1 | +0.229 ± 0.025 | +28.2 ± 2.7 |
| p22_v31 | p24c_v321 | 512 | 0.650 | +21.3 ± 2.5 | | |
| p22_v31 | r3b_final | 512 | 0.757 | +32.8 ± 2.2 | +0.106 ± 0.023 | +11.5 ± 2.3 |
| p22_v31 | r3c_final | 512 | 0.718 | +30.1 ± 2.2 | +0.067 ± 0.024 | +8.8 ± 2.4 |
| p22_v31 | r3b_iter90 | 512 | 0.736 | +31.2 ± 2.2 | +0.086 ± 0.022 | +9.9 ± 2.2 |
| p22_v31 | r3c_iter150 | 512 | 0.731 | +31.7 ± 2.2 | +0.081 ± 0.023 | +10.3 ± 2.3 |
| p5a_v22 | p24c_v321 | 512 | 0.982 | +63.5 ± 0.9 | | |
| p5a_v22 | r3b_final | 512 | 0.984 | +64.3 ± 0.9 | +0.002 ± 0.008 | +0.8 ± 1.1 |
| p5a_v22 | r3c_final | 512 | 0.977 | +64.2 ± 0.9 | -0.006 ± 0.008 | +0.7 ± 1.0 |
| p5a_v22 | r3b_iter90 | 512 | 0.987 | +65.0 ± 0.8 | +0.005 ± 0.008 | +1.5 ± 1.0 |
| p5a_v22 | r3c_iter150 | 512 | 0.986 | +64.3 ± 0.9 | +0.004 ± 0.008 | +0.8 ± 1.0 |
| p16_v27 | p24c_v321 | 512 | 0.979 | +69.0 ± 0.9 | | |
| p16_v27 | r3b_final | 512 | 0.980 | +69.7 ± 0.9 | +0.002 ± 0.009 | +0.8 ± 1.1 |
| p16_v27 | r3c_final | 512 | 0.990 | +69.9 ± 0.8 | +0.012 ± 0.007 | +0.9 ± 1.1 |
| p16_v27 | r3b_iter90 | 512 | 0.983 | +69.3 ± 0.8 | +0.005 ± 0.008 | +0.3 ± 1.0 |
| p16_v27 | r3c_iter150 | 512 | 0.988 | +69.8 ± 0.7 | +0.010 ± 0.008 | +0.8 ± 1.0 |
| v32_1 | p24c_v321 | 512 | 0.695 | +25.3 ± 2.4 | | |
| v32_1 | r3b_final | 512 | 0.848 | +43.2 ± 1.9 | +0.152 ± 0.021 | +17.8 ± 2.4 |
| v32_1 | r3c_final | 512 | 0.854 | +44.0 ± 1.9 | +0.158 ± 0.022 | +18.7 ± 2.4 |


(V32.1 not run for the two mid-run checkpoints.)

## Paired differences (margin +- SE ; win +- SE)

| Opponent | r3b - p24c | r3c - p24c | r3c - r3b |
|---|---|---|---|
| p24c_v321 (pool) | +26.3 +- 2.8 ; +0.192 +- 0.026 | +29.9 +- 2.8 ; +0.236 +- 0.026 | +3.6 +- 2.3 ; +0.044 +- 0.023 |
| p22_v31 (pool) | +11.5 +- 2.3 ; +0.106 +- 0.023 | +8.8 +- 2.4 ; +0.067 +- 0.024 | -2.7 +- 2.1 ; -0.039 +- 0.022 |
| V32.1 (pool) | +17.8 +- 2.4 ; +0.152 +- 0.021 | +18.7 +- 2.4 ; +0.158 +- 0.022 | +0.9 +- 2.0 ; +0.006 +- 0.018 |
| p5a_v22 (held out) | +0.8 +- 1.1 ; +0.002 +- 0.008 | +0.7 +- 1.0 ; -0.006 +- 0.008 | -0.2 +- 1.1 ; -0.008 +- 0.008 |
| p16_v27 (held out) | +0.8 +- 1.1 ; +0.002 +- 0.009 | +0.9 +- 1.1 ; +0.012 +- 0.007 | +0.2 +- 1.0 ; +0.010 +- 0.006 |

Mid-run vs final (margin vs p24c, paired): r3b iter90 +24.5 / +9.9 / +0.3 / +1.5 (p24c / p22 / p16 / p5a) vs final +26.3 / +11.5 / +0.8 / +0.8;
r3c iter150 +28.2 / +10.3 / +0.8 / +0.8 vs final +29.9 / +8.8 / +0.9 / +0.7. All within ~1 SE of the finals: the 16-seed "peaks" were noise, nothing is lost by taking final.

## Summary

- In-pool opponents: large, highly significant gains (p24c +26 to +30 margin, V32.1 +18, p22 +9 to +12; 4-11 sigma). Both r3b and r3c beat the p24c baseline head-to-head ~72-76% (baseline mirror 0.522 / +0.6 +- 2.7, consistent with 0.5 / 0).
- Held-out opponents (p5a, p16): +0.7 to +0.9 margin, win +-0.01, all within 1 SE. But these are saturated (baseline already 98% win, margin +64/+69), so they are insensitive probes; at least no regression, and no evidence of overfitting damage.
- r3c vs r3b: r3c better vs p24c (+3.6 +- 2.3, 1.6 sigma), worse vs p22 (-2.7 +- 2.1), equal vs V32.1 and held-out. Value retraining makes no clear difference; the two are statistically indistinguishable overall.
- vs r3a (seeds 950xxx, p24c-only pool, 332 iters): r3a final was +9.2 vs p24c and about 0 (-2.9) vs p22. r3b/r3c are far stronger: roughly 3x the gain vs p24c and a real +9 to +12 vs p22 (pool now includes p22 and V32.1, so p22 gain is in-pool exploitation, mirroring r3a's p24c gain; V32.1 +18 likewise in pool). Still no demonstrated transfer to held-out nets, but the held-out probes are saturated.

Raw per-game margins: `reports/rl/grpo_r3bc_eval/<cand>__<opp>__<lo>.json` (64-seed chunks; keys margin_side0/margin_side1).
