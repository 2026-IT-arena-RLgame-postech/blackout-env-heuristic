# GRPO r3a: 256-seed evaluation (2026-10-20)

Run: `checkpoints/rl/grpo_r3a` (start p24c_v321, 332 iters, 10.2 h). Eval: `examples/eval_grpo_ckpt.py` (same
mechanics as `grpo.eval_matches`: greedy NetPolicy, full matches, both sides, horizon 11000), seeds 950000..950255
(256 seeds x 2 sides = 512 games per cell, identical for every candidate), run on poscat_gpu2 GPUs 0-2 (not 3), ~13 min for the net opponents.
Baseline = p24c_v321.pt itself. Win rate counts draws as 1/2. Paired diff = per-game difference vs baseline on the same seed/side.
Note: the baseline-vs-p24c "mirror" gives 0.493 / -0.8 +- 2.6 (consistent with 0.5 / 0; learner is loaded in default precision, opponent fp16 on CUDA).

## Results (win / margin +- SE; paired diff vs baseline)

| Opponent | Candidate | Win | Margin +- SE | dWin +- SE | dMargin +- SE |
|---|---|---|---|---|---|
| p24c_v321 | baseline (iter 0) | 0.493 | -0.8 +- 2.6 | | |
| p24c_v321 | iter_150 | 0.612 | +12.7 +- 2.5 | +0.119 +- 0.023 | +13.5 +- 2.4 |
| p24c_v321 | iter_210 | 0.566 | +7.9 +- 2.5 | +0.073 +- 0.022 | +8.7 +- 2.3 |
| p24c_v321 | final (332) | 0.570 | +8.4 +- 2.5 | +0.077 +- 0.022 | +9.2 +- 2.5 |
| p22_v31 | baseline | 0.674 | +23.7 +- 2.3 | | |
| p22_v31 | iter_150 | 0.670 | +23.0 +- 2.3 | -0.004 +- 0.023 | -0.7 +- 2.3 |
| p22_v31 | iter_210 | 0.672 | +23.9 +- 2.3 | -0.002 +- 0.022 | +0.2 +- 2.3 |
| p22_v31 | final | 0.655 | +20.8 +- 2.4 | -0.019 +- 0.024 | -2.9 +- 2.4 |
| strategic_v19 | baseline | 0.986 | +78.5 +- 0.8 | | |
| strategic_v19 | iter_210 | 0.992 | +79.0 +- 0.6 | +0.006 +- 0.006 | +0.5 +- 0.7 |
| strategic_v19 | final | 0.994 | +79.7 +- 0.6 | +0.008 +- 0.006 | +1.2 +- 0.8 |

(V19 was extended from the requested 64 to 256 seeds because it was cheap; baseline included for pairing. V19 is saturated here, so it is not a sensitive probe for these nets.)

## Verdict

- GRPO improved over P24c head-to-head: +8 to +13 margin and +7 to +12 pts win rate (paired SE ~2.4 / 0.022, i.e. 3.8-5.6 sigma). This is real.
- The mid-run "peak then decline" in the 16-seed log was mostly noise (SE ~10). On 256 seeds iter_150 is best (+13.5 paired), iter_210 and final are both about +9; the 150 -> final difference (~4 margin) is only ~1.3 sigma (unpaired SE ~3.5), so at most a mild drift, not a collapse. The final 16-seed reading (0.41 / -4.1) was a ~1.5 sigma low draw.
- The gain did not transfer: vs P22 there is no improvement (paired diffs within noise; final -2.9 +- 2.4 is slightly negative), vs V19 saturated. So the policy improved specifically against its training opponent P24c (self-play style exploitation) without broader strength gain. Treat as P24c-specific; iter_150 is the best checkpoint by this eval but all three are statistically close.

## Training-curve notes (log.jsonl, means per 30 iters)

| iters | kl | ent | R_group_std | R_dev_std | clipfrac | grad_norm | R_mean | win_mean | lead_mean |
|---|---|---|---|---|---|---|---|---|---|
| 1-30 | 0.0037 | 0.7495 | 0.0290 | 0.0943 | 0.0019 | 0.8881 | 0.4140 | 0.3428 | 24.59 |
| 31-60 | 0.0063 | 0.7755 | 0.0204 | 0.0789 | 0.0022 | 0.8680 | 0.3929 | 0.3273 | 21.00 |
| 61-90 | 0.0089 | 0.7655 | 0.0211 | 0.0840 | 0.0020 | 0.8912 | 0.3796 | 0.3115 | 22.92 |
| 91-120 | 0.0090 | 0.7889 | 0.0226 | 0.0891 | 0.0019 | 0.8693 | 0.3748 | 0.3119 | 20.10 |
| 121-150 | 0.0092 | 0.7916 | 0.0191 | 0.0767 | 0.0016 | 0.8361 | 0.5535 | 0.4723 | 24.01 |
| 151-180 | 0.0094 | 0.7715 | 0.0222 | 0.0833 | 0.0017 | 0.8919 | 0.4599 | 0.3846 | 23.84 |
| 181-210 | 0.0093 | 0.7848 | 0.0198 | 0.0787 | 0.0022 | 0.8541 | 0.3829 | 0.3169 | 20.89 |
| 211-240 | 0.0093 | 0.7836 | 0.0200 | 0.0810 | 0.0016 | 0.8637 | 0.4474 | 0.3757 | 22.49 |
| 241-270 | 0.0119 | 0.7731 | 0.0251 | 0.0895 | 0.0020 | 0.8826 | 0.4367 | 0.3724 | 20.90 |
| 271-300 | 0.0127 | 0.7775 | 0.0207 | 0.0823 | 0.0023 | 0.8073 | 0.4212 | 0.3591 | 19.84 |
| 301-330 | 0.0125 | 0.7799 | 0.0220 | 0.0875 | 0.0021 | 0.8952 | 0.3524 | 0.2986 | 16.93 |
| 331-332 | 0.0131 | 0.7741 | 0.0260 | 0.0936 | 0.0019 | 0.7943 | 0.2608 | 0.2195 | 13.86 |

- kl rises fast to ~0.009 by iter 60, plateaus to ~210, then creeps to ~0.013 (beta 0.02, lr 3e-6): slow drift away from the reference late in the run.
- ent flat (~0.77-0.79): no entropy collapse. clipfrac ~0.002 and grad_norm ~0.85: updates are tiny and stable, no instability.
- R_group_std small (~0.02, 0.019-0.029) and no trend: within-group reward signal is weak throughout, so learning is slow and noise-dominated.
- R_mean/win_mean/lead_mean in training are noisy per-window (opponent/start mix dependent; the 121-150 bump and 301-332 dip are sampling variation, not a clear trend); the late decline in training rollouts matches the slight 150 -> final drift but is within noise.

## Raw rows

Per-game margins (seed-ordered, side0 then side1 per chunk file): `reports/rl/grpo_r3a_eval/*.json`; summary lines: `reports/rl/grpo_r3a_eval/results.txt`.
