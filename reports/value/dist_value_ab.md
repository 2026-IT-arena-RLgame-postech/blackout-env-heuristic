# Distributional (HL-Gauss) final-margin head vs regression head -- offline A/B

Question: does a distributional final head predict better, and does E[tanh(lead/30)] differ materially from tanh(E[lead]/30)
(the GRPO leaf `R = winness + 0.2 tanh(lead/30)` / PPO critic read)?  **Verdict: no switch** (details below).

## Setup
- Code: branch `feat/dist-value` (`final_head="hlgauss"`: `ValueNet(n_bins=112)`, bins of 2.5 points over +-140 points of the *remaining* final change, sigma = 0.75 bin, CE loss; the scalar final column = pmf mean so old consumers are unchanged; `decode_final_dist` gives pmf, mean, sd, E[tanh]; default and old checkpoints unchanged; tests `tests/test_value_dist.py`).
- **Data caveat.** The production net `s2v_f3xd2/iter2` was fitted by fitted value iteration on simulator branch data whose final targets are bootstrapped means (not outcome samples), and that dataset is no longer on the Mac/WSL (nor can a distribution head be bootstrapped from scalar targets).  So both arms use the **S1 recipe on true Unity outcomes** (`examples/value_train.py`, S1c settings): shard row cache rebuilt from the shards available locally (`datasets/value_s1_r64`, 64 rows/match: 154,560 train rows / 2,400 matches; held-out 8,256 rows / 129 matches; unseen-policy 60k rows sampled from 1,344 matches; unseen-OOD 60k rows from 4,752 matches), P10 trunk warm start, trunk lr x0.3, lr 3e-4, 16k steps, batch 256, fp16 + compile, final checkpoint (no selection).  Only the final head/loss differs: `reg` = Huber on remaining final change with beta 10 (the production `final_beta`, ~MSE -> mean), `hlgauss` = CE.  `regb1` = S1 default beta 1 (near-L1), 1 seed.  3 seeds per arm.  Result (W/D/L) head and delta heads identical in both.
- Evaluation: `examples/value_dist_ab_eval.py` (raw JSON in `reports/value/dist_ab/`), single view (no mirror symmetrisation; shard data is always the learner's side, mostly the stronger one: ~79% wins).  Numbers are mean +- seed sd (n=3).  Held-out has only 129 matches, so its seed sd is noisy; unseen_policy (P5a/D9/P4a/P10 matches) and unseen_ood are never trained on.

## Prediction quality (points; mean+-sd over 3 seeds)
| | heldout reg | heldout hlg | unseen_policy reg | unseen_policy hlg | unseen_ood reg | unseen_ood hlg |
|---|---|---|---|---|---|---|
| final MAE of mean | 11.73+-0.09 | **11.11+-0.28** | 10.35+-0.07 | **9.94+-0.02** | 14.97+-0.08 | **14.67+-0.02** |
| final RMSE | **23.13+-0.21** | 23.48+-0.86 | **21.42+-0.17** | 21.66+-0.10 | **27.88+-0.12** | 28.41+-0.09 |
| final R2 | **0.763** | 0.755+-0.018 | **0.714** | 0.707 | **0.733** | 0.723 |
| result 3-class logloss | 0.678+-0.027 | 0.658+-0.032 | 0.607+-0.014 | **0.567+-0.010** | 0.787+-0.031 | 0.783+-0.006 |
| win Brier | 0.061 | 0.059 | 0.052 | 0.050 | 0.069 | 0.069 |
| win ECE | 0.059 | 0.056 | 0.051 | 0.048 | 0.067 | 0.066 |
| MSE of tanh(true/30): tanh(E) reading | 0.149 | 0.147 | **0.107** | 0.110 | **0.166** | 0.177 |
| MSE of tanh(true/30): E[tanh] reading | | 0.144 | | 0.107 | | 0.173 |
| CRPS of distribution | (Gaussian ref, sd=RMSE: 11.05) | **9.84+-0.29** | (9.84) | **8.82** | (13.61) | **13.37** |
| NLL (nats per point of margin) | (Gaussian ref: 4.56) | 5.64+-0.05 | (4.48) | 5.49 | (4.75) | 7.16 |

Reg beta=1 (near-L1, seed 0): heldout MAE 11.51, RMSE 22.97, result logloss 0.618; unseen_policy MAE 10.03, logloss 0.547 -- as good as the best of everything on result logloss, i.e. differences between arms are of the order of the seed spread (reg seed sd up to 0.03 in logloss).
Reading: the HL-Gauss head gives a **slightly lower MAE** (-0.3..-0.6 pt, consistent over 3 seeds on all splits; it is median-like) and better CRPS, but a **slightly worse RMSE/R2 and worse tanh-target MSE on OOD** (tails/outliers), a mean of the same quality otherwise.  The distribution is **over-confident out of sample**: median predicted sd ~2 pts (train loss collapses on 154k correlated rows), so NLL is worse than even a homoscedastic Gaussian (5.6 vs 4.6) and far worse on OOD (7.2) although CRPS is better (a global-sigma Gaussian is a weak reference early in the game).  Win calibration is not hurt (ECE 0.056 vs 0.059; the pmf-implied P(final>0) is worse than the result head: logloss 0.42 vs 0.54 heldout is *better* for the pmf on heldout -- both are over-confident vs truth, ECE ~0.05).

## Jensen gap  E[tanh(final/30)] - tanh(E[final]/30)  (HL-Gauss nets, per state)
| split | mean abs | median abs | p90 abs | p99 abs | max | signed mean | mean abs in R (x0.2) |
|---|---|---|---|---|---|---|---|
| heldout | 0.009+-0.001 | 0.0005 | 0.009 | 0.209+-0.009 | 0.59 | -0.002 | 0.0018 |
| unseen_policy | 0.007 | 0.0002 | 0.005 | 0.189 | 0.55 | -0.003 | 0.0014 |
| unseen_ood | 0.009 | 0.0003 | 0.011 | 0.219 | 0.58 | -0.002 | 0.0018 |

- By phase (heldout, seed 0, mean abs): 0-10 s 0.027 (p90 0.088); 10-20 s 0.005; 20-30 s 0.003; 30-45 s 0.002; >45 s <= 0.0025.  unseen_policy: 0-10 s 0.017, then 0.002-0.004.
- By predicted uncertainty: the lowest three sd quartiles (sd <= 2.9 pts) have mean abs gap <= 0.001; the top quartile 0.034 (heldout) / 0.027 (unseen_policy), p90 0.08-0.11.  By P(win): 0.2-0.8 (only 1.4% of rows) mean abs 0.115, p90 0.31, max 0.51; >0.8 0.006; <0.2 0.014.
- So the gap is ~0 almost everywhere and large (0.1-0.5 in tanh, i.e. 0.02-0.1 in R) on a **few percent of states: the first ~10 s (the opening race) and genuinely contested states (P(win) 0.2-0.8)**.
- Same-match similar states (|d tanh(E)|<0.1 pairs, a sibling-like proxy): mean |difference of the gap| 0.005-0.006 (x0.2 = 0.001); the ordering of two same-match states with |d tanh(E)|<0.05 flips under E[tanh] in 2.3-3.1% of pairs.
- Empirical check (states bucketed by tanh(E), realised tanh(true/30)): E[tanh] is a better tracker than tanh(E) -- weighted bin-MAE 0.068 vs 0.076 (heldout), 0.053 vs 0.060 (unseen_policy), 0.074 vs 0.083 (OOD), and tanh-target MSE 0.144 vs 0.147 / 0.107 vs 0.107 / 0.173 vs 0.177 -- but both readings are off by 0.05-0.08 per bin (both over-predict: e.g. the top bin predicts 0.96 vs realised 0.915), **~8x the Jensen gap itself**: net miscalibration, not Jensen, dominates.  Caveat: because the learned pmfs are over-confident, the measured gap is likely a *lower bound*.  For reference an analytic Gaussian predictive with sd 23 pts gives gaps of -0.12..-0.14 at mean 20-40 pts (sd 10: -0.02..-0.04), i.e. about 0.03 in R -- the order of the p99 values above, only reached on the contested states.

## Scale vs GRPO reward noise
Within-group reward std is ~0.03-0.13.  Mean |dR| from the Jensen correction is 0.0014-0.0018 (1-6% of the std; the *differences* of the gap between sibling leaves ~0.001), p99 0.04, max ~0.12 (rare opening / 50-50 states).  Median change is 0.0001.  Even being generous (gap lower bound, x2-3 for under-dispersion) it is a small, mostly smooth, nearly order-preserving shift (rank flips of near-ties in ~3% of close pairs).

## Verdict
- **GRPO leaf: do not switch.**  The distributional head predicts the mean no better (RMSE/R2 slightly worse, MAE slightly better), is over-confident, and E[tanh] vs tanh(E) differs by ~0.002 in R on average, well below the 0.03-0.13 group std.  The small effect concentrates in the first 10 s and in contested states -- the only place where it could matter (opening race leaves); if tested, do it as a cheap post-hoc option, not a retrain.
- **PPO critic: no switch** (same quantity, smaller still in value units; the critic's own error dominates).
- If one wants to correct Jensen cheaply: regress tanh(final/30) directly with an extra scalar head (no distribution needed); expected effect same order as above, so only worth it if opening-phase leaf ranking is a suspected problem.
- Not tested / caveats: single-view readings; shard outcomes of distilled-student matches rather than the GRPO policy states; no bootstrapped/simulator branch data (production recipe); 3 seeds, 129 held-out matches; HL-Gauss hyper-parameters (bins, sigma, loss weight vs other heads) untuned -- the CE term (~3 nats) is ~10x the Huber term in the shared gradient, which may explain the slightly worse RMSE.
