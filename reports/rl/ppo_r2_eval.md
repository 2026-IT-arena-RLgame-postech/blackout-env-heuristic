# PPO r2 eval — regression critic vs HL-Gauss critic (256 seeds × 2 sides)

Setup: reports/rl/ppo_r2_setup.md. Both arms: r1 recipe + speed flags (update_fp16, critic_fp16, update_compile), critic_warmup 10,
1048 it, 33.35M simulator decisions (= r3b / r1 budget), ~10.5 h each on gpu2 (r1: 16.9 h).
`reg` = MSE/Huber scalar critic (control), `hlg` = 101-bin HL-Gauss CE critic ("Stop Regressing"), V = pmf mean.

Eval: `examples/eval_grpo_ckpt.py`, seeds 970000–970255, both sides (512 games per pair), final.pt vs frozen opponents.
Raw: `reports/rl/grpo_r3bc_eval/ppo_r2_{reg,hlg}__<opp>__<lo>.json`, driver log `wsl_r2_driver.log`, health log `wsl_health.log`.

**Hardware caveat:** r2 games ran on the WSL laptop RTX 3060; r1 / r3b / p24c baselines ran on gpu2 (2080 Ti). fp16 inference is not
bit-identical across the two, so pairing vs those baselines is weaker (same seeds, diverging trajectories). The SEs below are still valid
paired-diff SEs, just less variance-reduced. hlg − reg is fully paired (both on the 3060).

## Absolute (margin ± SE / win %)

| cand | p24c_v321 | p22_v31 | V32.1 | p5a_v22 (held-out) | p16_v27 (held-out) |
|---|---|---|---|---|---|
| ppo_r2_reg | +34.4±1.7 / 82.3 | +42.2±1.7 / 85.7 | **+48.1±1.6 / 89.5** | +67.5±0.8 / 99.0 | +72.5±0.6 / 99.2 |
| ppo_r2_hlg | +33.5±1.9 / 81.4 | +42.2±1.8 / 85.4 | +45.7±1.7 / 85.7 | +67.8±0.7 / 99.2 | +72.7±0.7 / 99.0 |
| ppo_final (r1) | +34.3±1.7 / 81.8 | +42.1±1.7 / 86.4 | +40.7±1.7 / 85.7 | +66.8±0.8 / 99.2 | +72.1±0.7 / 99.3 |
| r3b_final (GRPO) | +26.9±2.1 / 71.5 | +32.8±2.2 / 75.7 | +43.2±1.9 / 84.8 | +64.3±0.9 / 98.4 | +69.7±0.9 / 98.0 |
| p24c_v321 (start) | +0.6±2.7 / 52.2 | +21.3±2.5 / 65.0 | +25.3±2.4 / 69.5 | +63.5±0.9 / 98.2 | +69.0±0.9 / 97.9 |

## Paired diffs (margin ± SE ; win-rate pp ± SE)

| diff | p24c_v321 | p22_v31 | V32.1 | p5a_v22 | p16_v27 |
|---|---|---|---|---|---|
| **hlg − reg** | −0.9±2.0 ; −0.9±2.1 | −0.1±1.7 ; −0.4±1.8 | −2.4±2.0 ; −3.7±1.9 | +0.3±0.9 ; +0.2±0.6 | +0.2±0.8 ; −0.2±0.6 |
| reg − r1 | +0.1±2.0 ; +0.5±2.2 | +0.1±1.7 ; −0.7±1.8 | **+7.4±1.8 ; +3.7±1.7** | +0.7±0.9 ; −0.2±0.6 | +0.4±0.9 ; −0.1±0.5 |
| hlg − r1 | −0.8±2.1 ; −0.4±2.1 | +0.1±1.8 ; −1.1±1.8 | +5.0±2.0 ; +0.0±2.0 | +1.0±0.8 ; +0.0±0.5 | +0.7±0.8 ; −0.3±0.5 |
| reg − r3b | +7.5±2.5 ; +10.8±2.5 | +9.4±2.2 ; +10.1±2.2 | +5.0±2.2 ; +4.7±1.9 | +3.1±1.0 ; +0.6±0.6 | +2.8±0.9 ; +1.2±0.7 |
| hlg − r3b | +6.5±2.4 ; +10.0±2.3 | +9.3±2.2 ; +9.7±2.3 | +2.6±2.2 ; +1.0±2.0 | +3.5±1.0 ; +0.8±0.7 | +3.0±1.0 ; +1.0±0.7 |
| reg − p24c | +33.8±2.8 ; +30.1±2.5 | +20.9±2.3 ; +20.7±2.3 | +22.8±2.7 ; +19.9±2.4 | +4.0±1.1 ; +0.8±0.7 | +3.5±1.0 ; +1.4±0.8 |

## Conclusions

1. **HL-Gauss critic gives no gain.** hlg ≈ reg everywhere; vs V32.1 it is slightly worse (−2.4±2.0 margin, −3.7±1.9 pp win, ~1.9σ).
   Consistent with the offline dist-value A/B (reports/value/dist_value_ab.md): the return distribution is not where the error is.
2. **PPO r2 reg ≈ r1 on the training opponents, better vs V32.1 (+7.4±1.8)** — r1's one weak spot (−2.4 vs r3b on V32.1) is gone;
   possibly from critic_warmup 10 or run-to-run variance (single seed per arm; fp16 cross-GPU noise also applies). At ~0.62× r1 wall-clock.
3. PPO (both arms) stays clearly ahead of GRPO r3b at equal simulator-decision budget (+7…+10 margin, ~+10 pp win vs p24c/p22).
4. Held-out p5a/p16 saturated (≥99 % win), small consistent +3 margin over r3b.

**Recipe decision:** keep the scalar (reg) critic + speed flags (update_fp16, critic_fp16, update_compile, critic_warmup 10) as the RL
recipe for the post-patch pipeline; ppo_r2_reg/final.pt is the new best checkpoint. Learning-curve / sample-efficiency comparison: reports/rl/rl_sample_efficiency.md (next).
