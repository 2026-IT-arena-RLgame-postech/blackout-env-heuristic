# Softmax Collapse check (arXiv 2610.31642, 2026-10-29)

`reports/rl/softmax_margin_check.py`: greedy games vs p24c (4 seeds × 2 sides, full length, MPS fp32), every own-unit
logit row [33]; collapse margin m = −LSE_{j≠top}(z_j − z_top) (1 − p_top ≈ e^−m). Cliffs: bf16 5.5, fp16 ≈8.3, fp32 16.6.
Raw: `softmax_margin_check.txt`.

| policy | rows | median m | p99 | max | > bf16 | > fp16 | > fp32 |
|---|---|---|---|---|---|---|---|
| F1 (RL) | 54k | 1.4 | 4.7 | 9.2 | 0.37 % | 0.004 % | **0** |
| E4 (RL) | 104k | 1.1 | 4.6 | 8.4 | 0.16 % | 0.001 % | **0** |
| P24c (distill) | 49k | 1.2 | 4.5 | 8.2 | 0.18 % | 0 | **0** |
| P22 (distill) | 138k | 1.2 | 7.3 | 13.2 | 11.5 % | 0.35 % | **0** |

- No policy gets near the fp32 cliff (max m 13.2 < 16.6). PPO updates and the KL term run in fp32 (`update_fp16` off), and the
  distill loss (`set_valued_cross_entropy`) is computed on `.float()` logits with a z-loss → **Softmax Collapse does not occur
  in our training; the fix is not needed.**
- Only relevant if we ever compute the loss in bf16/fp16 (P22-like confident students would cross those cliffs on 0.3–11 % of rows).
