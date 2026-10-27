# PPO HPO sweep (gpu2 GPU2, 2026-10-27)

Recipe: `reports/rl/ppo_hpo.sh` (init = KL ref = P24c, β 0.01, pool 9 + V33, 10M decisions ≈ 313 its), one factor per trial
(`reports/rl/hpo_trials.txt`). Eval `reports/rl/hpo_eval.sh`: iter_50/100/200 at 128 seeds, final at 256 seeds × 2 sides vs
p24c / p22 / V32.1 / V33; **avg4** = mean margin over the four. Numbers via `reports/rl/hpo_analysis.py`.
Speed = **curve** = paired avg4 diff vs base averaged over the 4 checkpoints (first 128 seeds). Significant = ≥ +2 and > 2 SE.

## Base curve (avg4 margin)
| iter_50 | iter_100 | iter_200 | final (313) |
|---|---|---|---|
| +21.3±1.5 | +24.1±1.4 | +25.5±1.4 | +27.1±1.2 |

## Trials vs base
| trial | change | iter_50 | iter_100 | iter_200 | final | curve | verdict |
|---|---|---|---|---|---|---|---|
