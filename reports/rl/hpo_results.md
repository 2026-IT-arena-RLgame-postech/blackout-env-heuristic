# PPO HPO sweep (gpu2 GPU2, 2026-10-27)

Recipe: `reports/rl/ppo_hpo.sh` (init = KL ref = P24c, β 0.01, pool 9 + V33, 10M decisions ≈ 313 its), one factor per trial
(`reports/rl/hpo_trials.txt`). Eval `reports/rl/hpo_eval.sh`: iter_50/100/200 at 128 seeds, final at 256 seeds × 2 sides vs
p24c / p22 / V32.1 / V33; **avg4** = mean margin over the four. Numbers via `reports/rl/hpo_analysis.py`.
Speed = **curve** = paired avg4 diff vs base averaged over the 4 checkpoints (first 128 seeds). Significant = ≥ +2 and > 2 SE.

## Base curve (avg4 margin)
| iter_50 | iter_100 | iter_200 | final (313) |
|---|---|---|---|
| +21.3±1.5 | +24.1±1.4 | +25.5±1.4 | +27.1±1.2 |

Base lr = 3e-6 (PPOConfig default).

## Trials vs base
| trial | change | iter_50 | iter_100 | iter_200 | final | curve | verdict |
|---|---|---|---|---|---|---|---|
| lr6e-6 | lr 3e-6 → 6e-6 | +0.8±1.5 | +1.5±1.5 | −0.3±1.5 | **+6.2±1.2** (p24c +9.5, p22 +6.4, V32.1 −0.6, V33 +9.5) | +2.2±0.8 | **significant (final)**; gain appears only after it200 (abs. 25.2 → 33.3) |
| lr1.2e-5 | lr 3e-6 → 1.2e-5 | −0.7±1.6 | +3.0±1.5 | **+4.3±1.6** | +2.4±1.6 (p24c −0.2, p22 +8.8, **V32.1 −15.4**, V33 +16.4) | +1.8±0.9 | not significant; fastest early (it100–200) but specialises (V32.1 collapse) → too high |
| lr1.5e-6 | lr 3e-6 → 1.5e-6 | −4.3±1.5 | −3.3±1.5 | −0.7±1.4 | −0.6±1.2 | −1.7±0.8 | slower start, same final → rejected |

**lr summary:** 1.5e-6 slower, 6e-6 best final (+6.2, balanced), 1.2e-5 fastest to it200 but specialises → **lr 6e-6** for combos.
| ep4 | epochs 2 → 4 | −0.3±1.5 | +0.8±1.5 | −1.2±1.5 | +1.7±1.3 (V33 +6.0) | +0.2±0.8 | null (2× update cost, no gain) |
| mb512 | minibatch 1024 → 512 | −0.8±1.4 | +0.3±1.5 | −1.9±1.5 | −1.4±1.4 (p24c −7.6, p22 +8.9, V32.1 −9.1, V33 +2.3) | −0.7±0.9 | null/worse; specialises like lr1.2e-5 → rejected |
| clip0.1 | clip 0.2 → 0.1 | −3.1±1.4 | −2.5±1.5 | +0.2±1.4 | −2.5±1.2 | −1.6±0.8 | slower and slightly worse → rejected |
