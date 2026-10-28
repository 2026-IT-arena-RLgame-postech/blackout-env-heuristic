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
| clip0.3 | clip 0.2 → 0.3 | +0.8±1.4 | −0.6±1.6 | +3.0±1.5 | +0.8±1.2 | +1.3±0.8 | null (mildly positive, balanced) |
| lr6e-6_r2 | replicate of lr6e-6 | +1.5±1.4 | +1.6±1.4 | +1.2±1.5 | **+4.5±1.3** (p24c +5.2, p22 +7.2, V32.1 −1.4, V33 +7.2) | **+2.2±0.9** | **replicates** (r2 − r1: final −1.7±1.2, curve +0.0) |
| lr6e-6_clip0.3 | lr 6e-6 + clip 0.3 | +2.2±1.5 | +1.4±1.3 | **+3.5±1.4** | +3.1±1.2 (p24c −1.0, p22 +6.9, V32.1 +0.3, V33 +6.1) | **+2.9±0.8** | significant (curve, fastest); vs lr6e-6_r2: final −1.4±1.2, curve +0.7±0.8 → n.s. (slightly faster, not stronger) |
| lr8e-6 | lr 3e-6 → 8e-6 | **+4.6±1.5** | +3.6±1.4 | −2.7±1.7 | **+7.1±1.4** (p24c +2.5, p22 +13.4, **V32.1 −5.4**, V33 +17.9) | **+3.3±0.9** | significant (final + curve, highest numbers) but dips at it200 and leans to V33/p22 at V32.1's cost; vs lr6e-6_r2: final +2.6±1.3, curve +1.0±1.0 |

**Single-factor summary:** only lr 6e-6 is significant (final +6.2). Smaller steps (lr 1.5e-6, clip 0.1) are slower; more update
steps (ep4, mb512) do not help and mb512 specialises; lr 1.2e-5 specialises. Next (queued 10-28): lr6e-6_r2 (replicate),
lr6e-6_clip0.3 (combo of the two step-size-up settings), lr8e-6.

## Conclusion (10-29)
- **lr is the only lever that matters** in this recipe (10M, P24c start): 3e-6 → 6e-6 gives final +4.5…+6.2 (replicated) and curve +2.2;
  8e-6 gives the highest final/curve (+7.1 / +3.3) but is less balanced (V32.1 −5.4) and dips mid-run; 1.2e-5 over-specialises (V32.1 −15).
- clip 0.3 on top of lr 6e-6 makes the curve slightly faster (+2.9) without a higher final; epochs/minibatch/clip 0.1 do not help.
- **Recommended default: lr 6e-6** (balanced, replicated). lr 8e-6 is the aggressive option — adopt only after an OOD-26 worst-case check.

## OOD-26: lr 8e-6 vs lr 6e-6 finals (10-29, `reports/rl/hpo_ood_<t>/`, 32 seeds × 2 from 2600000, examples/ood_report.py)
| | mean | worst | cat (lost by > 20) |
|---|---|---|---|
| lr 6e-6 | +68.1 | +40.4 (v30_c0) | 3.8 % |
| **lr 8e-6** | **+71.3** | **+51.4** (v30_h4hunt) | 1.9 % |

lr 8e-6 − lr 6e-6: **R1–R4 PASS**, no opponent < −5 at 2 SE (lowest v30_camp −6.2±5.1); v30_c0 +25.6±8.0, v30_stage3 +15.4±4.8, v30_hunt +9.0.
→ lr 8e-6 is not fragile OOD (its V32.1 −5.4 in-pool dip does not carry over); **lr 8e-6 is a valid aggressive default**, 6e-6 the safe one.
