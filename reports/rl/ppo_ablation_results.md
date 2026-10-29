# PPO ablation results (league-B recipe, 2026-10-24 … 10-26)

Plan: `reports/rl/ppo_ablation_plan.md`. All arms = league-B recipe (`reports/rl/ppo_league_b.sh`) with one change, run on gpu2.
Eval: fast batched eval (`reports/rl/abl_eval.sh`), 256 seeds (970000–970255) × 2 sides vs p24c, p22, V32.1, paired by seed;
**avg3** = mean margin diff over the three. Raw jsons in `reports/rl/abl_eval/`, numbers via `reports/rl/abl_analysis.py a:b`.
Noise floor (second seed of B): B2 − B = −1.2±1.2 at it300 and −2.2±1.2 at it600.

| arm | change | budget | control | avg3 vs control | notes | verdict |
|---|---|---|---|---|---|---|
| E1 lam99 | GAE λ 0.95 → 0.99 (segment 256, games 128) | 10M | B it300 | −0.3±1.2 | | null |
| E2 lam995 | λ 0.995, segment 512 | — | — | — | dropped after E1 null | — |
| E5 offset | Gaussian angle offset, learnable σ, in ratio + KL | 10M | B it300 | +1.1±1.1 | hunt-OOD subset −3.6±1.9 (v30_reckless −12.6) | rejected |
| E3 refreset | KL anchor ← learner at each snapshot | 20M | B it600 | −0.4±1.2 | | null |
| **E4 beta01** | **β 0.02 → 0.01** | 20M | B it600 | **+4.6±1.1** | +6.8 vs B2 it600; H2H vs B600 +3.8±1.8; **OOD-26 PASS** (mean +70.3 vs +68.6, worst +40.7 vs +37.9) | **winner → new default** |
| beta005 | β 0.005 | 20M | B it600 | +3.2±1.2 | −1.4 vs E4 | worse than E4 |
| beta0 | β 0 | 20M | B it600 | +0.7±1.2 | specialises (p22 +9.7, V32.1 −6.0); KL 0.71 | worse than E4 |

## Combos with β 0.01 (20M, control = E4 final)

| arm | avg3 vs E4 | p24c / p22 / V32.1 | H2H vs E4 | final KL(π‖ref) | verdict |
|---|---|---|---|---|---|
| fine (32×32 head, fine_tau 1.5, fine_lr 2e-5) | −3.2±1.1 | −5.6 / −3.5 / −0.6 | −1.8±1.8 | 0.19 (fine 0.15) | rejected |
| offset (offset_rl) | **−20.6±1.4** | −25.6 / −3.5 / −32.8 | −5.6±1.8 | 0.43 | rejected (collapses vs p24c/V32.1) |
| lam99 (λ 0.99, seg 256) | −7.5±1.2 | −8.0 / −8.9 / −5.6 | −6.9±1.8 | 0.04 | rejected |

E4 itself ends at KL 0.04. With β 0.01 the stochastic-direction heads (offset, fine) drift far from the reference
(KL 0.19–0.43) and lose; the offset arm loses ~25–33 points vs p24c and V32.1. λ 0.99 keeps KL low but is still worse
(the γ=1 sparse terminal reward makes the longer λ trace mostly variance).

## Conclusions
- Keep the discrete 33-way head + deterministic offset; β 0.01; λ 0.95; KL anchor = init policy.
- Finer direction control did not help under PPO here (both offset and 32×32 fine head lose, alone or combined).
- Next: F1 = continue from E4 (init = KL ref = E4, critic warm-started) with V33 added to the pool (`reports/rl/ppo_f1.sh`).

## F1: continue from E4 with V33 in the pool (`reports/rl/ppo_f1.sh`, 20M, 2026-10-26/27)
init = KL ref = E4, critic warm-started from E4, β 0.01, pool = 9 B opponents + V33.
- vs E4 (paired, 256 seeds): p24c +2.4±1.9, p22 +6.6±1.8, V32.1 +3.4±1.7, **avg3 +4.1±1.1**; **H2H vs E4 +10.1±1.9 (win 62.2 %)**; avg3 vs B600 +8.7±1.2.
- vs V33 and hunt-style OOD (`reports/rl/angle_probe/f1__*`, paired with E4 = q0): **V33 +12.4±2.2 (E4 −12.1 → F1 +0.3)**,
  v30_h4hunt +7.8±1.6, v30_hunt +3.9±1.7, v30_reckless +3.0±1.8, early_rush +0.2, rush_all −0.4.
- **OOD-26 vs E4: PASS** (reports/rl/abl_ood_f1/, 32 seeds × 2 from 2600000, E4 rows from abl_ood_beta01): mean +71.2 vs +70.3,
  worst +43.7 (v30_c0) vs +40.7 (v30_h4hunt); v30_h4hunt +12.7±3.9, v30_stage3 +6.5; v30_c0 −7.2±3.9 (watch: E4's own weak spot, within 2 SE), no opponent < −5 at 2 SE.
- → **F1 = new best RL policy** (checkpoints/rl/f1_v33/final.pt on gpu2).

## F2: continue from F1 with lr 6e-6 (HPO winner; `reports/rl/q_f2.sh`, 20M, 2026-10-29)
init = KL ref = F1, critic F1, β 0.01, pool 9 + V33, **lr 3e-6 → 6e-6** (reports/rl/hpo_results.md).
- vs F1 (paired, 256 seeds): p24c +7.7±1.9, p22 +2.8±1.6, V32.1 −1.8±1.7, **avg3 +2.9±1.0**; **H2H vs F1 +22.6±2.1 (win 71.3 %)**.
- V33 + hunt OOD (`angle_probe/f2__*`, paired with E4 q0): **V33 +45.5 vs E4 (F1 +12.4) → F2 beats V33 by ≈ +33**;
  v30_h4hunt +8.8, v30_hunt +6.3, v30_reckless +5.9 (all ≥ F1).
- **OOD-26 vs F1: PASS** (reports/rl/abl_ood_f2/): mean +71.9 vs +71.2, worst +51.9 (v30_h4hunt) vs +43.7 (v30_c0);
  v30_c0 +11.3±7.7, no opponent < −5 at 2 SE (lowest cargo_hunter −3.8±2.0).
- → **F2 = new best RL policy** (checkpoints/rl/f2_lr6/final.pt on gpu2).

## F3: continue from F2 with lr 8e-6 (`reports/rl/q_f3all.sh`, 20M, 2026-10-29)
init = KL ref = F2, critic F2, β 0.01, pool 9 + V33, lr 8e-6.
- vs F2 (paired, 256 seeds): p24c −3.0±1.8, p22 +0.5±1.5, V32.1 −0.3±1.7, **avg3 −0.9±1.0 (null)**; **H2H vs F2 +25.2±2.1 (win 72.4 %)**.
- V33 + hunt OOD (paired with E4 q0): **V33 +33.3 vs F2 +45.5 → −12.2 vs F2**; v30_h4hunt +7.8 (F2 +8.8), v30_hunt +5.7 (+6.3), v30_reckless +6.0 (+5.9).
- OOD-26 vs F2: PASS (mean +72.4 vs +71.9, worst +48.0 v30_c0 vs +51.9; v30_c0 −7.0±6.7 within 2 SE).
- → **Mixed / non-transitive:** F3 beats F2 head-to-head by a lot but loses ground to V33 (−12) and p24c (−3) and is level elsewhere.
  **F2 stays the champion** (better vs the strongest heuristic, equal elsewhere); F3 kept as a league opponent candidate.
