# Value net A/B: unit velocity (Δpos) × attention pooling

**Verdict:** Δpos does not help the value net. Attention pooling gives a small in-distribution gain but is consistently worse on
the OOD suite (the worst-case split). Keep the mean-pool head without Δpos; a temporal (KDA) value net is not recommended on this evidence.

## Setup
- Recipe identical to `reports/value/dist_value_ab.md` (reg arm, final beta 10): S1c settings on true Unity outcomes, cache
  `datasets/value_s1_r64` (64 rows/match), P10 trunk warm start, trunk lr ×0.3, lr 3e-4, 16k steps, batch 256, fp16 + compile, final checkpoint.
- 2×2 {base, +Δpos} × {mean-pool, attn-pool}, 3 seeds each (12 runs, ~13 min each on the WSL RTX 3060), same rows / steps / seeds.
- **Δpos** (`blackout_env/value/velocity.py`): `agent_vel` [N,10,3] = (vx, vy, valid) from the previous shard row, same canonical frame,
  ÷ tick gap ÷ 0.02 (normal speed |v| ≈ 1). valid=0 on a match's first row, gaps > 10 ticks / unknown tick, and jumps > 0.045 per tick
  (death, respawn, teleport); 98.1 % valid. Enters through a zero-initialised linear map into the unit tokens (net = base at init),
  trained at the head lr. Cache: `examples/value_train.py addvel` (same rows, checked row by row).
- **Attention pool** (`ValueNet(pool="attn", n_queries=4)`): 4 learned queries, 4-head cross-attention over all 48 post-trunk tokens,
  LayerNorm'd keys/outputs; the head reads [team_state, CLS, 4 pooled] (6H) instead of [team_state, own mean, enemy mean, CLS] (4H) —
  the head-width change is part of "pool".
- Both flags default off; old checkpoints load (`tests/test_value_vel_pool.py`).
- Eval (`examples/value_vp_predict.py`, `examples/value_vp_analyze.py`): heldout = 8,256 rows / 129 matches; unseen_policy and
  unseen_ood = 60k rows each (never trained on), same rows for every arm. "Contested" = 12-run mean P(win) in 0.2–0.8.
  Effects paired by seed, mean ± SE over 3 seeds (|effect| < ~2 SE = noise). All tables: `vel_pool_ab/tables.md`, `metrics.json`, run logs `vel_pool_ab/logs/`.

## Main table (mean ± seed sd)
| split | arm | final MAE | RMSE | R² | result logloss | win Brier | ECE | p99 abs err |
|---|---|---|---|---|---|---|---|---|
| heldout | base | 11.69±0.12 | 23.02±0.25 | 0.765 | 0.674±0.018 | 0.0602 | 0.058 | 101.4 |
| | +Δpos | 11.67±0.14 | 23.08±0.26 | 0.764 | 0.678±0.016 | 0.0611 | 0.059 | 101.0 |
| | attn | 11.59±0.14 | 22.92±0.26 | 0.767 | 0.655±0.011 | 0.0596 | 0.058 | 98.1 |
| | Δpos+attn | **11.47±0.17** | **22.65±0.17** | **0.773** | **0.652±0.018** | 0.0601 | 0.058 | 98.3 |
| unseen_policy | base | 10.34±0.07 | 21.41±0.14 | 0.714 | 0.607±0.013 | 0.0522 | 0.050 | 101.1 |
| | +Δpos | 10.32±0.04 | 21.38±0.10 | 0.715 | 0.600±0.007 | 0.0510 | 0.049 | 100.9 |
| | attn | 10.25±0.03 | 21.29±0.06 | 0.717 | 0.603±0.029 | 0.0513 | 0.050 | 100.9 |
| | Δpos+attn | **10.21±0.04** | **21.24±0.16** | **0.718** | **0.596±0.008** | 0.0508 | 0.049 | 101.4 |
| unseen_ood | base | **14.94±0.08** | **27.84±0.10** | **0.734** | **0.785±0.029** | 0.0688 | 0.067 | 118.1 |
| | +Δpos | 15.09±0.14 | 28.08±0.18 | 0.730 | 0.802±0.020 | 0.0691 | 0.067 | 118.7 |
| | attn | 15.21±0.14 | 28.26±0.19 | 0.726 | 0.791±0.026 | 0.0693 | 0.067 | 119.3 |
| | Δpos+attn | 15.15±0.07 | 28.18±0.04 | 0.728 | 0.801±0.015 | 0.0692 | 0.067 | 119.5 |

Base reproduces the earlier A/B's reg arm (11.73 / 10.35 / 14.97).

## Factorial effects (mean ± SE; negative MAE/RMSE/logloss = better)
| split | metric | Δpos main | pool main | interaction |
|---|---|---|---|---|
| heldout | MAE | -0.07±0.11 | -0.15±0.13 | -0.10±0.04 |
| | RMSE | -0.11±0.20 | -0.27±0.15 | -0.33±0.15 |
| | R² | +0.002±0.004 | +0.005±0.003 | +0.007±0.003 |
| | logloss | +0.001±0.017 | -0.022±0.003 | -0.007±0.010 |
| unseen_policy | MAE | -0.03±0.02 | -0.11±0.03 | -0.01±0.05 |
| | RMSE | -0.04±0.02 | -0.13±0.04 | -0.01±0.11 |
| | R² | +0.001±0.000 | +0.003±0.001 | +0.000±0.003 |
| | logloss | -0.007±0.007 | -0.004±0.006 | -0.001±0.014 |
| unseen_ood | MAE | +0.04±0.09 | **+0.16±0.03 (worse)** | -0.21±0.14 |
| | RMSE | +0.08±0.10 | **+0.26±0.05 (worse)** | -0.31±0.17 |
| | R² | -0.002±0.002 | -0.005±0.001 | +0.006±0.003 |
| | logloss | +0.013±0.013 | +0.003±0.008 | -0.007±0.024 |

Worst of the two unseen splits (always unseen_ood): base 14.94 MAE / 0.785 logloss / p99 118.1; +Δpos 15.09 / 0.802 / 118.7;
attn 15.21 / 0.791 / 119.3; Δpos+attn 15.15 / 0.801 / 119.5. Worst-case MAE effects: Δpos +0.04±0.09, pool +0.16±0.03, interaction -0.21±0.14.

## By elapsed time (final MAE)
| split | bucket | base | +Δpos | attn | Δpos+attn |
|---|---|---|---|---|---|
| heldout | 0–10 s | 23.07 | 23.28 | 22.52 | 22.46 |
| | 10–20 s | 12.44 | 12.26 | 12.79 | 12.38 |
| | 20–30 s | 7.88 | 7.69 | 7.74 | 7.65 |
| | 30–45 s | 6.05 | 6.08 | 5.91 | 6.04 |
| | >45 s | 3.56 | 3.56 | 3.51 | 3.46 |
| unseen_policy | 0–10 s | 17.50 | 17.46 | 17.21 | 17.16 |
| | 10–20 s | 9.44 | 9.45 | 9.43 | 9.39 |
| | 20–30 s | 5.87 | 5.83 | 5.93 | 5.84 |
| | 30–45 s | 5.88 | 5.86 | 5.75 | 5.84 |
| | >45 s | 3.91 | 3.87 | 3.89 | 3.85 |
| unseen_ood | 0–10 s | 27.78 | 28.28 | 28.42 | 28.43 |
| | 10–20 s | 13.16 | 13.17 | 13.23 | 13.10 |
| | 20–30 s | 8.54 | 8.52 | 8.67 | 8.67 |
| | 30–45 s | 8.27 | 8.37 | 8.56 | 8.45 |
| | >45 s | 5.94 | 5.89 | 5.99 | 5.89 |

Contested subset (final MAE; n = 471 / 2,584 / 4,025):
| split | base | +Δpos | attn | Δpos+attn |
|---|---|---|---|---|
| heldout | 42.27±1.20 | 42.15±0.80 | 41.54±0.89 | 41.25±0.96 |
| unseen_policy | 38.08±0.51 | 37.21±0.62 | 37.36±0.44 | 36.93±0.08 |
| unseen_ood | 47.87±0.45 | 48.54±0.30 | 48.34±0.73 | 48.36±0.19 |

## Findings
1. **Δpos alone does not help**: main effects within ~1 SE of zero on heldout and unseen_ood; tiny gain on unseen_policy (MAE -0.03±0.02).
   One-step velocity is not a missing signal for the value on this data.
2. Where it should matter (contested states) it helps only in-distribution: unseen_policy contested MAE -0.65±0.13, logloss -0.14±0.05;
   on unseen_ood contested it reverses (+0.34±0.38). The opening shows no Δpos effect.
3. **Attention pooling**: small in-distribution gain — heldout logloss -0.022±0.003, p99 -3 points, unseen_policy MAE -0.11±0.03, largest
   in the opening (heldout 0–10 s MAE -0.69±0.33).
4. **…but worse OOD**: unseen_ood MAE +0.16±0.03, RMSE +0.26±0.05 (~5 SE) — the one clear effect there, on the worst-case split.
   Mean pooling is the OOD-safer head.
5. Best in-distribution arm is Δpos+attn (heldout MAE -0.22, unseen_policy -0.14±0.02 vs base; heldout interaction -0.10±0.04 hints
   Δpos only pays with the pool), yet it is worse than base on unseen_ood (MAE +0.21±0.08). Suggestive only (3 seeds, 129 heldout matches).

## Recommendation
- Δpos: not adopted as a value input (code stays, default off).
- Attention pooling: not adopted — gains are ~1–2 % MAE in-distribution, while OOD worst case (the gate the user set) gets worse.
- Temporal / KDA value net: not recommended now. The cheapest temporal signal showed no robust gain and was mildly negative OOD.
  A memory net could still capture longer intent (approach, camping) that one-step Δpos cannot; any such test must gate on the worst unseen split.
- The value net's remaining error (MAE ~11, R² ~0.77, opening MAE ~23) looks dominated by outcome stochasticity / opening uncertainty
  rather than missing per-frame observations.

## Caveats
- Single view, no mirror symmetrisation; shard data is the learner's side (mostly the stronger one).
- 3 seeds, 129 heldout matches (heldout seed sd noisy); final checkpoint, no selection.
- Δpos at play time: available in both pipelines (official `tracker.py` keeps per-observation positions; Unity env previous obs),
  but the code must pass the tick gap since the feature is per tick and decision cadence may differ from the shards (mostly 1 tick back).
- First launch lost to the WSL freeze (base_s0, vel_s0); both re-run from scratch after the NVIDIA/Intel driver update (0 nvlddmkm errors during the 3 h sweep).
