# Critic shift diagnostic — PPO r2_reg critic vs pool and "new strong" opponents

**Question (before league arm B):** the PPO critic is opponent-blind. How biased is it when a stronger opponent (a policy snapshot)
enters the pool, how much of that leaks into the advantages, and which safeguards are worth it?

**Verdict:** the critic is badly miscalibrated vs new strong opponents (MC bias −0.2…−0.3, −1.1 in the first 10 s), but with sparse
terminal reward, γ = 1 and GAE λ = 0.95 that bias telescopes away: mean advantage shifts only −0.005 (~50× smaller). What new opponents
add is *more within-game advantage variance in the decisive 10–45 s window* — i.e. more learning signal, not a harmful offset.
Safeguards for B can be minimal: per-opponent centering (cheap insurance), no per-opponent std, no long critic warm-up.

## Setup
- Learner: `checkpoints/rl/ppo_r2_reg/final.pt` + `final_critic.pt` (scalar critic), no parameter updates.
- Rollouts: `PPOTrainer.collect()` verbatim (games 280, segment 128, T = 1 sampling, GAE γ = 1 λ = 0.95, critic_fp16) through a read-only
  `diag_hook` in `blackout_env/rl/ppo.py` (no-op when unset). Balanced opponent / side assignment; 284 iterations, 10.2M rows.
- Opponents (≥ 400 finished games each): pool = p24c_v321, p22_v31, V32.1; new = r2reg_final (self), r2reg_it300, r1_final, r3b_final;
  crushed held-out p5a_v22 / p16_v27 (152 / 163 games).
- μ_GAE = mean(ret − V) = mean advantage; μ_MC = mean(final reward − V) over finished games (true calibration error of V).
  EV_gae uses the GAE returns (as in PPO's log); EV_mc the real outcome. SEs are game-clustered.
- Ran on the WSL RTX 3060 (gpu2 GPU0/1 franknoh, GPU2 another process). Code: `examples/critic_shift_diag.py` (collect / analyze /
  finetune / ftreport). Raw: `reports/rl/critic_shift_diag/stats.json`, `finetune*.json`; figures `bias_by_opponent_time.png`,
  `hist_v_ret.png`, `normalisation.png`, `critic_warmup_recovery.png`, `critic_warmup_lr.png`.
- Games are short (most end on score well before 420 s): rows by elapsed 0–10/10–20/20–30/30–45/>45 s = 9/9/9/13/60 % (pool), 7/7/7/10/70 % (new).

## Per opponent
| opp | win | margin | μ_GAE | μ_MC (±SE) | EV_gae | EV_mc | \|adv\| | adv std | V mean |
|---|---|---|---|---|---|---|---|---|---|
| p24c_v321 (pool) | .79 | +30 | −.0027 | −.117 ±.027 | .9964 | .70 | .0106 | .047 | .68 |
| p22_v31 (pool) | .84 | +37 | −.0023 | −.044 ±.022 | .9970 | .74 | .0106 | .046 | .65 |
| V32.1 (pool) | .88 | +45 | −.0015 | −.062 ±.019 | .9976 | .83 | .0078 | .041 | .70 |
| r2reg_final (new) | .49 | +0.4 | −.0053 | −.250 ±.029 | .9968 | .56 | .0141 | .053 | .20 |
| r2reg_it300 (new) | .50 | +0.3 | −.0049 | −.185 ±.029 | .9971 | .59 | .0141 | .052 | .13 |
| r1_final (new) | .48 | −2.4 | −.0049 | −.203 ±.028 | .9971 | .59 | .0136 | .052 | .16 |
| r3b_final (new) | .61 | +8.1 | −.0039 | −.161 ±.026 | .9972 | .67 | .0122 | .049 | .41 |
| p5a_v22 (crushed) | 1.00 | +67 | +.0001 | n/a | .9974 | n/a | .0026 | .0133 | 1.14 |
| p16_v27 (crushed) | .99 | +73 | +.0009 | n/a | .9988 | n/a | .0027 | .0117 | 1.13 |

By elapsed bucket (0–10 / 10–20 / 20–30 / 30–45 / >45 s):

| opp | μ_MC | μ_GAE | \|adv\| |
|---|---|---|---|
| p24c | −.40 / −.24 / −.14 / −.06 / −.06 | −.008 / −.010 / −.005 / −.004 / −.000 | .015 / .030 / .021 / .016 / .004 |
| V32.1 | −.19 / −.08 / −.07 / −.03 / −.04 | −.007 / −.003 / −.002 / −.003 / +.000 | .016 / .025 / .009 / .007 / .003 |
| r2reg_final | −1.10 / −.91 / −.67 / −.41 / −.03 | −.009 / −.017 / −.014 / −.014 / −.002 | .014 / .036 / .033 / .032 / .008 |
| r1_final | −1.12 / −.86 / −.58 / −.33 / −.01 | −.011 / −.022 / −.014 / −.013 / −.001 | .016 / .043 / .033 / .032 / .007 |
| r3b_final | −.84 / −.59 / −.40 / −.23 / −.02 | −.011 / −.019 / −.008 / −.010 / −.001 | .017 / .041 / .029 / .029 / .005 |

EV_gae by bucket: pool ≈ .80 / .96 / .985 / .996 / .9997; new .70–.86 / .95–.96 / .98 / .99 / .999. V is massed at +1 vs the pool and
bimodal vs new opponents (`hist_v_ret.png`): the critic is over-confident early because it cannot see who the opponent is.

## Findings
1. **Telescoping confirmed.** μ_MC −0.2…−0.3 overall (−1.1 in 0–10 s for self-play) vs μ_GAE −0.005. The identity
   mean adv ≈ 20·(R − V₀)/N_game predicts it (p24c −.0034 vs observed −.0029; V32.1 −.0020 / −.0015; r2reg_final −.0065 / −.0053):
   the bias only enters through the first / last ~20 decisions; the last 20 hold ~1 % of the squared-advantage mass.
2. **What leaks is within-game variance in 10–45 s, not an offset.** New-opponent games have 1.3–4× larger |adv| and 1.2–2× larger
   adv std there (20–30 s |adv| .033 vs .009 for V32.1), diluted globally because 60–70 % of rows are > 45 s (|adv| .003–.008).
   EV_gae (.997) is blind to this; only MC measures show it (EV_mc .56–.67 new vs .70–.83 pool).
3. **Variance decomposition (raw adv).** Pool: between-opponent offset 0.01 %, between-game 2.8 %, between-5 s-block 25.4 %, residual
   71.8 %; opponent × time-bucket mean structure 0.4 %. Adding any new opponent: offset 0.04–0.10 %, opp × bucket 0.7–0.9 %.
4. **Mixed batches** (game-cluster bootstrap ×300; pool-only global adv std 0.0448): r2reg_final at 5 / 10 / 20 % inflates the global
   std ×1.010 / 1.020 / 1.039 [1.017–1.064] → pool-game signal ×0.990 / 0.981 / 0.962; r3b_final at 20 % ×1.018. Real but small.
5. **Critic-only fine-tune (PPO's own critic update, lr 1e-5, 255 segments/it, Adam pre-rolled 5 pool-only it, held-out games)**
   barely moves the bias beyond a pool-only control: r2reg_final 10 % μ_MC −.285 → −.271, EV_mc .564 → .598 at k = 20 vs control
   −.279 / .588; other opponents ≤ 0.02 μ_MC / ≤ 0.03 EV_mc beyond control. lr ×3 / ×10 no better. Pool accuracy not hurt.
   The early bias is largely irreducible for an opponent-blind critic (early states look the same across opponents).

## Normalisation comparison (r2reg_final at 20 %; 10 % in brackets)
| scheme | new-opp offset (σ) | pool signal vs pool-only | offset power share | gain on pool opponents |
|---|---|---|---|---|
| (a) current: global std only | −0.114 [−.128..−.099] (−.116) | 0.962 (0.981) | 0.26 % (0.13 %) | 0.96 uniform |
| (b) global centering + global std | −0.054 [−.066..−.041] | 0.962 | 0.06 % | 0.96 |
| (c) per-opp centering + global std | 0 | 0.963 | 0 | 0.96 |
| (d) per-opp centering + per-opp std | 0 | 1.000 | 0 | p24c 0.948, p22 0.978, V32.1 1.09 |

Under (c) the new opponent's rows keep scale 1.14 (its extra within-game variance remains — that is signal). (d) amplifies crushed
opponents' noise 3.4–3.8× (adv std .012–.013 vs .045). All offsets are only 0.03–0.12 σ; pool-signal shrink is 1–4 %.

## Recommendation for league arm B
1. Normalisation (c): per-opponent centering, then global std; only for opponents with ≥ ~10 games in the batch. Not (d).
2. Critic warm-up on snapshot add: not needed (no offset leak to fix; k ≤ 20 buys nothing measurable). k = 2–3 at most as a margin.
3. Ramp / health gates on per-opponent stats over finished games: |μ_GAE| / adv std < 0.25 (observed .03–.10, non-binding);
   adv std / pool adv std ≤ 1.5 (new 1.1–1.2); EV_mc ≥ 0.4 with ≥ 100 finished games (new .56–.67). Do not gate on EV_gae or on early MC
   bias. A share ramp 5 → 10 → 20 % is safe (global std ≤ +4 % at 20 %, CI ≤ +6.4 %).
4. The plateau's likely cause: the advantage scale lives in the first ~45 s; rows past 45 s carry |adv| .003–.008 and are nearly silent
   after normalisation. Stronger opponents raise exactly the 10–45 s signal, which supports B; weighting / sampling the decisive window
   is a possible later lever.

## Caveats
- The real critic's Adam state is not saved; 5 pool-only iterations were pre-rolled.
- MC stats use ~400 finished games per opponent (in-flight games excluded), game-clustered SEs.
- Fine-tune used a random subset (~2,400 segments / opponent, 2.09M rows); eval 300 held-out segments per opponent per k (EV_mc noise ±0.01).
  Pool-only control only at lr ×1; lr ×3 / ×10 only for r2reg_final at 10 / 20 %.
- "Pool" here = the 3 named opponents with equal shares, not r2's full 9-opponent pool (which also has p21e, p20, p17, p14, d6, mixture).
- Win/margin are learner T = 1 sampling vs greedy opponents (lower than the greedy eval numbers). Entropy / logp were not recorded.
