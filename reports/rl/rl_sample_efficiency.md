# RL sample efficiency — GRPO r3b vs PPO r1 vs PPO r2 (reg)

Question: how fast does each RL recipe gain strength per simulator decision, per learner view and per wall-clock hour?
(Scope cut by the user to 3 runs: grpo_r3c ≈ r3b and the HL-Gauss critic gave no gain — reports/rl/ppo_r2_eval.md.)

## Setup
- Runs (all from p24c_v321, same ~33.35M simulator-decision budget, one training seed each):
  GRPO r3b (frozen value leaf, 210 it), PPO r1 (scalar critic, no speed flags, 1047 it, 16.6 h),
  PPO r2 reg (r1 + update_fp16 / critic_fp16 / update_compile, critic_warmup 10, 1048 it, 10.3 h).
- Every saved checkpoint (GRPO it 30…210, PPO it 150…900 + final) and step 0 = p24c_v321, vs p24c_v321, p22_v31, V32.1;
  seeds 980000–980063 × 2 sides = 128 games/point, WSL RTX 3060 (`examples/eval_grpo_ckpt.py`, driver `reports/rl/sample_eff.sh`).
- x-axes from each run's log.jsonl: decisions (PPO `decisions_total`; GRPO cumulative `decisions`), learner views
  (cumulative `views`; PPO views = decisions, GRPO ≈ 0.36 × decisions), wall-clock `wall_h`.
- Analysis `reports/rl/sample_eff_analysis.py`; raw jsons, run logs and tables in `reports/rl/sample_eff/`.

## Figures
- `sample_eff/fig_margin.png`, `fig_winrate.png`: rows = decisions / views / hours, columns = opponent + 3-opponent average, 95 % CI.
- `sample_eff/fig_heatmap.png`: checkpoint × opponent margin.
- `sample_eff/fig_paired_reg_minus_r1.png`: paired r2_reg − r1 per checkpoint.
- `sample_eff/fig_critic_diag.png`: critic diagnostics r1 vs r2_reg.

## Strength vs budget (3-opponent average; nearest checkpoint; step 0 = +16.9, win 0.63; SE ≈ 2 per point)

| run | 25 % | 50 % | 75 % | 100 % | avg win 25/50/75/100 % |
|---|---|---|---|---|---|
| GRPO r3b | +27.0 (it 60) | +31.3 (it 120) | +35.5 (it 150) | +32.0 (it 210) | .71/.74/.78/.76 |
| PPO r1 | +42.7 (it 300) | +41.3 (it 450) | +37.5 (it 750) | +38.5 (it 1047) | .84/.85/.81/.84 |
| PPO r2 reg | +39.3 (it 300) | +44.0 (it 450) | +38.7 (it 750) | +43.6 (it 1048) | .83/.88/.84/.88 |

Paired r2_reg − r1 (avg margin, SE) at it 150/300/450/600/750/900: +1.2 (2.5), −3.4 (2.1), +2.7 (2.2), +0.4 (2.6), +1.2 (2.4), +3.4 (2.1).
Per-checkpoint tables: `sample_eff/tables.md`, `budget_table.md`, `paired_table.md`.

## Findings
1. **PPO is far more sample-efficient than GRPO.** At ~4.8M decisions PPO is at +39–40 avg margin; GRPO is at +31 and only reaches
   +35–37 after 24–29M decisions. Per learner view GRPO's first point (1.7M views, +31) is competitive, but it then stalls and needs
   ~8–10M views to approach what PPO has at 4.8M.
2. **PPO plateaus after ~5–10M decisions (150–300 it, ~1.3–2.6 h for r2).** The remaining ~70 % of the budget gives no clear gain
   (+38…+44, win 0.81–0.88). Critic diagnostics agree: explained variance ≈ 0.99 within ~10 it, |adv| falls to ~6e-3 and adv std to
   ~0.03 after ~200 it — little learning signal is left against this fixed opponent pool.
3. **Speed flags don't change learning per decision** (paired diffs all within ~1.6 SE; vf_loss / v_mean / KL / entropy curves overlap)
   and cut wall-clock ~1.6× → r2_reg is strictly better per hour (+39.8 after 1.3 h vs r1 +38.6 after 2.6 h, GRPO +35.5 after 7 h).
4. Late behaviour by opponent: vs V32.1 PPO peaks ≈ +47 around 19M decisions; r1 then drops to +35.6 at final while r2_reg ends +49.1
   (likely run noise). GRPO's final weights dip vs p24c (+18.5 vs +31.0 at it 180; final.pt = iter_210 both agree).

## Caveats
- One training seed per run, 128 games per point → differences < ~5 avg-margin points are not resolved.
- Seeds 980000+ on the 3060 differ from the 256-seed 970000+ evals (gpu2 / WSL); absolute numbers are not comparable across reports.
  Step 0 p24c vs itself = +0.7 (expected 0).
- GRPO's views axis ends at 11.9M (only part of its decisions are trained on).
- Wall-clock mixes algorithm and implementation: r1 without speed flags, GRPO pre-perf code with branch rollouts, different machine loads.
- Budget-fraction table uses the nearest checkpoint (~4.8M decisions apart), no interpolation; the 3-opponent SE treats opponents as independent.
- driver.log `rc=` field records `date`'s status, not the job's; all 69 jsons exist and were checked instead.

## Implications for the post-patch pipeline
- RL recipe = PPO, scalar critic, speed flags. A ~10M-decision budget (~300 it, ~3 h) captures essentially all the gain vs this pool;
  longer runs need a stronger / growing opponent pool (e.g. adding PPO snapshots / league) rather than more iterations.
