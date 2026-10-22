# PPO r2: regression vs HL-Gauss critic (setup)

A/B on poscat_gpu2: identical except the critic head/loss. Launcher `reports/rl/ppo_r2.sh` (`ARM=reg|hlg GPU=<0|2> MAXDEC=33327244`),
flags for both arms: `update_fp16=true critic_fp16=true update_compile=true critic_warmup=10`, r1 pool/weights/pfsp=hard/games=256/
segment=128, eval every 150 iterations (checkpoint saved each eval), same init (P24c) and critic init (S2 value net). Code branch `feat/ppo-dist`.

## Design (`PPOConfig`, defaults = old behaviour, bit-identical)
* `critic_head="reg"`: V = (P(win)-P(loss)) + lam*tanh(final/scale) from the S2 heads, Huber loss (unchanged).
* `critic_head="hlgauss"` (Farebrother et al. 2024): a new `Linear(hidden, critic_bins)` (ValueNet's optional `bin_out`, fed by the
  head's hidden features; bin centres buffer ours) gives logits of a pmf over the *return*. Loss = CE(softmax, HL-Gauss(ret)) where the
  target is the Gaussian mass (sigma = `critic_sigma`=0.75 bin widths) per bin, renormalised; `log_softmax` in fp32 (works with
  `critic_fp16`, and with the compiled critic forward `_HLLogits`). V = sum(softmax * centers), used by bootstrap and GAE.
  Defaults: 101 bins on [-1.4, 1.4] (returns lie in ~[-1.2, 1.2]). The whole critic (trunk, head, bin head) trains with the same
  `critic_lr`, grad clip, AdamW as the reg arm.
* Init / pre-fit: in the first `collect()` (before that batch's V/GAE), the fresh bin head (everything else frozen) is fitted for
  `critic_prefit_epochs`=4 epochs (Adam lr 1e-3, minibatch 1024) with HL-Gauss CE to the S2 heads' scalar V of that batch's 32.8k views.
  So the bin head starts as a calibrated copy of the original V, iteration 0's GAE is meaningful, and the pre-fit costs a few seconds.
  Then `critic_warmup`=10 critic-only iterations (both arms) before any policy update.
* Diagnostics logged per iteration (both arms): `vf_loss` (Huber for reg, CE for hlg: not comparable), `expl_var` = 1 - Var(ret-V)/Var(ret)
  (V before the iteration's critic update, so it is out-of-sample w.r.t. the update but ret is GAE-bootstrapped from the same V, hence
  optimistic ~0.99 for both arms: compare arms, not absolutes), `abs_adv` (mean |adv| before normalisation), `ret_mean/ret_std`, `v_mean`,
  `pmf_entropy` (hlg; nats, ln101 = 4.6), `prefit_*` (iteration 1 only).
* Critic saved as `<ckpt>_critic.pt` (includes `bin_out.*`); policy checkpoints/eval untouched.
* Tests: `tests/test_grpo.py` 12 passed (new: `test_ppo_hlgauss_critic_cpu`, `test_ppo_reg_critic_diagnostics_cpu`).

## GPU smoke (200k decisions, `critic_warmup=2`, 7 iterations, GPUs 2 (hlg) / 0 (reg), loaded server)
| | s/iter after warm-up (wall) | trunk | leaf | notes |
|---|---|---|---|---|
| hlg | ~21-23 (update iterations) | 8-11 s | 1.4 s | iter 1 paid compile + graphs (~2 min); GPU mem ~0.5-2.4 GB; no NaN |
| reg | ~21 | 8 s | 1.4 s | same |

hlg pre-fit: CE 4.97 -> 2.94 (floor ~1.2 for sigma 0.75 bins), V rmse 0.104 vs S2-V std 0.53. Critic-only iterations: pmf entropy
3.3 -> 1.5, CE 3.3 -> 1.5, expl_var 0.58 -> 0.99 (iter 1 -> 6). Reg: vf_loss 0.157 -> 0.011, expl_var 0.54 -> 0.99. Policy updates
(pg/kl finite, kl ~1e-3) ran in both arms. The "EOFError" tracebacks at exit are heuristic-worker teardown, as in earlier runs.

## Full runs
`checkpoints/rl/ppo_r2_reg` (GPU 0) and `checkpoints/rl/ppo_r2_hlg` (GPU 2) in the server clone `~/blackout-env-ppodist`
(symlinked .venv/checkpoints of the mirror), `log.jsonl` / `run.log` in each; MAXDEC=33327244 (= r1/r3b budget, ~1017 iterations).
