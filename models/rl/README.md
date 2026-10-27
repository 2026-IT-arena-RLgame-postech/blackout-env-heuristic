# RL policies (tracked copies; checkpoints/ is git-ignored)

| file | what | train script | results |
|---|---|---|---|
| `f1_v33.pt` (+ `_critic.pt`) | **F1, current best RL policy**: PPO from E4 with V33 added to the pool, 20M decisions (gpu2 `checkpoints/rl/f1_v33/final.pt`) | `reports/rl/ppo_f1.sh` | `reports/rl/ppo_ablation_results.md` (F1 section): avg3 +4.1 vs E4, H2H +10.1, V33 +12.4, OOD-26 PASS |
| `e4_beta01.pt` (+ `_critic.pt`) | E4: league-B recipe with β 0.01 from P24c, 20M (gpu2 `checkpoints/rl/abl_beta01/final.pt`) | `ARM=beta01 reports/rl/ppo_ablate.sh` | same report: +4.6 vs B it600, OOD-26 PASS |

Policy files are PPOTrainer checkpoints (`policy_state`; loadable with `NetPolicy.from_checkpoint` / `net:<path>` specs);
`*_critic.pt` = critic state dict (`critic_init=` in PPOConfig).
