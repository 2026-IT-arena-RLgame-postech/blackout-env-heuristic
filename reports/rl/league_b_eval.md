# League arm B — evaluation (2026-10-25)

Candidate: `checkpoints/rl/league_b/final.pt` (PPO + league arm B: per-opponent advantage centering, gated
snapshots, ramp to 20 %, anchors ≥ 40 %; 1042 it, 33.3M decisions, 14.9 h on WSL 3060; setup in
`reports/rl/ppo_league_b_setup.md`). Baseline A = `ppo_r2_reg/final.pt` (best PPO so far, `ppo_r2_eval.md`).

## Phase 1 — 256 seeds × 2 sides (512 games per opponent)

Driver `league_b_eval/wsl_eval.sh`, raw `league_b_eval/league_b__<opp>__<lo>.json`, seeds 970000–970255.
B − A pairs are exact (same seeds, same WSL GPU). Pairs against r1 / r3b / p24c are weaker because those
runs were evaluated on gpu2.

| opponent | B margin / win | B − A margin | B − A win pp | B − r1 | B − r3b | B − p24c |
|---|---|---|---|---|---|---|
| p24c (pool) | +40.7 / 87.6 % | **+6.4 ± 2.1** | **+5.3 ± 2.0** | +6.5 ± 2.2 | +13.8 ± 2.3 | +40.1 ± 2.8 |
| p22 (pool) | +44.0 / 85.1 % | +1.8 ± 1.9 | −0.7 ± 1.9 | +2.0 ± 1.9 | +11.2 ± 2.1 | +22.7 ± 2.5 |
| V32.1 (pool) | +49.9 / 90.4 % | +1.8 ± 2.0 | +1.0 ± 1.7 | +9.2 ± 2.1 | +6.8 ± 2.1 | +24.6 ± 2.4 |
| p5a (held out) | +68.3 / 99.1 % | +0.8 ± 0.9 | +0.1 ± 0.6 | +1.5 ± 1.0 | +3.9 ± 0.9 | +4.8 ± 1.1 |
| p16 (held out) | +72.5 / 99.4 % | +0.0 ± 0.8 | +0.2 ± 0.5 | +0.5 ± 0.8 | +2.8 ± 1.0 | +3.6 ± 1.0 |

Head-to-head B vs A, 512 games: **+3.0 ± 1.7 margin, B wins 53.9 %** (0.4 % draws; side 0 +1.8, side 1 +4.2).

## Phase 2 — OOD-26 suite (25 opponents; run16_final checkpoint unavailable)

32 seeds × 2 sides from 2600000, cap 425 s, CPU. The run was split across machines: opponents 0–15 ran on WSL
and 16–24 on gpu2 CPU, with the same code copy and checkpoint md5s. The two halves were merged with no duplicates
into 6400 rows (`league_b_eval/ood26/matches.jsonl`). Report: `league_b_eval/ood26_report.txt`.

| candidate | mean margin (25 opps) | worst-opp margin | min win | max cat rate (margin < −20) | mean cat | §56 rule vs p24c |
|---|---|---|---|---|---|---|
| **B final** | **+69.2** | **+40.7** (v30_c0) | **0.73** (v30_c0) | **0.23** (v30_c0) | **0.034** | PASS |
| B iter 600 | +68.5 | +37.9 (v30_c0) | 0.69 | 0.27 | 0.035 | PASS |
| A ppo_r2_reg | +69.0 | +22.5 (v30_c0) | 0.58 | 0.41 | 0.038 | **FAIL** (v30_c0 −18.4 ± 6.8 vs p24c) |
| p24c | +64.1 | +30.7 (v30_h4hunt) | 0.72 | 0.25 | 0.064 | base |

Paired B − A per opponent:
- B is better on v30_c0, which is A's catastrophic weakness: **+18.2 ± 6.5**, win 0.58 → 0.73. B is also better on v30_hunt (+5.6 ± 3.6) and v30_reckless (+6.9 ± 4.7).
- B is slightly worse on v21 (−7.9 ± 3.7, win 1.00 → 0.95), v26 (−5.9 ± 3.3, 1.00 → 0.97) and v31_eps10 (−5.5 ± 3.2, 1.00 → 0.97). A crushes all three, so these are 2–3 extra losses out of 64 each. By the weak-opponent win-rate rule this is minor, but worth watching.
- Every other opponent is within about ±2.5.

## Verdict

- In distribution, B ≥ A everywhere and is clearly better only vs p24c (+6.4 ± 2.1). The head-to-head is a weak +3.0 ± 1.7, about 1.8 SE.
- On OOD worst case, B is clearly better: it fixes A's v30_c0 collapse (A fails the §56 rule, B passes) and has the best worst-opponent margin of all candidates. B's mean OOD margin equals A's.
- **Recommendation: make the B recipe the RL default.** Before relying on the small in-distribution gain, a second seed would confirm it is not a lucky run. Arm C (B + slow-EMA opponent) can follow.
