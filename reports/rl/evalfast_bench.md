# Fast eval validation (gpu2 GPU1, 2026-10-25)

Candidate ppo_r2_reg/final.pt; seeds 970000.. ; driver `scripts/evalfast_bench.sh` (on gpu2), raw in `evalfast_bench/`.

| run | games | wall |
|---|---|---|
| old `eval_grpo_ckpt.py` (fp32 eager, per opponent and side) vs p24c + V32.1, 32 seeds | 128 | 939 s |
| `eval_fast.py --fp32 --no-graphs`, same games | 128 | 647 s |
| `eval_fast.py` fp16 + graphs, heuristic workers 8 | 128 | 378 s (2.5×) |
| `eval_fast.py`, 256 seeds × {p24c, p22, V32.1}, chunk 256 | 1536 | 982 s (≈11× throughput); peak 8.2 GB |

Results are not game-identical on GPU, even in fp32. Batched forwards of a different batch size use other GEMM kernels, so the fp rounding differs, an argmax flips, and the game diverges. The CPU fp32 equivalence test is exact (tests/test_eval_batched.py).

Identical games out of 64, fast fp32 vs old: p24c 3 and V32.1 62. The fast fp16 run matched 6 and 16 of the old games.

The 256-seed results agree with the old 256-seed eval of the same checkpoint (`ppo_r2_eval.md`, WSL):

| opponent | old | fast |
|---|---|---|
| p24c | +34.4 / 82.3 % | +34.6 / 82.0 % |
| p22 | +42.2 / 85.7 % | +42.6 / 87.9 % |
| V32.1 | +48.1 / 89.5 % | +47.5 / 89.2 % |

Conclusion: use fast eval for the ablations, and re-evaluate the controls with it. Compare only fast with fast.
