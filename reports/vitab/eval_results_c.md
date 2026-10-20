# ViT A/B/C strength eval: 36-token (B) vs 144-token (C) encoders (2026-10-20)

Students: A cnn, B1 patch4 (36 tok), B2 patch6 (36 tok), C1 patch2 (144 tok), C2 patch4s2 (144 tok), each 2 training seeds
(s0, s1); C3 patch3s2 s0 only. All trained on the same V32.1 DAgger data from scratch, 30k steps.
Tool: `examples/eval_grpo_ckpt.py` (greedy NetPolicy, full matches, both sides, horizon 11000; per-game margins), stride-2 code from
worktree `~/blackout-env-vitabc` (c-arms). Seeds 960000.. (64-seed chunks, identical for every student), GPUs 0 and 1 only (4 procs each,
8 total, nice 30, threads capped at 2). Scheduler/scripts in `reports/vitab/eval_c/`, raw per-game margins in `reports/vitab/eval_c/raw/`,
full tables in `reports/vitab/eval_c/analysis_output.txt`.

Seeds actually run (the requested 256 was not affordable: a 64-seed chunk = 128 games took 14-25 min per process, C arms slowest, ~0.3 jobs/min over 8 procs):
P5a and V19 128 seeds x 2 sides; P22 and P24c 64 seeds x 2 sides; head-to-head 128 seeds x 2 sides. Wall time ~3.2 h.
Arm number = mean of the two training seeds' games (game-level SE of the seed-averaged margin). Win counts draws as 1/2.
V19 is NOT saturated for these from-scratch students (win 0.35-0.55) and is the most discriminating opponent; P5a is second; P22/P24c
beat every student by 55-75 so they only give a coarse ordering.

## Per arm (win / margin +- SE)

| Opponent (seeds) | A | B1 | B2 | C1 | C2 |
|---|---|---|---|---|---|
| V19 (128) | 0.397 / -6.2 +- 1.1 | 0.507 / -1.0 +- 1.5 | **0.546 / +4.1 +- 1.1** | 0.347 / -10.9 +- 1.3 | 0.448 / -3.1 +- 1.3 |
| P5a (128) | 0.060 / -52.9 +- 1.3 | 0.094 / -47.4 +- 1.6 | **0.106 / -45.0 +- 1.5** | 0.061 / -53.8 +- 1.4 | 0.097 / -48.0 +- 1.5 |
| P22 (64) | 0.033 / -63.4 +- 1.7 | 0.027 / -61.8 +- 1.8 | **0.055 / -56.7 +- 2.1** | 0.035 / -68.7 +- 1.6 | 0.035 / -63.2 +- 1.8 |
| P24c (64) | 0.008 / -72.3 +- 1.3 | 0.027 / -64.1 +- 1.8 | 0.031 / -66.4 +- 1.7 | 0.016 / -73.4 +- 1.4 | **0.041 / -62.0 +- 1.9** |

C3 (patch3s2, s0 only, 64 seeds): V19 0.352 / -9.1 +- 2.8; P5a 0.043 / -65.5 +- 2.5; P22 0.016 / -67.8; P24c 0.008 / -75.7. Worst of all arms.

## Paired diffs C - B (same seeds/sides, seed-averaged margin; dMargin +- SE)

| Opponent | C1-B1 | C2-B2 | C(C1,C2) - B(B1,B2) |
|---|---|---|---|
| V19 | -9.9 +- 1.8 (dWin -0.160) | -7.2 +- 1.5 (dWin -0.098) | **-8.5 +- 1.1** |
| P5a | -6.3 +- 1.9 | -3.0 +- 1.9 | -4.7 +- 1.3 |
| P22 | -6.9 +- 1.8 | -6.6 +- 2.4 | -6.7 +- 1.5 |
| P24c | -9.3 +- 2.2 | +4.3 +- 2.3 | -2.5 +- 1.6 |

Also B-A (anchor): V19 B1 +5.2, B2 +10.3; P5a +5.5 / +7.9; P22 +1.6 / +6.8; P24c +8.3 / +5.9 (all positive).
Best C vs best B on V19: C2-B2 -7.2 +- 1.5; on P5a C2-B2 -3.0 +- 1.9 (within noise).

## Head-to-head (C as "ours" vs B, 128 seeds x 2 sides; the C net is loaded fp32, the B opponent fp16: mirror gave 0.493 earlier)

| Pair | C win | C margin +- SE |
|---|---|---|
| C2_s0 vs B2_s0 | 0.566 | +4.8 +- 1.6 |
| C2_s1 vs B2_s1 | 0.338 | -9.8 +- 1.7 |
| C1_s0 vs B1_s0 | 0.316 | -12.6 +- 1.6 |
| C1_s1 vs B1_s1 | 0.564 | +1.1 +- 1.7 |

Mean over the four pairs: C margin -4.1. C2 vs B2 mean -2.5, C1 vs B1 mean -5.7. The sign flips with the training seed.

## Training-seed spread

Between the two training seeds of one arm (margin, s0 - s1), per game SE ~2-3: V19 A -10, B1 +3, B2 -6, C1 -12, C2 +5;
P5a A -13, B1 -1, B2 -4, C1 -20, C2 +3. So the typical training-seed spread is 5-15 margin points per arm (up to 20 for C1 vs P5a,
and in the head-to-head the same-seed pairs swing by ~15 points between seeds), the same size as the 15 quoted before. The arm-mean of two seeds
carries ~half of that (3-7). The C-B differences (-3 to -10 in arm means) are statistically significant on game noise (paired SE 1-2.5) but are of
the order of the seed spread, so "C is worse" is not strongly established; "C is not better" is.

## Verdict

**Use the 36-token B encoder.** The 144-token encoders are not stronger: C1 and C2 are below B on V19 (-10 / -7 margin, -16 / -10 win pts), below or
equal on P5a and P22, mixed on P24c (C2 +4, C1 -9), and C-B head-to-head is -4 on average with sign flipping per training seed. C1 (patch2) is no better
than the CNN anchor A, and C3 is the worst. Training cost (~1.7x) and forward cost (~3x) buy nothing here.

Between B variants, **B2 (patch6) ahead of B1 (patch4)** on V19 (+5.1 margin, +0.04 win), P5a (+2.4), P22 (+5.1); B1 only edges P24c (+2.3, within noise).
B2 is the only arm with a positive margin vs V19 and is the same pick as the earlier eval. Caveat: B1-vs-B2 gaps (2-5 points) are under the training-seed
spread (B2 s0-s1 spread up to 6 on V19, 9-13 on P22/P24c), so B2 over B1 is a lean, not proven.

Not done / limits: 256 seeds for P5a/V19 and 128 for P22/P24c were not affordable in the ~3 h budget (done: 128 and 64).
