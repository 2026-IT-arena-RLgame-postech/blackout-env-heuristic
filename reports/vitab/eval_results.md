# ViT encoder A/B (36 tokens): strength gate (work unit 3, 2026-10-20)

Arms: A = CNN, B1 = `map_encoder=patch4` (4x4/s4), B2 = `map_encoder=patch6` (6x6/s4, pad 1); 2 training seeds each (`s0`, `s1`), 30k steps, same V32.1-labelled DAgger data.
Encoder identity verified on load (NetPolicy.from_checkpoint): A has `graphic_encoder.pre_conv` (CNN); B1 `patch_embed` weight (256,19,4,4); B2 (256,19,6,6).

## Setup
- Tool: `examples/v27_eval.py play` (S2 batch simulator, full matches, cap 425 s, `--stop-exhausted`, both sides, greedy nets on CUDA), launched by `reports/vitab/eval.sh`; tables by `reports/vitab/summarize.py`; raw rows `reports/vitab/sim/matches.jsonl`.
- GPU 0 of poscat_gpu2 (checked empty with nvidia-smi before launch), 8 workers, nice 30. Seeds from 5800000, identical for every student, restricted to cells all six students played.
- Seed counts (reduced from the planned 64 because the shared CPU was loaded at 15-20: ~12 min per 16-game chunk): P5a 64 seeds (x2 sides = 128 games per student); V19 (`strategic_v19`) 32 seeds (64 games); P24c (`p24c_v321`) and P22 (`p22_v31`) 16 seeds (32 games); head-to-head 32 seeds (64 games), training seed 0 pairs only.
- Margin = our score - opponent score, mean +- SE over seeds (two sides averaged first). Arm value = mean of its two student means; SE combines the two students' SEs (does NOT include training-seed variance, which is large, see paired table).

### Per student (margin = our score - opp score, mean +- SE over seeds [sides averaged]; win rate with draws 1/2)

| student | P24c | P22 | P5a | V19 |
|---|---|---|---|---|
| As0 | -79.3+-2.3, win 0.000 (n=32) | -68.8+-4.0, win 0.000 (n=32) | -60.4+-2.6, win 0.062 (n=128) | -12.4+-3.3, win 0.273 (n=64) |
| As1 | -66.1+-3.4, win 0.000 (n=32) | -59.5+-5.8, win 0.031 (n=32) | -50.2+-2.7, win 0.055 (n=128) | -2.9+-3.3, win 0.477 (n=64) |
| B1s0 | -65.6+-4.1, win 0.031 (n=32) | -48.2+-6.4, win 0.094 (n=32) | -42.4+-2.7, win 0.082 (n=128) | -4.8+-4.7, win 0.562 (n=64) |
| B1s1 | -67.3+-3.2, win 0.000 (n=32) | -57.6+-4.8, win 0.031 (n=32) | -50.4+-2.4, win 0.047 (n=128) | -10.1+-3.8, win 0.430 (n=64) |
| B2s0 | -78.8+-2.3, win 0.000 (n=32) | -67.4+-3.5, win 0.000 (n=32) | -49.1+-2.8, win 0.066 (n=128) | +3.0+-2.7, win 0.617 (n=64) |
| B2s1 | -61.2+-5.5, win 0.062 (n=32) | -48.8+-6.9, win 0.125 (n=32) | -36.6+-3.3, win 0.188 (n=128) | +3.2+-3.8, win 0.531 (n=64) |

### Per arm (mean of the 2 training seeds; SE = from the two student means and, in brackets, pooled-seed SE)

| arm | P24c | P22 | P5a | V19 | mean of opps |
|---|---|---|---|---|---|
| A | -72.7+-2.1 | -64.2+-3.5 | -55.3+-1.9 | -7.7+-2.3 | -50.0 |
| B1 | -66.4+-2.6 | -52.9+-4.0 | -46.4+-1.8 | -7.5+-3.0 | -43.3 |
| B2 | -70.0+-3.0 | -58.1+-3.9 | -42.8+-2.1 | +3.1+-2.3 | -42.0 |

### Arm minus A, paired by (opponent, seed): mean over the 2 training seeds, SE over seeds

| arm | P24c | P22 | P5a | V19 |
|---|---|---|---|---|
| B1-A | +6.2+-3.1 (s0 +13.8, s1 -1.2) | +11.3+-4.3 (s0 +20.7, s1 +1.9) | +8.9+-2.3 (s0 +18.0, s1 -0.2) | +0.2+-3.9 (s0 +7.5, s1 -7.1) |
| B2-A | +2.7+-4.0 (s0 +0.6, s1 +4.8) | +6.1+-3.6 (s0 +1.4, s1 +10.8) | +12.5+-2.6 (s0 +11.3, s1 +13.7) | +10.7+-3.2 (s0 +15.4, s1 +6.1) |

### Head to head (row student vs column student; margin from the row student's view, sides averaged)

| match | margin | win rate |
|---|---|---|
| As0 vs B1s0 | -13.1+-2.6 | 0.250 (n=64) |
| As0 vs B2s0 | -5.2+-2.4 | 0.383 (n=64) |
| B1s0 vs B2s0 | -0.4+-3.8 | 0.484 (n=64) |

## Verdict
- Both B arms are at least as strong as A on every opponent; no arm-vs-A difference is negative. Mean over the 4 opponents: A -50.0, B1 -43.3, B2 -42.0.
- Sensitive opponents: P5a B1 +8.9+-2.3, B2 +12.5+-2.6 (both training seeds positive for B2; B1 s1 is ~0). V19 B1 +0.2+-3.9 (s0 +7.5, s1 -7.1: seed noise), B2 +10.7+-3.2.
- Head to head (training seed 0 only): B1 beats A by 13.1+-2.6 (win 0.75), B2 beats A by 5.2+-2.4 (win 0.62), B1 vs B2 even (-0.4+-3.8).
- Caveat: between-training-seed spread of one arm (up to ~15 margin points, e.g. B1-A s0 +18 vs s1 0 on P5a) is comparable to the effects, so with 2 seeds the SEs above (over match seeds) understate the real uncertainty. Held-out accuracy differences (0.81-0.83) were within noise as well.
- Conclusion: B >= A holds (B1 and B2 not weaker than A; B2 clearly better on the sensitive V19/P5a pair, B1 better on P5a/P22). The plan-doc condition for trying the stride-2 variants (121-144 tokens) is MET. B arms also train ~18% faster per step. All students remain 40-70 margin points below P24c/P22, as expected for a from-scratch 30k-step run.
