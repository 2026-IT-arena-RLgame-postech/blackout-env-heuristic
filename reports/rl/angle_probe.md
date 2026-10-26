# Angle-precision probe (E4, 2026-10-26)

Question: is the 32-bin + deterministic-offset direction head a precision bottleneck (heuristic dev found ~128 directions
needed vs hunter rush)?  E4 (checkpoints/rl/abl_beta01/final.pt) played with its move angle
- q0: unchanged (bin + distilled offset), q-1: bin centres (offset ignored, 32 dirs),
- q32 / q64 / q128: final angle rounded to 32 / 64 / 128 directions (`examples/eval_fast.py --angle-quant`, candidate only).

256 seeds (970000–970255) × 2 sides, fast eval; diffs paired by seed and side (margin, ±SE).
Script `reports/rl/angle_probe.sh`, analysis `reports/rl/angle_probe_analysis.py`, raw `reports/rl/angle_probe/`.

```
opp                      q0       q-1-q0       q32-q0       q64-q0      q128-q0
V33                   -12.1   -2.3± 1.9   -4.0± 1.9   +0.1± 1.7   +0.4± 1.6
early_rush            +82.8   +0.4± 0.4   -0.3± 0.5   +0.4± 0.4   -0.0± 0.4
v30_h4hunt            +46.4   -0.1± 1.8   -0.9± 1.8   -1.6± 1.6   -1.1± 1.5
V32.1                 +51.5  -15.3± 2.1   -5.2± 1.8   -1.3± 1.5   -2.4± 1.4
v30_hunt              +52.7   +0.1± 1.8   -1.9± 1.6   +0.4± 1.5   +0.9± 1.4
p22_v31               +47.6   -1.0± 1.9   -1.8± 2.0   -0.3± 1.8   +0.5± 1.7
p24c_v321             +45.0  -13.8± 2.0  -11.4± 1.9   -1.8± 1.7   -0.2± 1.7
v30_reckless          +57.7   -2.2± 1.9   -2.4± 1.9   -4.0± 1.9   +0.7± 1.6
rush_all              +96.7   +0.2± 0.4   -0.9± 0.5   -0.1± 0.4   +0.4± 0.4
mean                  +52.0   -3.8± 0.6   -3.2± 0.6   -0.9± 0.5   -0.1± 0.4
```

## Reading
- 128 directions lose nothing (−0.1±0.4); 64 lose little (−0.9±0.5); 32 directions lose −3.2 (bin centres −3.8).
- The loss at 32 directions is concentrated vs the **net opponents and the main heuristics** (p24c −11 to −14, V32.1 −5 to −15, V33 −2 to −4).
  It is **not** concentrated on hunt-style OOD (h4hunt, hunt, early_rush, rush_all ≈ 0).
- So the angle precision the policy already has (offset head, ≈ continuous) matters, but ~128 directions are enough. The current head
  already delivers that precision (q0 ≈ q128).
- A finer head (1024 bins / continuous) would add no precision at inference. The only possible gain is RL exploring at the
  sub-bin level, and the offset / fine combos (−20.6 / −3.2) gave no sign of that.
  → Re-distilling with a 1024-bin head is **not** a priority.
