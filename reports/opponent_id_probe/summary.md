# P2 opponent-identification probe

Checkpoint `checkpoints/distill/d5n/latest.pt` (frozen trunk); grid every 4 decisions (6.25 Hz) up to 60 s; forward-filled grid points 0.2%. Split by match seed.

## Classes (matches)

| family | all shards | chosen | train | test |
|---|---|---|---|---|
| checkpoint | 4628 | 300 | 233 | 67 |
| v1 | 142 | 142 | 127 | 15 |
| v14 | 139 | 139 | 114 | 25 |
| v15 | 566 | 300 | 250 | 50 |
| v16 | 491 | 300 | 231 | 69 |
| v17 | 1980 | 300 | 256 | 44 |
| v18 | 1143 | 300 | 232 | 68 |
| v19 | 1011 | 300 | 220 | 80 |
| v2 | 242 | 242 | 201 | 41 |
| v4fam | 7320 | 300 | 231 | 69 |
| v5 | 264 | 264 | 210 | 54 |

Chance = 1/11 = 0.091. Test matches still running: 2s: 582, 5s: 582, 10s: 582, 20s: 557, 30s: 466, 45s: 268, 60s: 161

## Accuracy by elapsed match time

| probe | 2 s | 5 s | 10 s | 20 s | 30 s | 45 s | 60 s |
|---|---|---|---|---|---|---|---|
| stateless | 0.352 | 0.418 | 0.402 | 0.406 | 0.427 | 0.433 | 0.404 |
| stateless+avg | 0.363 | 0.502 | 0.557 | 0.628 | 0.637 | 0.646 | 0.634 |
| gru | 0.390 | 0.510 | 0.617 | 0.666 | 0.670 | 0.713 | 0.727 |
| kda | 0.369 | 0.514 | 0.569 | 0.591 | 0.612 | 0.660 | 0.677 |
| tm | 0.340 | 0.543 | 0.579 | 0.654 | 0.637 | 0.713 | 0.658 |
| raw_stateless | 0.225 | 0.297 | 0.340 | 0.339 | 0.333 | 0.351 | 0.348 |
| raw_stateless+avg | 0.258 | 0.321 | 0.383 | 0.456 | 0.472 | 0.526 | 0.571 |
| raw_gru | 0.258 | 0.414 | 0.519 | 0.576 | 0.627 | 0.642 | 0.658 |
| raw_kda | 0.297 | 0.438 | 0.533 | 0.594 | 0.618 | 0.642 | 0.677 |
| majority | 0.137 | 0.137 | 0.137 | 0.138 | 0.144 | 0.164 | 0.199 |

## Macro-F1 by elapsed match time

| probe | 2 s | 5 s | 10 s | 20 s | 30 s | 45 s | 60 s |
|---|---|---|---|---|---|---|---|
| stateless | 0.342 | 0.397 | 0.376 | 0.368 | 0.385 | 0.364 | 0.328 |
| stateless+avg | 0.358 | 0.501 | 0.526 | 0.582 | 0.586 | 0.577 | 0.558 |
| gru | 0.385 | 0.511 | 0.604 | 0.649 | 0.652 | 0.675 | 0.671 |
| kda | 0.351 | 0.495 | 0.544 | 0.566 | 0.573 | 0.602 | 0.599 |
| tm | 0.319 | 0.526 | 0.556 | 0.619 | 0.613 | 0.674 | 0.594 |
| raw_stateless | 0.211 | 0.275 | 0.304 | 0.295 | 0.286 | 0.288 | 0.279 |
| raw_stateless+avg | 0.211 | 0.291 | 0.353 | 0.418 | 0.435 | 0.466 | 0.496 |
| raw_gru | 0.218 | 0.384 | 0.476 | 0.526 | 0.577 | 0.562 | 0.578 |
| raw_kda | 0.279 | 0.430 | 0.507 | 0.556 | 0.576 | 0.577 | 0.606 |

## Coarse accuracy (v4fam/v17/v18/v19/checkpoint/other) by elapsed match time

| probe | 2 s | 5 s | 10 s | 20 s | 30 s | 45 s | 60 s |
|---|---|---|---|---|---|---|---|
| stateless | 0.557 | 0.632 | 0.625 | 0.621 | 0.644 | 0.646 | 0.621 |
| stateless+avg | 0.524 | 0.677 | 0.723 | 0.752 | 0.749 | 0.750 | 0.764 |
| gru | 0.581 | 0.612 | 0.713 | 0.740 | 0.740 | 0.791 | 0.801 |
| kda | 0.540 | 0.644 | 0.694 | 0.711 | 0.738 | 0.772 | 0.826 |
| tm | 0.596 | 0.689 | 0.713 | 0.763 | 0.773 | 0.825 | 0.783 |
| raw_stateless | 0.371 | 0.540 | 0.543 | 0.562 | 0.571 | 0.582 | 0.571 |
| raw_stateless+avg | 0.457 | 0.534 | 0.598 | 0.643 | 0.657 | 0.709 | 0.764 |
| raw_gru | 0.418 | 0.636 | 0.704 | 0.729 | 0.762 | 0.784 | 0.770 |
| raw_kda | 0.464 | 0.615 | 0.677 | 0.718 | 0.732 | 0.735 | 0.764 |

## Confusion (rows = true, cols = predicted), 10 s

**kda**

| true \ pred | checkpoint | v1 | v14 | v15 | v16 | v17 | v18 | v19 | v2 | v4fam | v5 | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| checkpoint | 59 | 0 | 3 | 0 | 0 | 1 | 0 | 2 | 0 | 2 | 0 | 0.88 |
| v1 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0.47 |
| v14 | 0 | 0 | 15 | 0 | 4 | 1 | 0 | 0 | 0 | 5 | 0 | 0.60 |
| v15 | 0 | 0 | 0 | 18 | 11 | 1 | 0 | 0 | 0 | 19 | 1 | 0.36 |
| v16 | 0 | 0 | 0 | 21 | 18 | 2 | 0 | 0 | 3 | 24 | 1 | 0.26 |
| v17 | 2 | 0 | 0 | 0 | 0 | 34 | 0 | 7 | 0 | 1 | 0 | 0.77 |
| v18 | 0 | 0 | 0 | 0 | 0 | 3 | 52 | 13 | 0 | 0 | 0 | 0.76 |
| v19 | 2 | 0 | 0 | 0 | 0 | 8 | 10 | 60 | 0 | 0 | 0 | 0.75 |
| v2 | 2 | 2 | 0 | 7 | 2 | 0 | 3 | 0 | 15 | 10 | 0 | 0.37 |
| v4fam | 2 | 1 | 1 | 7 | 6 | 3 | 0 | 1 | 2 | 41 | 5 | 0.59 |
| v5 | 4 | 8 | 0 | 6 | 0 | 2 | 0 | 1 | 0 | 21 | 12 | 0.22 |

**gru**

| true \ pred | checkpoint | v1 | v14 | v15 | v16 | v17 | v18 | v19 | v2 | v4fam | v5 | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| checkpoint | 56 | 2 | 1 | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 2 | 0.84 |
| v1 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 2 | 0.33 |
| v14 | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| v15 | 0 | 0 | 0 | 26 | 11 | 0 | 0 | 0 | 0 | 13 | 0 | 0.52 |
| v16 | 0 | 0 | 0 | 22 | 28 | 0 | 0 | 0 | 1 | 18 | 0 | 0.41 |
| v17 | 1 | 0 | 0 | 0 | 0 | 37 | 0 | 4 | 0 | 2 | 0 | 0.84 |
| v18 | 0 | 0 | 0 | 0 | 0 | 4 | 57 | 7 | 0 | 0 | 0 | 0.84 |
| v19 | 1 | 0 | 0 | 0 | 0 | 10 | 11 | 58 | 0 | 0 | 0 | 0.72 |
| v2 | 0 | 4 | 0 | 1 | 0 | 0 | 0 | 0 | 13 | 18 | 5 | 0.32 |
| v4fam | 2 | 0 | 0 | 6 | 5 | 2 | 0 | 0 | 4 | 43 | 7 | 0.62 |
| v5 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 7 | 34 | 11 | 0.20 |

Runtime: features 2 s, total 1513 s.
## Reading (2026-09-30)

- Test n = 582 matches (seed-disjoint from train), so accuracy SE is about ±0.02 up to 20 s; 45/60 s use only the
  268/161 matches that last that long (longer games, not the same population).
- **Memory helps, and early.** The frozen-trunk stateless probe plateaus at ~0.40 from 5 s on. The GRU probe reaches
  0.51 at 5 s, 0.62 at 10 s and 0.67 at 20 s. Averaging the stateless log-probs over time (no learned memory) gets
  0.56 at 10 s and 0.63 at 20 s, so most of the gain is plain evidence accumulation, plus ~+0.04–0.08 from a learned
  memory. Most of the identification happens in the first 10 s; after 20 s the curve rises slowly.
- The decision-relevant classes are separable early: v17/v18/v19/checkpoint recall at 10 s is 0.72–0.84 (GRU), and
  their errors are mostly among themselves (v19 ↔ v18, v19 → v17). The rest of the confusion is the weak-heuristic
  blob (v15 ↔ v16 ↔ v4fam, v2/v5 → v4fam), which matters little for counter-play.
- KDA (one residual layer on the 128-d frame code) and TM (the b2 TemporalMemory block on the 11 trunk tokens) are
  within noise of the GRU or slightly below it (the TM probe had fewer sequence samples: 1000 × 16 against 1500 × 32).
  At this horizon the probe does not show an advantage for KDA over the GRU.
- Trunk features matter for a single frame (stateless 0.40 against raw 0.34; at 2 s, 0.35 against 0.23). With memory
  the gap shrinks to ~0.05–0.09 at 5–20 s and disappears by 30 s: the history of raw positions/classes carries most
  of the identity.
- The run used `--kda-solve solve`. The MPS default (Neumann `doubling`) gave NaN in the TM probe (see the
  `use_kda_solver` docstring in the script).
