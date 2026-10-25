# PPO ablation plan (2026-10-25)

All arms use the league-B recipe (`reports/rl/ppo_league_b.sh`) with one change each, run via `reports/rl/ppo_ablate.sh`. No arm runs to 33M decisions any more.

## Control and noise floor
- **C0 = league_b checkpoints at the same budget.** `iter_300` is about 9.8M decisions and `iter_600` about 19.7M.
- **C1 = league_b2.** A second seed of the same recipe, stopped at it 600 on gpu2 GPU0.
- **Noise floor.** The B vs B2 difference at it 300 and it 600 is the run-to-run spread. An arm "wins" only if it beats C0 by more than this spread **and** by more than 2 SE.

## Arms

| id | change (vs control) | budget | compare with | hypothesis | expected |
|---|---|---|---|---|---|
| E1 lam99 | gae_lambda 0.95→0.99, segment 128→256, games 256→128 (same 32k rows/it) | 10M | B it300 | Advantage horizon 0.8 s → about 4 s, so less reliance on local critic bias and better credit for multi-second events (chases, captures) | +2–5 margin vs p24c/p22; adv std ↑, EV slightly ↓ |
| E2 lam995 | λ 0.995, segment 512, games 64 | 10M | B it300, E1 | Even longer horizon; shows the trend | Below or equal to E1 (variance, fewer games ⇒ slower PFSP/ramp); slower it/s |
| E3 refreset | KL anchor π_ref ← learner at each snapshot (`league_ref_reset`) | 20M | B it600 | KL(π‖P24c) saturates at 0.022–0.024 from it ~300, the same point as the performance plateau, so the anchor is binding | KL to P24c keeps growing; margin keeps rising after it 300; risk: worse OOD worst case |
| E4 beta01 | β 0.02→0.01 | 20M | B it600, E3 | Same claim via a weaker anchor. E3 vs E4 separates "anchor strength" from "anchor location" | If E3 and E4 both gain: KL-limited. If neither gains: plateau comes from elsewhere (opponents, critic) |
| E5 offset_rl | angle offset sampled N(μ, σ) with learnable σ; its log-prob is in the PPO ratio and KL (code in progress) | 10M → 20M if promising | B it300 | 32 bins (11.25°) plus a deterministic distilled offset cap aim precision (heuristics needed about 128 directions vs hunter rush) | Gains concentrated vs hunt-style opponents (v30_h4hunt, v30_hunt, early_rush) and p24c; σ shrinks over training |
| (later) | 32 × 32 hierarchical bins if E5 fails to learn; per-unit advantage (GPAE / counterfactual) | – | – | – | Only after E1–E5 |

## Evaluation per arm
- **Phase A (GPU):** 256 seeds × 2 sides vs p24c, p22 and V32.1, plus head-to-head vs the control checkpoint. Pairing is exact by seed.
- **Phase B (CPU):** the OOD-26 suite, only for arms that pass phase A. The run is split over WSL and gpu2 CPU, about 1 h. Gate: OOD worst case no worse than control −5.
- **Combination:** winners are combined into one longer run.

## Order and time
- **Training speed.** gpu2 runs about 16 s/it, so 10M decisions take about 1.4 h and 20M about 2.8 h. WSL runs about 51 s/it, so 10M take about 4.3 h. E2 is expected to be slower per decision (fewer parallel games).
- **Phase A eval:** about 2 h on WSL, less on gpu2.
- **Queue.** A conservative free-GPU watcher runs (two checks 10 min apart). WSL is in use by another job and gpu2 GPU2 by another user. GPU3 is never used.

| step | where | what | about |
|---|---|---|---|
| 1 | gpu2 GPU0 | B2 to it 600, then stop | until ~22:30 KST 10-25 |
| 2 | gpu2 GPU0 | E1 lam99 | 1.5 h |
| 3 | gpu2 GPU0 | E3 refreset | 3 h |
| 4 | gpu2 GPU0 | E4 beta01 | 3 h |
| 5 | gpu2 GPU0 | E2 lam995 | 2–3 h |
| 6 | gpu2 GPU0 | E5 offset_rl (after code + smoke) | 1.5–3 h |
| ∥ | WSL / gpu2 GPU2 when free | take the next queued arm; run phase-A evals | – |

**Total:** about 12–14 h of training plus evals, so roughly 1–1.5 days if a second GPU frees up.
