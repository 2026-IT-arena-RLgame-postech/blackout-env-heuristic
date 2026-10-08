---
name: project-distill-next-steps
description: "Distillation status as of 2026-09-30 and what's queued next — D4 (v21-teach) then an OOD test; read before resuming distillation work"
metadata:
  node_type: memory
  type: project
  originSessionId: ae7bdc3d-fc33-4539-a05d-ed299772337d
  modified: 2026-09-30T01:22:25.181Z
---

State (2026-09-30), branch `feat/v17-distill` in blackout-env:
- Distillation pipeline `blackout_env/train/distill/` (async DAgger). D4 (v21-teach) held-out vs V4/V17/V19: +44.8/+25.6/+16.3 (first student to beat V19; D4 − D3b paired: −15.6/−8.5/+38.1). D3b was +60.4/+34.2/−21.8.
- Teachers: v17-teach, v20-teach (spawn-camp counterplay), and **v21-teach** (a Collector transforms into a Hunter only when there's a route clear of enemy Hunters). v21 itself vs V19 scores +3.4, against −38.9 for v17-teach.

**D4 launched 2026-09-30** (pids 67898 train, 67899 held-out waiter; log checkpoints/distill/d4/stdout.log, held-out → reports/distill_heldout/stdout_d4.log). It was started with `checkpoints/distill/launch_d4.sh` (D3b settings + v21-teach; Run 16 checkpoints held out of the opponent pool). After D4, run the OOD test designed in `docs/distill_plan_ko.md` §6. The test compares student vs teacher on unseen checkpoints, V20/V21, out-of-range perturbations, random/idle opponents, and mirror matches. The code is done (commit 70f837b, `--opponents ood`). A waiter (`launch_ood_d4.sh`, pid 70661) runs it on D4 and D3b latest after the D4 held-out eval, writing to reports/distill_ood/.

If OOD handling falls short, the user's idea: generate plausible synthetic states and label them with the memoryless teacher. Agreed order: structured perturbation of real shard states first, a VAE or generative model only if that's not enough; plan §6.

**Why:** the user wants to check that the distilled student handles OOD situations, not just the held-out seeds against known opponents.

**How to apply:** D4 is running; report progress in Korean. Long runs go detached on AC ([[project-long-runs-detached]]). Judge only on the held-out eval (24 games per opponent, paired differences). The in-training 3-seed eval is too noisy, and Unity is non-deterministic even on a fixed seed.

**OOD test done (2026-09-30, reports/distill_ood/):** no axis widened the student − teacher gap. D4 even beat the v21 teacher head-to-head (+16.0±9.6). So synthetic-state augmentation is not needed now. The remaining weakness is flicker: students reverse 3–30× more often than the teacher, and they die more against checkpoints.

**2026-10-01 status.**
- **D6 done:** STAND-free, wide opponent pool. Held-out vs V4/V17/V19: +60.3/+32.6/+16.2, best so far.
- **P1 chain running** in worktree blackout-env-p1: `act:strategic_v19` teacher, stateless vs temporal student.
  - Output: reports/p1_heldout/stdout.log; the chain prints "P1 CHAIN DONE" when finished.
  - The stateless arm ended at agree 0.62; the temporal arm's mid-training agree is about 0.72.
  - Judge the two arms on the held-out paired comparison.
- **Next:** P4, a K-bin + STAND policy head with any-angle teacher labels.

P1 result (2026-10-01, reports/p1_heldout): act:strategic_v19 teacher, 20k steps, init d5n+STAND. Temporal − stateless paired: v4 +23.8±10.4, v17 −0.8±12.5, v21 +31.8±10.9, all +18.3±6.6 (72 pairs). Agree T 0.65–0.73 vs S 0.55–0.71; reversals/1k T 146–213 vs S 350–461. Both still far below the V19 teacher itself (T−V19: v17 −67, v4 −19, v21 −7) → memory helps but act-label distillation of V19 is undertrained. Next: decide P4 / longer P1.

Backlog (user-approved 2026-10-01, run later):
- Any-angle V21 target selection: replace octile distance with a Euclidean/any-angle distance field so the teacher itself gets stronger (travel is already ~optimal; selection is still 8-way).
- Add D6 (and perturbed D6) to the opponent pool for robustness vs strong neural opponents.
Running 2026-10-01: D7 chain (d7s/d7t/d7a from D6, launch_d7.sh, p1 worktree) → P4 K-bin head (subagent, main checkout, launch_p4.sh waits for 'D7 CHAIN DONE'); P3 hypothesis check = best-response matrix reports/p3_matrix (no training). Old v1–v19 tournament: v17 best single, only v18/v19 counter it (CV oracle gain +2.9 avg, +48 vs v18).
P3 matrix done 2026-10-01 (reports/p3_matrix/margin_matrix.json): best single = strategic_v21_anyangle (avg +51 vs D6 +41); CV best-response gain ≤0 everywhere; only real counter V19 vs V17 (+51 vs +27.5). Opponent-conditional teacher shelved.
2026-10-02: P4 done. p4a (K=32, act:strategic_v21_anyangle, from D6) = NEW BEST: heldout −D6 +14.8±4.3, OOD +15.2±3.1, beats its teacher (V17 +18.6, V21 +21.8). p4s (K=32 v21-teach) ≈ D6. arc_mean inference no gain. D7 (all --init continuations) regressed vs D6 (d7s −10.4) → D8 diagnoses continuation; queued after P5a.
2026-10-02: P5a (logits head, K=32, anyangle teacher) ≈ p4a (+3.8±4.1 heldout, +1.9 OOD) → IQN removal free; p5a−D6 +18.5 heldout/+17.1 OOD. Default going forward: logits+K32+anyangle. D8 running next.
2026-10-02: D8 done — continuation regression = LR 3e-4 too high after convergence (d8l lr1e-4 −D6 +0.8, vs d7s +11.2; ring copy d8w no help; true resume d8c −4.7). Use --lr 1e-4 (or decay) for all --init fine-tunes; p4a/p5a gains likely underestimated.
2026-10-02: P5b (lr 1e-4) ≈ P5a. P6 anyangle_ed +3.4±2.9 (n.s., branch feat/anyangle-targets ef28511, not merged). D9 launched from worktree blackout-env-d9: p5a init, cosine 3e-4→3e-5, students (frozen d6/p5a in checkpoints/distill/frozen) in pool, 30k.
2026-10-02: D9 done — vs heuristics d9−p5a +5.9±3.5 (n.s.), OOD +0.1; head-to-head D9 vs P5a −17.2±11 (teacher anyangle vs P5a −29.5). Student opponents hit teacher ceiling: distillation can't learn to beat opponents the teacher loses to. Beyond this needs a stronger teacher (search/best-response) or RL.
2026-10-02: P9 strategic_v22 (branch feat/teacher-v2 fc7dcb2): +42.8±4.1 vs anyangle teacher, beats P5a +54. Key = opening race to centre sanctuary (don't pick batteries while racing, sentry, tie-freeze fix). Size items no effect. P10 (distil act:strategic_v22 from p5a, worktree blackout-env-p10) queued after P7P8.
2026-10-02: S2 simulator done (branch feat/sim 6a0012a, blackout_env/sim/): bit-exact replay of 60/60 Unity matches incl. RNG; mid-match init 5s 99.3–99.8%; 4.6µs/decision; policies are the rollout bottleneck. Found env bug: early 100-pt finish reports pre-final-deposit score (winner may be wrong).
2026-10-02: merged feat/teacher-v2, feat/anyangle-targets, feat/sim + env terminal-score fix (41caca7) into feat/policy-v2 (ea1f505); full suite 432 passed. Unity fix committed (blackout 0ed01c6, perf/mapobs-on-decision) build/mac rebuilt 2026-10-02 with the fix (backup build/mac_pre_scorefix_20261002); smoke: early wins report 100+.
2026-10-02: P7t (KDA on P5a recipe) −6.7±3.9 heldout, P8k (K=64) −2.9±4.3 → no gain; keep stateless K=32.
2026-10-02: P10 (P5a recipe + act:strategic_v22, from p5a) = NEW BEST: −p5a heldout +12.0±4.2, V22 set +18.0, OOD +12.0±2.7, head-to-head +31.6±11.5 (win 0.79). Frozen copy checkpoints/distill/frozen/p10_v22.pt. Student ≈ teacher on most opponents but −47.9 vs anyangle V21 (teacher +60.6, student +12.8) — imitation gap in sanctuary race. S1/S3 comparisons should now use P10.
2026-10-02: P10 loses to V22 head-to-head −38.6: opening sanctuary race is decided by single 0.04 s ticks; quantizing the teacher's own moves to 32/64 bins kills all ties (sim) → needs ~2° precision → P11 = bin + continuous angle head, queued after S1 (user approved). P12 strategic_v23 (merged into feat/policy-v2 b161412): exact any-angle field + zero wall margin only while racing → race never lost; V23−V22 +8.5±2.5, vs P10 +76.6. Use V23 as P11 teacher.
2026-10-02: P11 (P10 + continuous-angle head, teacher act:strategic_v23, branch feat/p11-angle f66bb99/a8210e1 + docs, merged into feat/policy-v2 6a6ff66) = NEW BEST, frozen checkpoints/distill/frozen/p11_v22.pt: −P10 V23-set +39.4±4.2, heldout +11.2, OOD +7.0; vs V23 head-to-head +4.5±6.8 (race ties 23/24). Student now ≈ best teacher; next gains need search (S3) or better teacher.
2026-10-03 P13 strategic_v24 (merged): late-game relief/reenter/base_exit rules fix pinned-carrier freeze (14% of late roots, +23.8 there), but full-match gain ≈0 (sim +0.6±0.4, Unity +0.1±1.7). Keep V23 as teacher; late game has little headroom — matches decided before 30 s.

- 2026-10-03 r5 (RolloutNet attn, 0.67M) vs P11 (5.81M): r5 is 3–9 pts weaker (sim 1000 g/opp, Unity 400): vs V19 −8.9 sim / −7.4 Unity, vs V23 ~0, h2h −2..−4. Loss is post-race 10–30 s scoring + late drift, not the race. Shrinking needs r5-driven sim DAgger (+ maybe d256×3 ~2M). Merged c1def6f.
- 2026-10-04 rollout-net arch (§22, merged dfd33ec; ckpts checkpoints/rollout_net/arch/): r5's gap was the missing mid-range map view. f3 6×6 map tokens as cross-attn K/V: `f3_x_d2` (0.82M, 1.11× rollout cost) vs P11 V19 −0.7 / P5a −2.8 / V23 +3.9; best `d256_f3_x32_d2` (1.83M, 31% of P11, 1.27×) +0.1 / −2.2 / +3.8 (sim). Local windows/fovea/ViT patches/self-attn worse; DAgger round 2 ≈ +1. Q-head distillation of search ≈ r6f (~20% of search gain) → stopped; use search as the player.
- 2026-10-04 **P14b = NEW BEST student** (`checkpoints/distill/frozen/p14_v25.pt`, merged 2cf7c51, §29): P11 recipe + teacher `act:strategic_v25` (V23 + exact race walk + race_commit, §28; Unity-confirmed +73.5 vs V23 / +72.4 vs P11), trained on wsl_gpu with `--max-replay-ratio 2.3` (replay ratio 8.6 → P14a was −25 vs V25). Sim paired −P11: V23 +72.5, V19 +7.5, P5a +0.4, V25 +74.3; vs V25 −0.4±4.0 (teacher parity); h2h vs P11 +75.2; arrives at decision 99 in 1536/1536. Race is now symmetric 99 vs 99 with V25-like opponents. Next: Unity check of P14b; re-distil rollout/opponent nets (r5/f3_x_d2) from V25 for search.
