---
name: project-inflight-2026-10-06
description: "In-flight agents as of 2026-10-06 ~14:00 KST (Q12/V31 on gpu2+gpu1, P20 on wsl) and champions — read first after compaction"
metadata:
  type: project
---

feat/policy-v2 HEAD 5dcaf89 (§45 P19). Usage budget no longer binding.

Champions: student **P17** (p17_v28.pt); teacher **V30** `strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true` (§44, Unity h2h +15.9 vs V29, beats P17 +38.5); search = P17 + o4fmean 5 s + rollout v28d256 + leaf V s2v_f3xd2/iter2 (+17.6, §40).
Rejected students: P18 (§42), P18b/P18bp2 (§43; candidate: h2h +39.4 vs P17 but V28 −10.8), P19 (§45: V30 from P18bp2 20k; h2h −0.7, V28 −20.7). Pattern: V29/V30 students lose 10–21 vs V28.
Frozen: Mac checkpoints/distill/frozen/{p18_v29,p18bp2_v29,p19_v30}.pt; wsl ~/p14frozen/{p18,p18bp2,p19}.pt.

Agents (new session cf90b6b9; old agent IDs dead):
- **Q12** agent a0266a6d2d2d0eb20, branch feat/v31-rule (worktree .claude/worktrees/v31-rule), gpu2+gpu1 clones ~/blackout-env-q12: V31 on V30 (Carrier evasion / risk_weight / exact_evasion params) → §46.
- **P20** agent a9c7ce2e1cca7226c, branch feat/p20, wsl clone ~/blackout-env-p18b: fix V28 regression of V30 students (short fine-distil from P18bp2, V28-weighted) → §47.
Network errors (ENOTFOUND) kill agents occasionally — resume with SendMessage telling them to check pids first.
- **D1** agent adc56834a0b8880f0 (user request): root-cause why distillation fails, branch feat/distill-why, may use gpu1/gpu2 spare CPU AND the Mac (clone blackout-env-d1, ≤6 workers) — this agent only → §48. P20 = §47, Q12 = §46.

State at compaction (2026-10-06 ~15:00 KST): feat/policy-v2 HEAD 38a139b (pushed gpu2+gpu1). Summary/index have "진행 중" rows for Q12/P20/D1 — replace them with final rows when each agent merges (agents also add their own index rows → dedupe on merge). Section order §46 Q12, §47 P20, §48 D1.
P20 diag 1: P19 lost the opening join (decision-99 4-Hunter games vs V28: P19 14/64, V30 38, P18bp2 44, P17 53; vs P17: P19 13, V30 41); fades between P18bp2 (5k) and P19 (20k). P20 next: snapshots v_0005/10/15 h99, then 5k fine-distil from P18bp2 with V28 weighted.
No branch monitor armed in this session (old brmon.sh lives in old scratchpad /private/tmp/claude-501/-Users-mac-project-26rl/ae7bdc3d-.../scratchpad/brmon.sh; B="feat/v31-rule feat/p20 feat/distill-why" if re-armed). Agents report via SendMessage.

## Update ~16:30 KST
- D1 merged as a8489d5 (§48, pushed to gpu2/gpu1). Root cause: rare join rows at decisions 95–99 get blurred during long training. Two-net bound P18bp2→P19: vs V28 +55.4, vs P17 +34.1. Deployable hybrid: V30 until decision 100, then P19 (not gated). D1 index/summary rows are final.
- Still running: Q12 (256-seed gates for the threat0.13+camp6.5 combo and sentry8) and P20 (pilots A/B/C with --join-weight). When merging, put §46 and §47 BEFORE §48 and replace their "진행 중" rows.
- gpu2 CPU guard: ~/cpu_guard.sh (PID 2997804, log ~/cpu_guard.log). It SIGSTOPs the pgids in ~/cpu_extra.pgids when other users' CPU is ≥ 18 cores and SIGCONTs them at ≤ 12. Removing the file stops the guard (it resumes the jobs on exit). Remove the guard when Q12/P20 are done. The user wants gpu2 extras only when they don't disturb others.
- ~10-06 evening, two new agents:
  - R3: branch feat/r3-rollout-p20. Re-distils the rollout and leaf V nets from P20 and puts search on top; gpu2 GPU, collection under the guard while Q12 runs. Writes §49.
  - P21: branch feat/p21 from feat/p20. Refines v_0005 on wsl after P20's wsl evals finish. Writes §50.
  - Merge order: §46 Q12 → §47 P20 → §48 D1 (already in) → §49 → §50.
- Q12 gpu2 gate1024a was paused for R3 and resumed at 10:26 server time (R3 chain done).
- ~10-06 afternoon, two new agents:
  - B1: branch feat/b1-batched-search from feat/r3. Batches the rollout net across many games per process. Develops on the Mac, benchmarks on gpu2 GPU 0 with ≤ 2 CPU processes.
  - O1: branch feat/o1-v30-speed. Speeds up V30 on the Mac (user approved the Mac for it, ≤ 6 workers); actions must stay bit-identical.
  - Both append "B1:"/"O1:" sections; renumber them after §50 on merge.
- P20 is ADOPTED (sim 256). Its Unity milestone is queued on gpu1 (~/blackout-env-u20, waiting for PID 3109411). P21c beats P20 vs V28 by +10.5±2.9 at 64 seeds; gate in progress.
- QZ agent (branch feat/qz-quant): P20 PTQ int8 / fp16+compile, and QAT if fidelity drops; r20d256 optional. Uses gpu2 GPU 2 (≤ 2 CPU processes at nice 19). Goal is search/eval speed. B1 phase 2 approved by the user: profile step/render, then a SIMD trial; decide on a GPU port with the user afterwards.
- P20 merged as 3a0214a. Its Unity run is going on gpu1: PID 3324113, ~/blackout-env-u20; the build is a symlink to ~/blackout-env/build.
- QZ handed back early. Result: fp16c is 3.2–3.9x faster with argmax agreement 0.998, so QAT is not needed; int8 was dropped. Branch feat/qz-quant (35dd6cc, based on feat/b1).
  - Still running on gpu2: search lanes pgid 2934000/2934001 (GPU 0) and plain64 pgid 2857162 (guarded). Main waits for the lanes (task bgh6xi0rd), then resumes QZ.
  - QZ cleanup to do: git remote gpu2qz, ~/qzruns, ~/blackout-env-qz, /tmp/qz_mon_*.txt.

## State at compaction #2 (2026-10-06 ~15:00 server time / ~24:00 KST)
- feat/policy-v2 HEAD 14b2093, pushed to gpu1/gpu2. Merged: D1 §48, P20 §47 (ADOPTED student champion), O1 §51 (V30 1.8x faster, bit-identical). 623 tests pass.
- Champions:
  - Student P20 = checkpoints/distill/frozen/p20_v30.pt (md5 711d842f). Its Unity milestone runs on gpu1 (PID 3324113, ~/blackout-env-u20, ~1721/2400 at 14:30); task bdvvywszq waits for it, then pull reports/p20_unity and write §47.x.
  - Teacher V31 (Q12, adopted in sim) = V30 + threat_radius=0.13, camp_engage_radius=6.5, about +1.7 over V30. Its Unity is deferred; batch it with the next gpu1 Unity run.
- Agents and merge plan:
  - Q12 (a0266a6d2d2d0eb20): writing §46. Merge it before §47.
  - R3 (a2763f65b8f4177a0): it hands back often; resume it via SendMessage. Its gpu2 lanes (incl. o4f_r20 lanes e/f) are being waited on by main (bn0bxr4w1); on wake, resume R3 to run gate_report, the 256-seed stage, §49 and cleanup. r20d256 is md5 febb7f18.
  - P21 (a6b3bee15f6e391de): p21c beats P20 by +8 to +10.5 vs close-reserve opponents at 64 seeds; the 256-seed stage runs on wsl plus gpu1 (≤ 4 workers, 8 after Q12's PID 3382040). It writes §50. If adopted: Unity for p21c (batch with V31).
  - B1 (a1b0b1413a315de67): pipelined search is bit-identical. Phase 2 (step/render profile, SIMD trial) is on the Mac; the GPU port decision is the user's.
  - QZ (a18784b1c532404de): fp16c 3.2–3.9x, fidelity 0.998, no QAT needed. Main waits for its gpu2 lanes (bgh6xi0rd), then resumes QZ to finish. Branch is based on feat/b1, so merge B1 first.
  - Merge order and numbers: §46 Q12, §49 R3, §50 P21, §51 O1 (done), §52 B1, §53 QZ.
- gpu2 rules: other users' cppjudge has priority; our base level is 16 workers at nice 19 (`nice -n 39`). Extras go through the guard (~/cpu_guard.sh, ~/cpu_extra.pgids). Remove the guard when all gpu2 work is done.

## Update after compaction #2 (~24:40 KST)
- Q12 merged as §46 (HEAD 2159258, pushed gpu1/gpu2, 632 tests pass); v31-rule worktree removed. V31 Unity still pending (batch with p21c on gpu1).
- P21 resumed with instructions: 256 stage (wsl + gpu1 ~/blackout-env-p21), own waiter, then §50 + cleanup.
- B1 merged as §52 (HEAD 51c505e, pushed, 649 tests). GPU sim port (3–5 agent-days, ~3–4x per core) awaits USER decision — not started.
- QZ resumed (lanes done): merge policy-v2, p20c−p20 compare, combined pipeline+fp16c timing, §53. R3 waiter re-armed as bor9ngnas (lanes e/f left).
- 2026-10-07 USER DECISION: no GPU sim port (GPU is the scarce resource; port would only save CPU). Search default stays fp32.
- 2026-10-07 USER: pause new heuristic dev; after R3/P21 add "search vs V31" check, then distil + search strengthening.
- P20 Unity done & recorded §47.5 (bc679eb): direct +36.6, vs V30 +55.6, V28 +14.1, V25 +2.7. ~/blackout-env-u20 removed. Next gpu1 Unity batch = V31 + p21c (after P21 256 frees gpu1).
- QZ merged §53 (934418a, 653 tests); gpu2 guard stopped and removed (pgids empty). Remaining: R3 (bor9ngnas), P21.
- V31 Unity running on gpu1 (~/blackout-env-u31, PID 3720175, seeds 4100000+, 2400 matches ~3h), waiter b?; P21 gpu1 cleaned. P21 E leading (no regressions), V30 rows on wsl pending. R3 extending V31 to 1024 seeds.
- P21 merged §50 (dfbd0d9): P21e student champion (sim), p21e_v30.pt md5 7b04c317 copied to gpu1/gpu2. Unity for P21e queued after V31 Unity on gpu1 (do it myself).
- New agents: R4 a48845aff73b9635c (r21d256 rollout from P21e on gpu2 GPU2 → search gate §54; tell it when R3 frees GPUs 0/3), P22 a967747d7e424457b (wsl: P22a V30 teacher / P22b V31 teacher from P21e → §55; must ask before gpu1).
- V31 Unity done §46.6 (edb88a7): direct +6.2±1.8, P17 +5.1, V25 +3.5, P20 −0.4. P21e Unity running gpu1 ~/blackout-env-u21e PID 3794747 (seeds 4200000+, vs V31/V30/P20/V28), ~1.5–3h.
- R3 merged §49 (search champion P20+r20d256; vs V31 search share +1.9±1.2 undecided at 1024). All §46–§53 merged. Active: R4, P22, P21e Unity (bo9n06df0).
- OOD1 agent a1f3aa4572de940e5: OOD suite on gpu2 CPU (16 nice 19), plain lineage first, search later with my OK for GPU → §56.
- Pending after OOD1 diag: Q13/V32 agent (V19 alloc+late game + V30 opening rush), user-approved.
- User idea 2026-10-07: if h4hunt proves a true counter (non-transitive), reconsider opponent-conditional teacher switching (old "상대 조건부 교사", needs opponent ID → memory/KDA or search with opponent posterior). OOD1 asked for best-response matrix + early ID signals.

## State at compaction #3 (2026-10-07, user-approved plan)
HEAD feat/policy-v2 250de7a (all of §46–§53 merged, pushed gpu1/gpu2). Champions: student P21e (Unity: +5..+6 vs heuristics over P20, h2h −2.6 vs sim +6.6), teacher V31 (Unity +6.2 vs V30), search P20+r20d256 (R4's P21e+r21d256 is ~+8 better, pending).
Agents running:
- R4 a48845aff73b9635c (gpu2 GPUs 0/2/3, GPU1 only idle): P21e+o4f_r21 256 lanes + V31 1024 (pair with R3 1024 rows) + stale-control; → §54. At 256: vs V31 +29.1, search share +5.2±2.3, vs P20+o4f_r20 +8.6; pooled share +8.3±0.9.
- P22 a967747d7e424457b (wsl + gpu1 4 workers): P22b (V31 teacher from P21e) 256 stage; partial V30 +9.9±2.1, V31 +4.3, no regression; pending h2h/P20/P17/V29/V28 → §55. Before adopting P22b: OOD-26 check + Unity (incl. h2h vs P20/P21e).
- OOD1 a1f3aa4572de940e5 (gpu2 CPU ≤16 nice 19 + gpu1 4 workers): OOD-26 suite (feat/ood1). No student collapse vs teacher; v30_h4hunt beats whole lineage (P20 −11, V30 −16). Doing: 128-seed follow-up, per-variant mechanism diag (diag2.py), best-response matrix (h4hunt as candidate) to test non-transitivity + early ID signals → §56.
PLAN (user-approved 2026-10-07), in order:
1. OOD1 finishes diag + matrix → decide: true counter (cycle beyond 2 SE) or just stronger.
2. Start V32 agent (Q13): V30 opening (exact race, reserve join, sentry) + V19-style allocation (4 Hunters, exit guard, sanctuary sweep) and late game, tuned by OOD1's cause. Use as teacher candidate AND distill-pool/OOD opponent. Gate: sim 64→256→1024 vs pool+V31, vs P21e/P22b, OOD-26; Unity if adopt.
3. If cycle confirmed: try opponent-conditional response — first cheapest: search with opponent pool incl. h4hunt/V32 + in-game opponent posterior; then opponent-switching teacher + student with memory/obs history (KDA); or mixture of students.
4. Add OOD rule (from §56) to every student/search gate; add h4hunt-type opponents to distill pool for next P round.
gpu1 shares: OOD1 4 + P22 4 (≤8 total, nice 10/19).

## Update 2026-10-07 (after compaction #3)
- P22 merged (§55, f6e99d8): P22 = P22b `checkpoints/distill/frozen/p22_v31.pt` md5 1355c72b, sim student champion (+3.4±0.7 mean vs P21e, h2h +8.1). Copied to gpu1/gpu2.
- P22 Unity running gpu1 `~/blackout-env-u22` (needs `ln -s ~/blackout-env/build build` in fresh clones!), seeds 4300000+, ours P22;P21e vs V31/V30/P21e/P20/V28. Pull reports/p22_unity, report with --base p21e, then §55.5 + cleanup clone.
- P23 agent (a967747d7e424457b resumed, wsl): P23a (P22+V31 same pool) vs P23b (+h4/h4hunt/c0 pool), §57, plus OOD-26 screen of P22.
- OOD1 diag: only real weakness = V30 hunter_quota=4 (h4/h4hunt), (3,4) state at decision 99 → lost second race; inherited from V30, students 11 better than V30; not early-identifiable → structural fix. BR matrix still running (gpu2).
- OOD1 BR matrix: h4 (V30+hunter_quota=4) simply stronger, no cycle → opponent-conditional teacher NOT needed. h4 beats P21e +8, V31 +18, V30 +19.
- V32 agent a74a9b6d9ec867cd1 started (feat/v32 worktree .claude/worktrees/v32): baseline h4 + V19 allocation/exit guard/sweep/late-game ablations; gates on gpu2 after OOD1 frees; §58.
- OOD1 merged (§56, 5f672f4). OOD gate rule §56.6 (OOD-26 × 32 seeds × 2 sides, seed 2600000, R1–R4) now applies to every adoption. Pending: search-champion OOD run (P20/P21e + o4f vs h4/h4hunt/c0) on gpu2 GPU after R4. gpu2 CPU handed to V32.
- P22 Unity done (§55.5, a91f579): h2h vs P21e +9.4±2.2 (matches sim), P20 +5.9, V30 +5.5, V31 +0.2 → P22 champion confirmed (+ OOD rule pass). gpu1 ~/blackout-env-u22 kept ONLY as Unity build source for V32 spot-check; delete after V32 Unity.
- V32 B = A + V19 role order/guard/sweep/cargo_transform (`solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,sweep=true,cargo_transform=true`): sim 64 seeds B−A +19.6, beats P22 +15.7. Sim mechanics verified vs C#. Unity spot-check on gpu1 approved (8 workers).
- R4 merged (§54, ade5b15): search champion = P21e + r21d256 (+5.0 over P20+r20; vs V31 1024 +23.5, search share +4.5). r21d256 md5 ae35fa30 at checkpoints/rollout_net/r4/.
- R5 agent ae714ba98986fa962 (gpu2 GPUs): r22d256 from P22, P22+o4f_r22 gate, search OOD check (h4/h4hunt/c0), search vs V32 B; §59.
- V32: B fails OOD R3 / regression vs weak opps (margin compression, still +55); waiver for teacher use asked to user — pending. User idea queued to V32 agent: opponent-conditional sweep (gate sweep on observed enemy Hunter pressure). Section numbers: P23 §57, V32 §58, R5 §59.
- V32 (= B) code merged b2b09bf (`v32_four_hunters.V32`). Win-rate rule (seed SE) flags V23 −3.3±0.9 %p and v4 −3.1 → conditional adoption (teacher use), champion status pending V32.1 (opponent-conditional sweep; user wants margin loss minimised too). §58 pending from V32 agent.
- P24 assigned to P23 agent (wsl GPU): P22 + V32 teacher, §60. Section map: P23 §57, V32 §58, R5 §59, P24 §60.
- 2026-10-08: merged P23 §57 (not adopted; labels are the limit) and V32 final §58 (cf34b6c). Champion heuristic = V32.1 (`v32_four_hunters.V32_1` = V32 + adapt_k=3), passes all rules; V32 (B) = P24 teacher. Running: P24 (P23 agent a967747d7e424457b; train wsl, gates gpu1 8w), P25 = P22 + V32.1 teacher planned; R5 (ae714ba98986fa962; gpu2 GPUs + gpu1 4w). Pending: Unity milestone for V32.1 together with P24 student; add V32/V32.1/h4 to distill+OOD pools.

## State at compaction #4 (2026-10-08) — WRAP-UP MODE (user decision)
User: "near the ceiling — only finish V32.1 distill + rollout, then conclude experiments". No new experiments after these.
- P24 agent a967747d7e424457b (feat/p24, worktree .claude/worktrees/p24): P24c (P22 + V32.1 teacher, 4k) = lead student; 64-seed pooled +14.9 vs P22, h2h +22.3, vs V32 +47.9, passes weak win-rate rule (P24a +16.8 but V23 regression). Running: P24c 256 seeds (wsl + gpu1), OOD screen, Unity on gpu1 (~/blackout-env-u24: P24c/P22 vs V32.1/V32/V31/P22/P21e seeds 4400000; V32.1/V31 vs V31/P22/V23 seeds 4500000). Deliver §60, ckpts p24c_v321.pt / p24a_v32.pt.
- R5 agent ae714ba98986fa962 (feat/r5-rollout-p22): r22d256 md5 48bb786d; P22+o4f_r22 − plain +7.8 (64), vs champion +0.6 → 256 running; OOD: search no collapse, no h4 fix; vs V32 B 256 seeds: search −12.4, plain −14.1 → pause criterion NOT met. Then §59, then R6 = rollout net r24c from P24c + short 64-seed check (§61), then final cleanup.
- After both: merge §59/§60/§61 in order, update summary (final champions + conclusion section), pytest, push, remove worktrees, check servers clean (gpu1 ~/blackout-env-u24, ~/blackout-env-p24, ~/p24nets, ~/blackout-env-r5; gpu2 ~/blackout-env-r5; wsl clones).

## Final Elo/heatmap plan (agreed 2026-10-08, run AFTER P24/R5/R6 wrap-up)
- 24 policies, simulator, full matches. Heuristics (12): V1,V4,V5,V7,V15,V17,V19,V22,V25,V28,V31,V32.1 (champion configs from docs).
- Students (12): d6_v42,p5a_v22,p11_v22,p14_v25,p16_v27,p17_v28,p18bp2_v29,p19_v30,p20_v30,p21e_v30,p22_v31,p24c_v321. User may swap 1-2 students after P24/R6 results (P24a candidate).
- Stage 1: every pair 20 games (10 seeds x side swap) = 5520 games -> heatmap. Stage 2: adaptive A-optimal pair picks (elo_active-style) until every Elo SE <= 12 (~12-15k games total).
- Compute: gpu1 + gpu2 + wsl + Mac (only on AC power) CPUs. Default rating = W/D/L Elo; heatmap also shows mean margin.
- Added 2026-10-08: every pair's win-rate SE ≤ 5% (seed-level paired SE, smoothed for lopsided pairs) in addition to Elo SE ≤ 12 and the 20-game floor. Tool being built on feat/elo24 (worktree .claude/worktrees/elo24).
- 2026-10-08 13:30: R5 final 256: P22+o4f_r22 vs plain +7.3±0.5, vs champion P21e+o4f_r21 +1.6±0.6 (2.5 SE, not adopted, tie). R6 r24cd256 md5 7697fe4a, agreement 0.803/10.0°. Plain P24c 64 seeds: +34.1 V31, +18.9 P22, +24.5 vs V32.1 (student beats best heuristic). New wrap-up agents: P24 a9580972bbe6aa4b1, R5/R6 a0fe5a3040338930a.
- Elo24 stage 1 RUNNING on Mac only (user allowed ≤18 procs on AC; max_workers 16, --session-min 10), worktree .claude/worktrees/elo24 (feat/elo24 3b56e72), checkpoints symlinked into worktree, log reports/elo24/run1.log. Resume = rerun same command.
- 2026-10-08 ~17:40 WRAP-UP DONE: merged R5/R6 (§59, §61) + P24 (§60) into feat/policy-v2 (82d763b), summary ff46b42 (P24c = student champion, V23 cat 2.0 SE borderline awaiting user call), tests 662 passed, pushed gpu1/gpu2 (gpu2 mirror back on feat/policy-v2). Worktrees p24, r5 removed; servers clean (only elo24 clones ~/blackout-env-elo24 on gpu1/wsl).
- Elo24 stage 2 running (mac+gpu1+wsl), all 276 pairs pass win-rate SE ≤5%; Elo SE mean ~25 — SE 12 needs ~5x more games; asked user: stop at 20k cap / continue / top-10 only. Pending user answers: (a) adopt P24c despite V23 cat flag, (b) Elo continuation option.
