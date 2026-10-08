---
name: project-inflight-2026-10-04
description: "In-flight experiments/agents as of 2026-10-04 evening (Q2, R1, P16, V27 Unity) and how to finish each — read first after compaction"
metadata:
  node_type: memory
  type: project
  originSessionId: ae7bdc3d-fc33-4539-a05d-ed299772337d
  modified: 2026-10-04T08:47:49.472Z
---

State at compaction (2026-10-04). Main checkout /Users/mac/project/26rl/blackout-env, branch feat/policy-v2 at fbf4ffb (pushed to poscat_gpu2 mirror; not GitHub). Tests 576 passed / 10 skipped. poscat_gpu1 is OFF-LIMITS (other user) — see [[project-remote-gpu-servers]].

Merged today (results in docs/policy_v2_design_ko.md): §24 search-new, §25 H5 rejected, §26 S4 (o4fmean), §27 H8 rejected, §28 V25 (+Unity §28.7), §29 P14b (+Unity §29.5: 99-arrival 2000/2000, Δ vs P11 V23 +77.8), §30 P15 rejected, §32 Q5 (no race-window search; start search at 5 s with P14b base), §33 Q6 (second race; strategic_v27 reserve_stage=1.5: +50 vs V25, +44 vs P14b sim).

In flight (background agents; each gets a completion notification / SubagentHandback):
- **Q2**: DONE, merged 0e7d590 (§31). v25d256 ≈ P14b (within 1 SE), v25f3x fallback; ckpts in checkpoints/rollout_net/v25/ (Mac + gpu2). R1 told to add v25d256 variant (leaf V needs refit). wsl_gpu freed → P16 told to start.
- **R1** agent a7dff7bc38bfafa55, branch feat/blind-search, gpu2 GPU1–2 (~/blackout-env-r1): unknown-opponent search, P14b base + search from 5 s, opponent models det/o4fmean/mixtures, held-out pool incl. V27 → docs §34.
- **P16**: agent handed back early; MAIN finishes it. Training on wsl_gpu ~/p14runs/p16 (pid file run.pid, stdout.log; started 18:11, ~3 h), worktree .claude/worktrees/v27-distill (feat/v27-distill e5a0605), laptop clone ~/blackout-env-p16. Eval chain armed (pid ~/p16tmp/eval.pid) → ~/blackout-env-p16/reports/p16_sim/stdout.log ends with 'P16 EVAL DONE' (~2 h more). Then: copy reports/p16_sim (report.txt, matches.jsonl) + run metrics/run_info as reports/p16_train into worktree; scp ~/p14frozen/p16.pt → checkpoints/distill/frozen/p16_v27.pt (no commit) and to gpu2; write §35 (Δ vs P14b per opp, second-race W/T/L, reserve100 median/p10, decision-99 count) + experiments_index row after P14; commit by name, merge.
- **V27 Unity check**: DONE, merged as §33.6 (71a23f5): V27−V25 +50.7±2.9 vs V25, +42.7±4.5 vs P14b; gpu2 clone deleted.

- **Q7** agent a704647253748a54b, branch feat/v28-rule: why P16 beats V27 (+35) → rule `strategic_v28`; runs on gpu1 CPU-only (~/blackout-env-q7, nice, ≤8 workers) → docs §36.
- Loop plan after P16 eval: rollout re-distil from P16 on wsl (Q2 recipe); Unity milestone P16 vs V25/V27/P14b on gpu2 after R1 Unity; small search-on-P16 check on gpu2. Goal = single student net.
- gpu1 old clones -s3/-s4/-search deleted by the user (2026-10-04). -arch/-unity have uncommitted state: keep.

Merge procedure: `git merge --no-ff --no-commit <branch>`, resolve docs by keeping both sides in section order, stage files BY NAME, check no examples/evaluate_checkpoint_vs_heuristic.py / examples/gui_match.py / .DS_Store staged, commit, `pytest -q tests`, `scripts/remote_sync.sh push poscat_gpu2`, report concisely in Korean ([[feedback-concise-result-reports]]). Re-arm a branch-tip Monitor (bash loop on git rev-parse of the in-flight branches) every 30 min.

Current champions (2026-10-04): student P16 (p16_v27.pt, sim eval partial: −P14b +58 vs V25, +92 vs V27, +60 h2h; beats teacher V27 +35); teacher V27(1.5) (Unity-confirmed); rollout v25d256 (from P14b; re-distil from P16 next); search = P14b/P16 base + o4fmean from 5 s.
Monitors to re-arm (30 min): P16 eval (wsl ~/blackout-env-p16/reports/p16_sim/stdout.log 'P16 EVAL DONE', eval pid ~/p16tmp/eval.pid); R1 branch tip feat/blind-search; Q7 branch feat/v28-rule. R1 Unity o4f progress: gpu2 ~/blackout-env-r1/reports/blind_search/unity/o4f.log.
P16 finish: copy ~/blackout-env-p16/reports/p16_sim + ~/p14runs/p16 metrics/run_info into worktree .claude/worktrees/v27-distill as reports/p16_sim, reports/p16_train; write §35 + index row (exists, update); commit by name; merge; update experiments_summary_ko.md.

Next after these: Phase 2 continues in R1 (swap in Q2 rollout net; Unity confirm best config); possibly re-distil rollout nets from P16 (V27).

- 2026-10-04 23:20: P16 eval tail split to the Mac: clone /Users/mac/project/26rl/blackout-env-p16eval (run_mac.sh, main .venv + PYTHONPATH, nets on CPU) plays opponents V19→P11→V23 into reports/p16_sim_mac/matches.jsonl. When the union of wsl ~/blackout-env-p16/reports/p16_sim/matches.jsonl + Mac file covers 4608 keys: kill both plays (wsl eval chain pid ~/p16tmp/eval.pid + its python children; Mac pkill -f p16_sim_mac), merge dedup into one matches.jsonl, run `v27_eval.py report --base p14b`. Note in §35 that V19/P11/V23 rows partly ran with CPU nets (float-level differences only). Delete the Mac clone after.

- 2026-10-05 ~00:30: **P16 DONE**, merged 3811bf5 (§35, index, summary updated; tests 577 pass; pushed gpu2+gpu1). P16 = student champion (−P14b +58 V25, +92 V27, +63 h2h; beats V27 +35). Mac clone deleted, wsl eval procs killed.
- **Q8** agent a094773a48927a236, branch feat/rollout-v27 (worktree .claude/worktrees/rollout-v27), wsl_gpu (+Mac CPU evals): rollout nets v27d256/v27f3x from P16 per §31 recipe → docs §37; ckpts checkpoints/rollout_net/v27/. Merge when it reports.
- Still pending on gpu2 after R1 Unity (~910/1200 at 00:30): P16 Unity milestone (vs V25/V27/P14b ×300) + small search-on-P16 check (64 seeds). Branch monitor now covers feat/blind-search, feat/v28-rule, feat/rollout-v27.
- **P16 Unity milestone** queued on gpu2: ~/blackout-env-p16u/run.sh (pid 2116380, waits until R1's Unity players are gone for 2 checks, then 8 workers: ours P16,P14b × opp V25,V27(1.5),P14b, 150 seeds, seed0 4000000, cap 425 → reports/p16_unity; log play.log ends 'P16 UNITY DONE'). R1 Unity v25d256 ETA ~18:45 gpu2 time (UTC) = 03:45 KST 10-05. Then copy reports/p16_unity to main, add §35.4 Unity, commit; delete ~/blackout-env-p16u.
- R1 added `--base p16` sim lane (7370f11) = search-on-P16 check.
- 2026-10-05 ~02:00: **Q8 DONE**, merged b6f7a79 (§37): rollout champion = v27d256 (checkpoints/rollout_net/v27/, Mac+gpu2+wsl ~/rv27runs/ck). Leaf V refit on v27d256 still needed — offered wsl to R1. Q7 pilot (16 seeds): V27(reserve_stage=0.5) +44 vs V27(1.5), +22 vs P16; V28 trade_hold=advance +27 vs V27(0.5) — gating in progress (if it holds → P17 = distil V27(0.5)/V28 from P16 on wsl).
- **P17** (speculative, started 2026-10-05 ~03:30 KST) on wsl_gpu: clone ~/blackout-env-p17 (feat/v28-rule 9a2fa5c), run ~/p14runs/p17 (run.pid, stdout.log, ~3 h), INIT p16.pt, teacher act:strategic_v28:reserve_stage=0.5,trade_hold=advance, 9 workers, launch script ~/p17tmp/launch.sh (p16 script with R swapped). Kill if Q7 gate rejects V28(0.5,advance). Afterwards: freeze ~/p14frozen/p17.pt, eval like §35 with base p16 (ours p17;p16;V28 vs V28, V27(1.5), P16, V25, P14b, V23/P11) — can split tail to Mac until 10-05 evening.
- P17 eval chain queued on wsl (~/p17tmp/eval.sh, pid ~/p17tmp/eval.pid): waits for run, freezes ~/p14frozen/p17.pt, plays ours p17;p16;V28(0.5,advance) × opp V28, p16, V27(1.5), V25, P14b, V23, 64 seeds (1100000+), base p16 → ~/blackout-env-p17/reports/p17_sim (pre-seeded with 512 P16 rows from reports/p16_sim). Ends 'P17 EVAL DONE'. Plan: split tail to Mac like P16 (reverse opponent order, clone + run_mac.sh, BLACKOUT_NETS with p17 scp'd to Mac checkpoints/distill/frozen/p17_v28.pt).
- Q7 gate (256 seeds, Δ vs V27(1.5)): V28(0.5,advance) h2h +36.6±2.4, vs P16 +72.5±8.0 (64), pooled 8 opp +23.4±2.5; P14b −1.6±1.5 / perturbed V27 −0.6±0.4 being re-run at 1024 on gpu1. P16 Unity on gpu2 820/1800 at ~05:40 KST.
- 2026-10-05 ~06:00: **R1 DONE**, merged 5fc5b8b (§34). Search champion = student base + o4fmean from 5 s + rollout distilled from the acting student (P16 → v27d256), leaf V s2v_f3xd2/iter2 unrefit. P16-base 32 seeds: pooled +16.4±2.1. Only Q7 branch left in the branch monitor.
