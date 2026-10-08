---
name: feedback-autonomous-improvement-loop
description: "From 2026-10-04 the user wants the improvement cycle run autonomously: search → rule-ize → distil + rollout re-train → new search, staged evals (small→large), pipelined across machines"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae7bdc3d-fc33-4539-a05d-ed299772337d
  modified: 2026-10-04T13:52:12.518Z
---

User (2026-10-04): run the work process autonomously as a repeating cycle:
1. Check whether a search-based policy improves on the current champion.
2. If yes, extract the gain into a rule → stronger heuristic (e.g. V25 race_walk, V27 reserve_stage).
3. If a stronger heuristic exists, distil it (student) and re-distil the search rollout/opponent nets from it.
4. Build the next search policy on the new student/rollout nets → back to 1.

Evaluation between every step is mandatory. Start small (e.g. 64 seeds × 2 sides) and only scale up (256 → 1024, then Unity) when the result is not significant or information is insufficient.
Pipeline the stages across machines in parallel (gpu2 = sim evals/search/Unity, wsl_gpu = GPU-heavy/CPU-light jobs; gpu1 = CPU-only jobs (nice, ≤8 workers) — [[project-remote-gpu-servers]]).

**Why:** the V25 (race) and V27 (second race) gains both came from exactly this loop; the user wants it to keep turning without being prompted for each step.

**How to apply:** pick the next step myself when one finishes, keep a champion registry (best teacher/student/rollout/search config) up to date in docs, report each gate result concisely ([[feedback-concise-result-reports]]), commit/merge per [[feedback-sync-and-commit]]. Still ask before GitHub pushes, new axes (RL, Unity C# changes), or anything touching user files.

User follow-ups (2026-10-04): end goal = **a single student net for inference** (runtime search may be allowed, but don't count on it — search is a teacher). Guideline additions accepted (pre-set staged gates 64→256→1024→Unity with |Δ|≥3 SE adopt / <1 SE at ≥256 reject; fixed benchmark pool + worst-case; new champion must beat old ones head-to-head; Unity only at milestones; champion registry; systematic rule extraction). Keep the opponent pool **moderate** — too many is slow, too few loses diversity (≈6–8 opponents: previous champions + a couple of perturbed/held-out variants).

User (2026-10-04): **update the experiment record files (docs/experiments_summary_ko.md, experiments_index_ko.md, design-doc section) every time one pipeline cycle completes** (and per experiment as it merges).

Refinement (2026-10-05, main session, recorded in §36): a "regression" needs > 1 SE **and** > 1 point — at ceiling opponents SE falls to ~0.2, so 0.3-point differences would otherwise block adoption (V28 vs perturbed V27 −0.34±0.21 was adopted).
