---
name: feedback-heuristic-dev-paused
description: From 2026-10-07 no new heuristic (V32+/Q13 rule-izing) work until search is confirmed clearly stronger than the best heuristic; focus on distillation + search strength
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf90b6b9-cf12-47d9-8496-6b6e85b4751b
  modified: 2026-10-06T15:19:58.663Z
---

User decision 2026-10-07: pause new heuristic development (no new Q-series rule-izing / V32) until search is confirmed clearly better than the best heuristic (V31/V30). Focus on distilling the strongest student from existing heuristics (V31/V30 teacher) and strengthening search (base net, rollout nets, search settings).

**Why:** the loop's "rule-ize" step was producing small gains (V31 ≈ +1.7 over V30) while distillation and search have more headroom.

**How to apply:** the autonomous loop [[feedback-autonomous-improvement-loop]] skips "rule-ize" for now. Next steps should include an explicit check "search policy vs V31" (head-to-head + pool, staged seeds) to decide when to lift the pause.

**Exception approved 2026-10-07:** user asked for V32 = V19-style unit allocation (4 Hunters) + V19 late-game management, with V30's opening Hunter transform rush — to be started only AFTER OOD1's analysis of why V30 h4hunt beats the lineage. Use V32 both as a teacher candidate and as a distill-pool / OOD opponent.

**Update 2026-10-08:** even if the resume criterion is met (P24c alone beat V32.1 +24.5 at 64 seeds; R6 search check pending), do NOT resume heuristic development until the 24-policy Elo/win-rate evaluation (elo24) is finished and reported. Then ask/plan next steps.
