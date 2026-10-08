---
name: feedback-ood-robustness
description: User priority (2026-10-07) — not collapsing out-of-distribution matters as much as search/distill accuracy; gates should check OOD worst-case
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf90b6b9-cf12-47d9-8496-6b6e85b4751b
  modified: 2026-10-07T01:05:39.080Z
---

User said (2026-10-07): besides improving search and distillation accuracy, not collapsing OOD is very important.

**Why:** recent gates (P14b…P21e) only used strong strategic heuristics + own nets; the old OOD set (examples/distill_heldout_eval.py OOD_OPPONENTS) was last run in the P11 era. Search with a mismatched rollout/opponent model already showed big losses (P20 + v28d256 −27 vs V30).

**How to apply:** OOD1 agent (2026-10-07) builds an OOD suite + proposes a non-regression rule; once merged, include the OOD worst-opponent / catastrophic-loss-rate check in every student and search adoption gate. Related: [[feedback-heuristic-dev-paused]], [[feedback-autonomous-improvement-loop]].
