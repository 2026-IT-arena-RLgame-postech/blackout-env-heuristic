---
name: state-2026-10-08-final
description: Final state after the 2026-10-08 wrap-up + Elo-24 — champions, where results live, what is next (read first after compaction)
metadata:
  type: project
---

All experiments wrapped up 2026-10-08; nothing running on any host, worktrees p24/r5/elo24 removed, remote clones cleaned.

- Champions: student **P24c** (`checkpoints/distill/frozen/p24c_v321.pt`, md5 20a631e1; user confirmed adoption despite V23 cat 2.0 SE borderline), heuristic **V32.1**, search champion still **P21e + r21d256** (§54); R5 P22+r22 tied (§59). R6 rollout `checkpoints/rollout_net/r6/r24cd256.pt` (md5 7697fe4a): 64-seed P24c+search beats V32.1 +41.1 (not gated).
- Elo-24 (§62, `reports/elo24`, `examples/elo24.py`): 23,634 sim games; P24c 2666 > V32.1 2611 > P21e 2567 ≈ P22 2566 > P20 2518 > P18bp2 ≈ V31 2447. Plots use the old tournament style (RdYlGn heatmap with n/SE, Elo dot chart with 95% CI, LOO diagnostics) — user wants that style for all such images.
- feat/policy-v2 at docs commit after 4419bec, pushed to gpu1/gpu2; tests 670 passed.
- Next (needs user go-ahead): heuristic-dev pause may now be lifted (user said decide after Elo-24); candidate follow-ups: P24c+o4f_r24c 256-seed gate + OOD, P24c+r22d256 control.

Related: [[feedback-heuristic-dev-paused]], [[feedback-weak-opp-winrate]], [[feedback-ood-robustness]].
