---
name: feedback-weak-opp-winrate
description: For opponents we already crush, judge regressions by win rate, not mean margin (user, 2026-10-07)
metadata:
  type: feedback
---

User (2026-10-07, about V32 B's margin compression vs V23/P14b/v4/v21): against weak opponents, win rate matters more than mean score margin; if the win rate barely changes, adoption is fine.

**Why:** a smaller blowout (+60 instead of +75) against an opponent we always beat costs nothing real, while the gain vs strong opponents shows up as large win-rate gains.

**How to apply:** in gates and the OOD rule R3 ([[feedback-ood-robustness]]), for opponents both arms beat with win ≥ ~0.9, use paired win-rate Δ (and loss count / catastrophic rate) as the regression test instead of margin Δ; keep margin-based rules for peer/strong opponents. Pending formalisation in §58 / §56.6 addendum.

Follow-up (same day): still aim to minimise margin loss vs weak opponents — win rate is the adoption test, but margin compression is a research target (e.g. opponent-conditional sweep for V32.1); keep investigating such adaptive fixes.
