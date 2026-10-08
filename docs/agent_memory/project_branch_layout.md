---
name: project-branch-layout
description: Since 2026-09-30, architecture work (policy v2) is on its own branch and experiments run from a separate stable worktree — which checkout to run what from
metadata:
  type: project
---

Since 2026-09-30:
- **`feat/policy-v2`** is the branch in the main checkout, `/Users/mac/project/26rl/blackout-env`. It carries the KDA memory student, `act:` teachers, the policy head and finer actions. Design doc: `docs/policy_v2_design_ko.md`.
- **`feat/v17-distill`** is the stable branch, checked out as the worktree `/Users/mac/project/26rl/blackout-env-runs`. Run D5/D6 and other experiments from there.
  - checkpoints, build, datasets and .venv are symlinks to the main checkout. The reports result dirs are linked one by one, because the reports dir has tracked files.
  - Scripts must `cd` into the worktree and set `PYTHONPATH=<worktree>`. The `.venv` editable install points at the main checkout, so without it `examples/*.py` imports the main checkout's code.

**Why:** subagents were editing code in the one checkout while the D5 eval spawned workers from it. The workers imported half-edited files, and 94 held-out matches had to be discarded. The user also asked for a separate branch for the architecture change.

**How to apply:** never launch long runs from a checkout where code is being edited. Launch scripts live in `checkpoints/distill/` (`eval_d5.sh`, `launch_d6.sh`) and already target the runs worktree.

2026-10-01: a third worktree, `/Users/mac/project/26rl/blackout-env-p1` (detached at 4bf4955, feat/policy-v2 code), runs the P1 chain (`checkpoints/distill/launch_p1.sh`). The chain waits for "D6 CHAIN DONE". Results go to `reports/p1_heldout`.
