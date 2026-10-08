---
name: feedback-sync-and-commit
description: Keep the SSH servers and the Mac in sync with scripts/remote_sync.sh, and commit local work to git often
metadata:
  type: feedback
---
User (2026-10-03): keep the poscat_gpu1/2 servers and local data in sync, and commit local data to git frequently.

**Why:** work now runs on the remote servers ([[project-remote-gpu-servers]]); results must not live only on a server, and local history should track every step.

**How to apply:**
- `scripts/remote_sync.sh status|push|pull|up|sync [host]` (blackout-env repo). `push` = git push of the current branch into the server mirror ~/blackout-env (updateInstead; the mirror is reset to clean, never edit code there — task code goes in ~/blackout-env-<task>). `pull` = reports/ + checkpoints/ back (rsync --update, no deletes). `sync` = push + pull + commit pulled reports (<20 MB files).
- Push before every server launch; pull + commit when a server run finishes (and at milestones of long runs).
- Commit code/docs/reports locally after each meaningful step (small commits), never the user's own edits (examples/evaluate_checkpoint_vs_heuristic.py, untracked examples/gui_match.py) and never large data (checkpoints/datasets are gitignored). No GitHub push without asking.

When resolving merges in the main checkout, stage only the conflicted/merged paths by name — never `git add -A <dir>` / `git add examples`: it swept the user's uncommitted examples/evaluate_checkpoint_vs_heuristic.py edit, untracked examples/gui_match.py and .DS_Store into a merge commit on 2026-10-04 (amended out before any GitHub push). Check `git show --stat HEAD` for those names before pushing.
