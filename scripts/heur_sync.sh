#!/bin/bash
# Push the current branch to poscat_gpu1 and fast-forward the heuristic-search worktree ~/blackout-env-heur.
set -e
git push -q -f poscat_gpu1:blackout-env HEAD:refs/heads/heur-incoming
ssh poscat_gpu1 'cd ~/blackout-env-heur && git merge -q --ff-only heur-incoming && git log --oneline -1'
