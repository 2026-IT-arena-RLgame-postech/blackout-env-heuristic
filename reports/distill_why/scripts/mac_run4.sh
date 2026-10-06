#!/bin/bash
cd /Users/mac/project/26rl/blackout-env/.claude/worktrees/distill-why
export PYTHONPATH=. D1_FROZEN=/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
P=/Users/mac/project/26rl/blackout-env/.venv/bin/python
nice -n 10 $P examples/d1_why.py play --ours 'nostand:2.5,6:v30;nostand:0,0:v30' --opponent v28 --shadows 'v30' --seeds 32 --chunk 4 --workers 6 --out /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/out
echo ALLDONE
