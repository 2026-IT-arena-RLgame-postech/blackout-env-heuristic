#!/bin/zsh
# old vs new action/state hashes for every registered heuristic (+ the gated descriptors), both sides of 14 games
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 ALLSIDES=1
F="_s91000[03]|_s91010[03]|_s91020[03]|_s91030[03]|_s91040[03]|_s91050[03]|_s91060[03]"
mkdir -p /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/allpol
for P in $(cat /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/policies.txt); do
  T=$(echo $P | tr ':=,' '___')
  for tree in orig new; do
    D=$([ $tree = orig ] && echo /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/orig || echo /Users/mac/project/26rl/blackout-env/.claude/worktrees/o1)
    (cd $D && POLICY=$P PYTHONPATH=. nice -n 10 /Users/mac/project/26rl/blackout-env/.venv/bin/python /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/replay_any.py /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/corpus /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/allpol/${T}_$tree.json 6 "$F" > /dev/null 2>&1)
  done
  echo -n "$P: "; /Users/mac/project/26rl/blackout-env/.venv/bin/python /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/cmp.py /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/allpol/${T}_orig.json /private/tmp/claude-501/-Users-mac-project-26rl/cf90b6b9-cf12-47d9-8496-6b6e85b4751b/scratchpad/o1/allpol/${T}_new.json | tail -1
done
