#!/bin/bash
# Q5 core matrix (docs §32): 128 seeds x both sides, full matches (150 s cap), three processes.
# Usage: bash reports/race_search_p14b/scripts/run_core.sh   (from the worktree root; detached, pid files in the log dir)
cd "$(dirname "$0")/../../.."
PY=../../../.venv/bin/python
LOG=reports/race_search_p14b/logs
mkdir -p $LOG
TAGS=p14b,p14b_r10,p14b_r10_team,p14b_s5_10
run() {  # name, command...
  local name=$1; shift
  nohup "$@" > $LOG/$name.log 2>&1 &
  echo $! > $LOG/$name.pid
}
run nets env PYTHONPATH=. bash -c "$PY examples/race_search_p14b_eval.py play --opponent p14b --tags $TAGS,f3,f3_r10 && \
  $PY examples/race_search_p14b_eval.py play --opponent p11 --tags $TAGS"
run v25 env PYTHONPATH=. $PY examples/race_search_p14b_eval.py play --opponent v25 --tags $TAGS,f3,f3_r10
run v23 env PYTHONPATH=. $PY examples/race_search_p14b_eval.py play --opponent v23 --tags $TAGS
