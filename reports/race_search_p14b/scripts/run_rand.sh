#!/bin/bash
# Q5 random-branch control (docs §32): same non-base rate as p14b_r10 without search.
cd "$(dirname "$0")/../../.."
PY=../../../.venv/bin/python
LOG=reports/race_search_p14b/logs
nohup env PYTHONPATH=. bash -c "for o in p14b v23 v25; do $PY examples/race_search_p14b_eval.py play --opponent \$o --tags p14b_r10_rand || exit 1; done" > $LOG/rand.log 2>&1 &
echo $! > $LOG/rand.pid
