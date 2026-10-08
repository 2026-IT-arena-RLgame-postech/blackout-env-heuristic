#!/bin/bash
# Q5 extras (docs §32): (1) 10 s probes with Hunter counts (reports/race_search_p14b/probe10);
# (2) whole-match search vs search from 5 s / 10 s, P14b acting, vs P14b, 64 seeds x both sides.
cd "$(dirname "$0")/../../.."
PY=../../../.venv/bin/python
LOG=reports/race_search_p14b/logs
P=examples/race_search_p14b_eval.py
nohup env PYTHONPATH=. bash -c "for o in p14b v23; do $PY $P play --opponent \$o --tags p14b,p14b_r10,p14b_s5_10,p14b_r10_team --cap-s 10 --out reports/race_search_p14b/probe10 || exit 1; done; \
  $PY $P play --opponent p14b --tags p14b_from10 --seeds 64" > $LOG/extra1.log 2>&1 &
echo $! > $LOG/extra1.pid
nohup env PYTHONPATH=. bash -c "$PY $P play --opponent p14b --tags p14b_full,p14b_from5 --seeds 64" > $LOG/extra2.log 2>&1 &
echo $! > $LOG/extra2.pid
