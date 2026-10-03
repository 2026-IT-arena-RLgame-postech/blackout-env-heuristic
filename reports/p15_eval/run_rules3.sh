#!/bin/bash
# P15 step 3c: trades only with a Hunter surplus, and short-range (opportunistic) trades
cd "$(dirname "$0")/../.."
PY=/Users/mac/project/26rl/blackout-env/.venv/bin/python
export PYTHONPATH=.
OUT=reports/p15_eval
while kill -0 $(cat $OUT/rules2.pid) 2>/dev/null; do sleep 20; done
$PY examples/late_hunter_eval.py play --opponent strategic_v25 --workers 4 --out $OUT --ours "strategic_v26:trade_mode=surplus;strategic_v26:trade_mode=always,trade_radius=3;strategic_v26:race_walk=false,trade_mode=surplus;strategic_v26:race_walk=false,trade_mode=always,trade_radius=3"
