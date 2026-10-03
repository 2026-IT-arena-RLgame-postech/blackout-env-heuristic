#!/bin/bash
# P15 step 3: strategic_v26 variants, paired with the step-1 base runs (256 matches per pair)
cd "$(dirname "$0")/../.."
PY=/Users/mac/project/26rl/blackout-env/.venv/bin/python
export PYTHONPATH=.
OUT=reports/p15_eval
while kill -0 $(cat $OUT/measure.pid) 2>/dev/null; do sleep 20; done
$PY examples/late_hunter_eval.py play --opponent strategic_v25 --ours 'strategic_v26;strategic_v26:trade_mode=always;strategic_v26:trade_until=30;strategic_v26:trade_walk_reach=4;strategic_v26:race_walk=false;strategic_v26:race_walk=false,race_commit=false;strategic_v26:race_walk=false,trade_mode=always' --out $OUT --workers 3
$PY examples/late_hunter_eval.py play --opponent 'strategic_v23;p11' --ours 'strategic_v26;strategic_v26:trade_mode=always;strategic_v26:trade_walk_reach=4' --out $OUT --workers 3
