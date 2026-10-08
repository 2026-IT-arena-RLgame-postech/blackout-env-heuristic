#!/bin/bash
# P15 step 3b: favourable-only trades (deficit / camper) and the far walker commit, on V25 and on
# the race-losing base (V25 walk off = V23 + race_commit); canonical parity rule vs V23 / P11
cd "$(dirname "$0")/../.."
PY=/Users/mac/project/26rl/blackout-env/.venv/bin/python
export PYTHONPATH=.
OUT=reports/p15_eval
W=late_trade=false,trade_walk_reach=3,trade_walk_count=deficit
$PY examples/late_hunter_eval.py play --opponent strategic_v25 --workers 3 --out $OUT --ours "strategic_v26:trade_mode=deficit;strategic_v26:trade_mode=camper;strategic_v26:$W;strategic_v26:$W,trade_walk_safe=false;strategic_v26:race_walk=false,trade_mode=deficit;strategic_v26:race_walk=false,trade_mode=camper;strategic_v26:race_walk=false,$W;strategic_v26:race_walk=false,$W,trade_walk_safe=false"
$PY examples/late_hunter_eval.py play --opponent 'strategic_v23;p11' --workers 3 --out $OUT --ours 'strategic_v26'
