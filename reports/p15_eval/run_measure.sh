#!/bin/bash
# P15 step 1: measurement + base runs (256 matches per pair, cap 150 s, 4 workers)
cd "$(dirname "$0")/../.."
PY=/Users/mac/project/26rl/blackout-env/.venv/bin/python
export PYTHONPATH=.
OUT=reports/p15_eval
$PY examples/late_hunter_eval.py play --opponent strategic_v25 --ours 'strategic_v25;strategic_v23;strategic_v25:race_walk=false;p11' --out $OUT
$PY examples/late_hunter_eval.py play --opponent strategic_v23 --ours 'strategic_v25;strategic_v23' --out $OUT
$PY examples/late_hunter_eval.py play --opponent p11 --ours 'strategic_v25' --out $OUT
