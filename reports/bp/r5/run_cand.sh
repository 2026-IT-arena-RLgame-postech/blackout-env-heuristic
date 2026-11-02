#!/bin/bash
# usage: run_cand.sh NAME BASE 'extras' 'opps' SEEDS [SEED0]   (50 Hz)
cd /Users/mac/project/26rl/blackout-env-teach
export BLACKOUT_TPD=1
exec .venv/bin/python scripts/bp_cand.py "$@"
