#!/bin/bash
# usage: screen_gpu1.sh NAME SEEDS SEED0 CKPT 'cands' [WORKERS]   -- runs on poscat_gpu1 (nice 19), pulls matches.jsonl back when done
N=$1
ssh poscat_gpu1 "cd ~/blackout-env-bp && mkdir -p reports/bp/exp2 && PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python examples/v27_eval.py play --hz 50 --ours '$5' --opponent 'net:/home/cucumbersaurus/bpruns/$4.pt' --seeds $2 --seed0 $3 --chunk 6 --cap-s 425 --stop-exhausted --workers ${6:-12} --out reports/bp/exp2/$N > reports/bp/exp2/$N.log 2>&1"
mkdir -p /Users/mac/project/26rl/blackout-env-teach/reports/bp/exp2/$N
scp -q poscat_gpu1:blackout-env-bp/reports/bp/exp2/$N/matches.jsonl /Users/mac/project/26rl/blackout-env-teach/reports/bp/exp2/$N/
