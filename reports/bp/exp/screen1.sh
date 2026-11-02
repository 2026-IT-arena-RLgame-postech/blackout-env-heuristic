#!/bin/bash
# usage: screen1.sh OUTDIR SEEDS SEED0 RLCKPT 'cand;cand;...'
cd /Users/mac/project/26rl/blackout-env-teach
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$5" --opponent "net:/Users/mac/project/26rl/blackout-env-teach/checkpoints/bp/$4.pt" --seeds $2 --seed0 $3 --chunk 6 --cap-s 425 --stop-exhausted --workers ${W:-16} --out $1 > $1.log 2>&1
