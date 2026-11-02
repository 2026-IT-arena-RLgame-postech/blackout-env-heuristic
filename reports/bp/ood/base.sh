#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
R=/Users/mac/project/26rl/blackout-env-teach/checkpoints/bp
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "B6;net:$R/rl_bp6_1050.pt;net:$R/rl_bp5_450.pt" --opponent oodbp --seeds ${SEEDS:-12} --seed0 6000000 --chunk 6 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/base > reports/bp/ood/base.log 2>&1
