#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
J=$(PYTHONPATH=. .venv/bin/python -c "
from blackout_env.train.distill.ood_bp import OODBP
print(';'.join(d for _,d,_,_ in OODBP))")
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "$J" --opponent B6 --seeds 24 --seed0 4500000 --chunk 6 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/ood/strength > reports/bp/ood/strength.log 2>&1
# behaviour stats
PYTHONPATH=. OMP_NUM_THREADS=1 BLACKOUT_TPD=1 .venv/bin/python examples/econ_stats.py play --hz 50 --ours "B6;$J" --opponent B6 --seeds 8 --seed0 4600000 --chunk 4 --workers 16 --out reports/bp/ood/econ > reports/bp/ood/econ.log 2>&1
