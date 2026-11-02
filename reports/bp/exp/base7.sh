#!/bin/bash
cd /Users/mac/project/26rl/blackout-env-teach
R=/Users/mac/project/26rl/blackout-env-teach/checkpoints/bp
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python examples/v27_eval.py play --hz 50 --ours "net:$R/rl_bp7_1200.pt" --opponent "oodbp=raid_south,blockade,exit_camp,storm,hunter_duel,turtle,stealer,fwd_post,camp_after_open" --seeds 24 --seed0 20000000 --chunk 6 --cap-s 425 --stop-exhausted --workers 16 --out reports/bp/exp/base7 > reports/bp/exp/base7.log 2>&1
