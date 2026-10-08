#!/bin/bash
# §25 replication arms vs V23 (+V19) in the sim, same seeds as runs/sim.
cd ~/blackout-env-p11m; export PYTHONPATH=$PWD BLACKOUT_MAIN=$HOME/blackout-env CUDA_VISIBLE_DEVICES=2
exec ~/blackout-env/.venv/bin/python examples/rollout_net_strength.py sim --device cuda --workers ${WORKERS:-6} \
  --seeds 500 --seed0 1100000 --candidates $CANDS --opponents ${OPPS:-v23} --h2h --out ${OUT:-runs/sim_rep}
