#!/bin/bash
# Q8 Mac CPU sim eval: $1 = ours (';'), $2 = opponents (';'), $3 = seeds
cd "$(dirname "$0")/../../.."  # a clone of feat/rollout-v27, not the main checkout
export PYTHONPATH=. OMP_NUM_THREADS=1
export BLACKOUT_SMALL="v25d256=/Users/mac/project/26rl/blackout-env/checkpoints/rollout_net/v25/v25d256.pt,v25f3x=/Users/mac/project/26rl/blackout-env/checkpoints/rollout_net/v25/v25f3x.pt,v27d256=/Users/mac/project/26rl/blackout-env/checkpoints/rollout_net/v27/v27d256.pt,v27f3x=/Users/mac/project/26rl/blackout-env/checkpoints/rollout_net/v27/v27f3x.pt"
export BLACKOUT_NETS="p16=/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p16_v27.pt,p14b=/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p14_v25.pt,p11=/Users/mac/project/26rl/blackout-env/checkpoints/distill/frozen/p11_v22.pt"
exec nice -n 10 .venv/bin/python examples/v27_eval.py play --ours "$1" --opponent "$2" --seeds $3 --seed0 1100000 --chunk 32    --cap-s 425 --stop-exhausted --workers 8 --device cpu --out reports/rollout_v27/sim_mac
