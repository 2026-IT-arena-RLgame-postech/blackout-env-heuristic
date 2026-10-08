#!/bin/bash
# Q9 Mac CPU sim eval (docs §39): $1 = ours (';'), $2 = opponents (';'), $3 = seeds; run from a clone of feat/rollout-v28
cd "$(dirname "$0")/../../.."  # a clone of feat/rollout-v28, not the main checkout
export PYTHONPATH=. OMP_NUM_THREADS=1
M=${BLACKOUT_MAIN:-/Users/mac/project/26rl/blackout-env}/checkpoints
export BLACKOUT_SMALL="v27d256=$M/rollout_net/v27/v27d256.pt,v27f3x=$M/rollout_net/v27/v27f3x.pt,v28d256=$M/rollout_net/v28/v28d256.pt,v28f3x=$M/rollout_net/v28/v28f3x.pt"
export BLACKOUT_NETS="p17=$M/distill/frozen/p17_v28.pt,p16=$M/distill/frozen/p16_v27.pt,p14b=$M/distill/frozen/p14_v25.pt,p11=$M/distill/frozen/p11_v22.pt"
exec nice -n 10 .venv/bin/python examples/v27_eval.py play --ours "$1" --opponent "$2" --seeds $3 --seed0 1100000 --chunk 32 \
   --cap-s 425 --stop-exhausted --workers 8 --device cpu --out reports/rollout_v28/sim_mac
