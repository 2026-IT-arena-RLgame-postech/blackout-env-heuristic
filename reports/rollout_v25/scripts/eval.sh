#!/bin/bash
# docs §31 sim strength on wsl_gpu: candidates (first = paired reference) x opponents, seeds 1100000+ x both sides.
# usage: CANDS="f3x new" OPPS="p11 v23 v25 p14b" SEEDS=256 OUT=reports/rollout_v25/sim bash eval.sh
cd ~/blackout-env-rv25
export PYTHONPATH=. OMP_NUM_THREADS=1 BLACKOUT_MAIN=$HOME/rv25main
export BLACKOUT_CKPTS="p14b=$HOME/p14frozen/p14b.pt,${EXTRA_CKPTS}"
.venv/bin/python examples/rollout_net_strength.py sim --candidates $CANDS --opponents $OPPS --h2h ${H2H} \
   --seeds ${SEEDS:-256} --chunk ${CHUNK:-32} --workers ${WORKERS:-6} --device cuda --ref ${REF:-f3x} --out $OUT
