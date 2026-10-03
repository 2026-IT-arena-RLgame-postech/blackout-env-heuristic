#!/bin/bash
# paired sim eval of a Q-head policy (r5 + learned alt switch) vs plain r5, seeds 2100000+ (as r6 evals)
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=2
Q=$1; M=$2; TAG=${Q}_m$M; OUT=reports/rollout_net_arch/qeval/$TAG
for opp in r5 p11 ${EXTRA_OPP:-}; do
  $PY examples/s3_search_eval.py play --qhead checkpoints/rollout_net_arch/$Q.pt --margin $M --opponent $opp --seeds ${SEEDS:-128} \
     --seed0 2100000 --tag $TAG --device cuda --out $OUT
done
$PY examples/s3_search_eval.py report --out $OUT
echo QEVAL DONE
