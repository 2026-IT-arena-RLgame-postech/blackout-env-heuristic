#!/bin/bash
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=2
R=$1; TAG=rand$R; OUT=reports/rollout_net_arch/qeval/$TAG
for opp in r5 p11; do
  $PY examples/s3_search_eval.py play --qhead checkpoints/rollout_net_arch/q6.pt --q-random $R --opponent $opp --seeds ${SEEDS:-128} \
     --seed0 2100000 --tag $TAG --device cuda --out $OUT
done
$PY examples/s3_search_eval.py report --out $OUT
