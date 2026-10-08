#!/bin/bash
# throughput of random-init variants: full BatchRollout loop, 2080 Ti
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=4
$PY examples/rollout_net_arch.py init --out checkpoints/rollout_net_arch/init
for v in "$@"; do
  $PY examples/batch_rollout_bench.py throughput --policy small:checkpoints/rollout_net_arch/init/$v.pt \
     --configs cuda:fp32 cuda:fp16 --sizes 1024 2048 --threads 8 --min-decisions 200000 --tag _${v}${TAGSUF:-} --out reports/rollout_net_arch/throughput
done
echo BENCH DONE
