#!/bin/bash
# usage: run_d1.sh WORKERS "ours|opp|shadows" ...
cd ~/d1
export PYTHONPATH=$HOME/d1 D1_FROZEN=$HOME/d1/frozen NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=$1; shift
for spec in "$@"; do
  IFS='|' read ours opp sh <<< "$spec"
  nice -n 10 ~/blackout-env/.venv/bin/python examples/d1_why.py play --ours "$ours" --opponent "$opp" --shadows "$sh" --seeds 32 --chunk 4 --workers $W --out ~/d1/out
done
echo ALLDONE
