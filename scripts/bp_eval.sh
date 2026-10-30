#!/bin/bash
# Balance-patch heuristic screening: every candidate in CANDS (';'-joined descriptors, file) vs each opponent,
# 32 seeds x both sides (sim, CPU).  Usage: scripts/bp_eval.sh CANDS_FILE OUT_DIR WORKERS [SEEDS] [OPPONENTS...]
set -u
cands=$(cat "$1"); out=$2; workers=$3; seeds=${4:-32}; shift 4 2>/dev/null || shift $#
opps=("$@"); [ ${#opps[@]} -gt 0 ] || opps=(V33bp V32.1 V30 V19)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
for opp in "${opps[@]}"; do
  nice -n 19 .venv/bin/python examples/econ_stats.py play --ours "$cands" --opponent "$opp" --seeds "$seeds" \
    --seed0 9200000 --workers "$workers" --out "$out/$opp"
done
echo EVAL_DONE "$out"
