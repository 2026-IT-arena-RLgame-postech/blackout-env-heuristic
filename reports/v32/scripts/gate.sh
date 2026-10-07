#!/bin/bash
# V32 (docs §58) sim runs on poscat_gpu2 CPU: OURS (';'-separated, aliases of names.sh) vs OPP, SEEDS x 2 sides
# from SEED0, cap 425 s, --stop-exhausted, nice 19 (ssh sessions start at -20), 1 thread each.
# Usage: gate.sh <out dir under reports/v32/> ; env OURS, OPP, SEEDS (64), SEED0 (5200000), EW (workers, 12).
R=${R:-$HOME/blackout-env-v32}; OUT=$R/reports/v32/$1
cd $R; mkdir -p $OUT
. reports/v32/scripts/names.sh
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
ours=$(expand "$OURS"); opp=$(expand "$OPP")
nice -n 39 .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 ${SEED0:-5200000} --cap-s 425 \
  --stop-exhausted --device cpu --chunk ${CHUNK:-8} --workers ${EW:-12} --out $OUT --ours "$ours" --opponent "$opp" \
  >> $OUT/stdout.log 2>&1
echo "GATE DONE" >> $OUT/stdout.log
