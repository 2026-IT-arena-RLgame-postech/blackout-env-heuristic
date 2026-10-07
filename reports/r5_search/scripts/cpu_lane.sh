#!/bin/bash
# CPU-only plain-P22 lane (docs §59; poscat_gpu1, nice 10, no GPU): cpu_lane.sh NAME SET SEED0 SEEDS OPP1,OPP2,...
#   rows -> reports/r5_search/<SET>_p22 (merged with the gpu2 rows afterwards; play skips rows already there)
set -u
NAME=$1; SET=$2; S0=$3; SEEDS=$4; OPPS=$5
cd "$(dirname "$0")/../../.."
OUT=reports/r5_search/${SET}_p22
mkdir -p $OUT/logs
echo $$ > $OUT/logs/cpu_$NAME.pid
export CUDA_VISIBLE_DEVICES= PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o"
  .venv/bin/python examples/blind_search_eval.py play --base p22 --opponent $o --tags plain --seeds $SEEDS \
      --seed0 $S0 --out $OUT --device cpu --chunk 64 || echo "FAILED $o"
done
echo "== $(date +%T) cpu lane $NAME done"
rm -f $OUT/logs/cpu_$NAME.pid
