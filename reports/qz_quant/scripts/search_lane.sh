#!/bin/bash
# QZ in-search check (docs "QZ"): blind_search_eval o4f_v28r with base p20 (fp32) or p20c (fp16c), B1 pipeline + graphs,
# one 128-game chunk per opponent (64 fresh seeds x 2 sides) -> reports/qz_quant/search_<BASE>.
#   search_lane.sh GPU BASE OPP1,OPP2 [TAGS]   (GUARD=1: register the process group in ~/cpu_extra.pgids)
set -u
GPU=$1; BASE=$2; OPPS=$3; TAGS=${4:-o4f_v28r}
cd "$(dirname "$0")/../../.."
OUT=reports/qz_quant/search_$BASE
mkdir -p $OUT
G=$(ps -o pgid= -p $$ | tr -d ' ')
[ "${GUARD:-1}" = 1 ] && echo "$G" >> ~/cpu_extra.pgids
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
PY=${PY:-$HOME/blackout-env/.venv/bin/python}
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o $TAGS"
  /usr/bin/time -f "TIME wall %e user %U sys %S" $PY examples/blind_search_eval.py play --base $BASE --opponent $o --tags $TAGS \
      --seeds ${SEEDS:-64} --seed0 ${SEED0:-2100000} --chunk 128 --out $OUT --device cuda --pipeline 1 --pipeline-graphs \
      || echo "FAILED $o"
done
echo "== $(date +%T) lane done"
[ "${GUARD:-1}" = 1 ] && sed -i "/^$G\$/d" ~/cpu_extra.pgids
