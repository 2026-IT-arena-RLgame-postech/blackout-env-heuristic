#!/bin/bash
# One sim lane of the R4 search gate (docs §54) on poscat_gpu2, after reports/r3_search/scripts/lane.sh:
#   lane.sh GPU NAME BASE SEEDS TAGS OPP1,OPP2,...   (rows -> reports/r4_search/sim_<BASE>, fresh seeds 1800000+)
# plays every (tag, opponent) in order; blind_search_eval.py play skips rows already there.
# B1 pipeline (fp32, one process per GPU): --pipeline 1 --pipeline-graphs is always on.
# launch: setsid nohup nice -n 39 bash lane.sh 2 a p21e 64 plain,o4f_r21 v30,v28 </dev/null >lane.log 2>&1 &
set -u
GPU=$1; NAME=$2; BASE=$3; SEEDS=$4; TAGS=$5; OPPS=$6
cd "$(dirname "$0")/../../.."
OUT=reports/r4_search/sim_$BASE
mkdir -p $OUT/logs
echo $$ > $OUT/logs/lane_$NAME.pid
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o $TAGS"
  .venv/bin/python examples/blind_search_eval.py play --base $BASE --opponent $o --tags $TAGS --seeds $SEEDS \
      --seed0 ${SEED0:-1800000} --out $OUT --device cuda --pipeline 1 --pipeline-graphs ${EXTRA:-} || echo "FAILED $o"
done
echo "== $(date +%T) lane $NAME done"
rm -f $OUT/logs/lane_$NAME.pid
