#!/bin/bash
# One sim lane of the R3 search gate (docs §49) on poscat_gpu2, after reports/blind_search/scripts/sim_lane.sh:
#   lane.sh GPU NAME BASE SEEDS TAGS OPP1,OPP2,...   (rows -> reports/r3_search/sim_<BASE>, fresh seeds 1800000+)
# plays every (tag, opponent) in order; blind_search_eval.py play skips rows already there.
# GUARD=1: register this lane's process group with the gpu2 CPU guard (~/cpu_extra.pgids) while it runs.
# launch: setsid nohup nice -n 39 bash lane.sh 1 a p20 64 plain,o4f_r20 v30,v28 </dev/null >lane.log 2>&1 &
set -u
GPU=$1; NAME=$2; BASE=$3; SEEDS=$4; TAGS=$5; OPPS=$6
cd "$(dirname "$0")/../../.."
OUT=reports/r3_search/sim_$BASE
mkdir -p $OUT/logs
echo $$ > $OUT/logs/lane_$NAME.pid
G=$(ps -o pgid= -p $$ | tr -d ' ')
[ "${GUARD:-1}" = 1 ] && echo "$G" >> ~/cpu_extra.pgids
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o $TAGS"
  .venv/bin/python examples/blind_search_eval.py play --base $BASE --opponent $o --tags $TAGS --seeds $SEEDS \
      --seed0 ${SEED0:-1800000} --out $OUT --device cuda ${EXTRA:-} || echo "FAILED $o"
done
echo "== $(date +%T) lane $NAME done"
sed -i "/^$G\$/d" ~/cpu_extra.pgids
rm -f $OUT/logs/lane_$NAME.pid
