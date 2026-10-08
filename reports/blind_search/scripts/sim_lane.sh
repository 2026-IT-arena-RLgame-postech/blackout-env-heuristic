#!/bin/bash
# One sim lane on poscat_gpu2 (docs §34): sim_lane.sh GPU NAME SEEDS TAGS OPP1,OPP2,...
# Plays every opponent in order (blind_search_eval.py play skips rows already there).
# Launch detached: setsid nohup bash sim_lane.sh 1 a 128 plain,det,o4f v23,p11 </dev/null >lane.log 2>&1 &
set -u
GPU=$1; NAME=$2; SEEDS=$3; TAGS=$4; OPPS=$5
cd "$(dirname "$0")/../../.."
OUT=reports/blind_search/sim
mkdir -p $OUT/logs
echo $$ > $OUT/logs/lane_$NAME.pid
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o $TAGS"
  ~/blackout-env/.venv/bin/python examples/blind_search_eval.py play --opponent $o --tags $TAGS --seeds $SEEDS \
      --out $OUT --device cuda ${EXTRA:-} || echo "FAILED $o"
done
echo "== $(date +%T) lane $NAME done"
rm -f $OUT/logs/lane_$NAME.pid
