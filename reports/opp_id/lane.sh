#!/bin/bash
# docs §63 A/B: o4f_r24c vs o4f_id (opponent identification), P24c base, sim, 150 s cap, seeds 6200000+.
#   lane.sh GPU OPP1,OPP2,...
set -u
GPU=$1; OPPS=$2
cd ~/blackout-env
OUT=reports/opp_id/sim
mkdir -p $OUT/logs
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
for o in ${OPPS//,/ }; do
  echo "== $(date +%T) $o"
  nice -n 5 .venv/bin/python examples/blind_search_eval.py play --base p24c --opponent $o --tags o4f_r24c,o4f_id \
      --seeds 32 --seed0 6200000 --out $OUT --device cuda --chunk 64 || echo "FAILED $o"
done
echo "== $(date +%T) lane $GPU done"
