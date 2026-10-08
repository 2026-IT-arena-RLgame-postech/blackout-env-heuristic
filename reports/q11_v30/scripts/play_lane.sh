#!/bin/bash
# Q11 recording lane (docs §44): play_lane.sh GPU OPP SEEDS [TAGS]  -- plain + search (o4f_v28r) vs one opponent.
# Launch detached: setsid nohup bash reports/q11_v30/scripts/play_lane.sh 1 v28 64 </dev/null >lane.log 2>&1 &
set -u
GPU=$1; OPP=$2; SEEDS=$3; TAGS=${4:-plain,search}
cd "$(dirname "$0")/../../.."
mkdir -p reports/q11_v30/logs
echo $$ > reports/q11_v30/logs/lane_$OPP.pid
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for t in ${TAGS//,/ }; do
  echo "== $(date +%T) $t vs $OPP"
  nice -n 10 ~/blackout-env/.venv/bin/python examples/q11_why.py play --tag $t --opponent $OPP --seeds $SEEDS \
      --device cuda ${EXTRA:-} || echo "FAILED $t $OPP"
done
echo "== $(date +%T) lane $OPP done"
rm -f reports/q11_v30/logs/lane_$OPP.pid
