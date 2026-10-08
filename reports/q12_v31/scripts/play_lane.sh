#!/bin/bash
# Q12 recording lane (docs §46): play_lane.sh GPU OPP SEEDS [SEED0 [TAGS]]  -- plain V30 + search (o4f_v28r) on top of V30.
# Launch detached: setsid nohup bash reports/q12_v31/scripts/play_lane.sh 1 v29 8 2100000 </dev/null >lane.log 2>&1 &
set -u
GPU=$1; OPP=$2; SEEDS=$3; SEED0=${4:-2100000}; TAGS=${5:-plain,search}
cd "$(dirname "$0")/../../.."
mkdir -p reports/q12_v31/logs
echo $$ > reports/q12_v31/logs/lane_${OPP}_$SEED0.pid
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for t in ${TAGS//,/ }; do
  echo "== $(date +%T) $t vs $OPP"
  nice -n 10 ~/blackout-env/.venv/bin/python examples/q11_why.py play --base v30 --tag $t --opponent $OPP --seeds $SEEDS --seed0 $SEED0 \
      --chunk ${CHUNK:-8} --device cuda --out reports/q12_v31 ${EXTRA:-} || echo "FAILED $t $OPP"
done
echo "== $(date +%T) lane $OPP done"
rm -f reports/q12_v31/logs/lane_${OPP}_$SEED0.pid
