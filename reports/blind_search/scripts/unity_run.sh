#!/bin/bash
# Unity confirmation on poscat_gpu2 (docs §34): unity_run.sh OUT TAGS DEEP_JSON SERVER_GPUS WORKERS [OPPS] [SEEDS]
# P14b acting (--base), search from 5 s, rollout f3_x_d2, V s2v_f3xd2 iter2; 'plain' = P14b alone.
# Launch detached: setsid nohup bash unity_run.sh ... </dev/null >log 2>&1 &
set -u
OUT=$1; TAGS=$2; DEEP=$3; SG=$4; W=$5; OPPS=${6:-p14b,v25,v23,p11}; SEEDS=${7:-150}
cd ~/blackout-env-r1
mkdir -p $OUT
echo $$ > $OUT/run.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P14B=checkpoints/distill/frozen/p14_v25.pt
OPPD=""
for o in ${OPPS//,/ }; do
  case $o in p14b) d=ckpt:$P14B;; v25) d=strategic_v25;; v27) d=strategic_v27:reserve_stage=1.5;; *) d=$o;; esac
  OPPD=${OPPD:+$OPPD,}$d
done
~/blackout-env/.venv/bin/python examples/unity_search_eval.py play --opponent "$OPPD" --seeds $SEEDS --seed0 4000000 \
    --tags $TAGS --cap-s 150 --search-from-s 5 --base $P14B --net ${NET:-checkpoints/rollout_net/arch/f3_x_d2.pt} \
    --value checkpoints/value/s2v_f3xd2/iter2.pt ${DEEP:+--deep "$DEEP"} --server spawn --server-gpus $SG \
    --workers $W --gpus "" --server-max-roots ${MAXR:-8} --out $OUT
echo "== $(date +%T) done"
rm -f $OUT/run.pid
