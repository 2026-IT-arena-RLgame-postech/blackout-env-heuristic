#!/bin/bash
# docs §49 leaf V refit for r20d256 on poscat_gpu2 (the §39.6 / §24 recipe): branch data with r20d256 as the
# continuation policy of both teams (root opponents self 0.5 / P20 0.25 / v28d256 0.25), then 4 fitted-value
# iterations x 2500 steps from s2v_f3xd2/iter2.  CPU: PROCS processes x THREADS threads (count them in the budget).
# usage: setsid nohup nice -n 39 bash reports/r3_rollout/scripts/value.sh > ~/r3runs/logs/value.log 2>&1 < /dev/null &
set -e
cd ~/blackout-env-r3
PY=.venv/bin/python
T=${THREADS:-4}
export PYTHONPATH=. OMP_NUM_THREADS=$T NUMBA_NUM_THREADS=$T CUDA_VISIBLE_DEVICES=${GPU:-0}
R=~/r3runs; L=$R/logs; D=$R/value_r20d256; O=$PWD/checkpoints/value/s2v_r20d256
NET=$R/ck/r20d256.pt; P20=$PWD/checkpoints/distill/frozen/p20_v30.pt; V28D=$PWD/checkpoints/rollout_net/v28/v28d256.pt
MIX="self:0.5,net:$P20:0.25,rnet:$V28D:0.25"
echo $$ > $R/value.pid
PIDS=""
for s0 in ${SEED0S:-6500000 6600000 6700000}; do
  $PY examples/value_net_branch_data.py --net $NET --mix "$MIX" --seeds ${SEEDS:-256} --seed0 $s0 --threads $T --device cuda --out $D > $L/vdata_$s0.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
echo "$(date +%T) data done"
$PY examples/value_branch_train.py train --data $D --out-dir $O --init $PWD/checkpoints/value/s2v_f3xd2/iter2.pt \
   --shard-frac 0 --iters 4 --steps 2500 --group-roots 16 --rank-weight 2.0 --final-beta 10.0 --amp fp16 --packed --device cuda \
   > $L/vtrain.log 2>&1
echo "$(date +%T) VALUE DONE"
