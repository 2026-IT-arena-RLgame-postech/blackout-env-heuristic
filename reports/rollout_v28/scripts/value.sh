#!/bin/bash
# docs §39.6 leaf V refit for v28d256 on wsl_gpu (the §24 recipe): branch data with v28d256 as the continuation
# policy of both teams, then 4 fitted-value iterations x 2500 steps from s2v_f3xd2/iter2.
# usage (detached): setsid -f bash -c 'echo $$ > ~/rv28runs/value.pid; exec bash reports/rollout_v28/scripts/value.sh' > ~/rv28runs/logs/value.log 2>&1
set -e
cd ~/blackout-env-q9
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=5 NUMBA_NUM_THREADS=5
R=~/rv28runs; L=$R/logs; D=$R/value_v28d256; O=$R/v_v28d256
NET=$R/ck/v28d256.pt; P17=$HOME/p14frozen/p17.pt; V27D=$HOME/rv27runs/ck/v27d256.pt
MIX="self:0.5,net:$P17:0.25,rnet:$V27D:0.25"
# roots: seeds 6200000+ / 6300000+ (both sides), two processes
$PY examples/value_net_branch_data.py --net $NET --mix "$MIX" --seeds ${SEEDS:-256} --seed0 6200000 --threads 5 --device cuda --out $D > $L/vdata_a.log 2>&1 &
A=$!
$PY examples/value_net_branch_data.py --net $NET --mix "$MIX" --seeds ${SEEDS:-256} --seed0 6300000 --threads 5 --device cuda --out $D > $L/vdata_b.log 2>&1 &
B=$!
wait $A; wait $B
echo "$(date +%T) data done"
$PY examples/value_branch_train.py train --data $D --out-dir $O --init $HOME/rv25assets/ck/s2v_f3xd2/iter2.pt \
   --shard-frac 0 --iters 4 --steps 2500 --group-roots 16 --rank-weight 2.0 --final-beta 10.0 --amp fp16 --packed --device cuda \
   > $L/vtrain.log 2>&1
echo "$(date +%T) VALUE DONE"
