#!/bin/bash
# docs §31 Q2 on wsl_gpu: the §22 recipe with P14b as the teacher, from an existing map-token net.
#   stage 1: P14b's Unity collection shards (act:strategic_v25 labels) + KL to P14b 1.0, 30k steps lr 2e-3
#   DAgger : the stage-1 net drives in the batched simulator, P14b labels both teams' views (GPU)
#   stage 2: 20k steps from stage 1, 50 % DAgger rows, lr 1e-3, KL 0.5
# usage: bash chain.sh NAME INIT.pt   (detached: setsid -f bash -c 'echo $$ > pid; exec bash chain.sh ...')
set -e
cd ~/blackout-env-rv25
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4
N=$1; INIT=$2
R=~/rv25runs; CK=$R/ck; L=$R/logs; D=$R/data
P14B=$HOME/p14frozen/p14b.pt; P11=$HOME/p14assets/checkpoints/distill/frozen/p11_v22.pt; F3X=$HOME/rv25assets/ck/f3_x_d2.pt
mkdir -p $CK $L
TR="examples/rollout_net_train.py train --data $D --amp fp16 --compile --eval-every 10000"
[ -f $CK/${N}_s1.pt.done ] || { $PY $TR --out $CK/${N}_s1.pt --init $INIT --steps 30000 --lr 2e-3 --p11-kl 1.0 > $L/${N}_s1.log 2>&1 && touch $CK/${N}_s1.pt.done; }
POOL="{\"self\": 0.3, \"teacher\": 0.25, \"net:$P11\": 0.1, \"small:$F3X\": 0.1, \"heur:strategic_v25\": 0.15, \"heur:strategic_v23\": 0.1}"
[ -f $R/dagger_$N/train.npz ] || $PY examples/rollout_net_collect.py --net $CK/${N}_s1.pt --teacher net:$P14B --pool "$POOL" \
   --games ${GAMES:-1536} --per-job 128 --workers ${WORKERS:-4} --device cuda --teacher-frac 0.1 --seed0 5100000 \
   --out $R/dagger_$N > $L/${N}_dagger.log 2>&1
[ -f $CK/${N}.pt.done ] || { $PY $TR --out $CK/${N}.pt --init $CK/${N}_s1.pt --steps 20000 --lr 1e-3 --p11-kl 0.5 \
   --extra $R/dagger_$N/train.npz --extra-frac 0.5 > $L/${N}_s2.log 2>&1 && touch $CK/${N}.pt.done; }
echo CHAIN DONE $N
