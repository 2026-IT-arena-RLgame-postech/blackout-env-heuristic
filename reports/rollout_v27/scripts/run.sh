#!/bin/bash
# docs §37 Q8 on wsl_gpu: the §31 (Q2) recipe with P16 (V27 student) as the teacher, warm-started from v25*.
#   pack   : P16's Unity collection shards (act:strategic_v27:reserve_stage=1.5 labels) + P16 logits/offsets (KL targets)
#   stage 1: 30k steps lr 2e-3 OneCycle, KL to P16 1.0, init = v25f3x / v25d256
#   DAgger : the stage-1 net drives in the batched simulator, P16 labels both teams' views (GPU)
#   stage 2: 20k steps from stage 1, 50 % DAgger rows, lr 1e-3, KL 0.5
# order (RAM, 23 GB): pack; f3x s1; d256 s1 || f3x DAgger; f3x s2 || d256 DAgger (3 workers); d256 s2
# usage (detached): setsid -f bash -c 'echo $$ > ~/rv27runs/run.pid; exec bash reports/rollout_v27/scripts/run.sh' > ~/rv27runs/logs/run.log 2>&1
set -e
cd ~/blackout-env-q8
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4
R=~/rv27runs; CK=$R/ck; L=$R/logs; D=$R/data
P16=$HOME/p14frozen/p16.pt; P14B=$HOME/p14frozen/p14b.pt; P11=$HOME/p14assets/checkpoints/distill/frozen/p11_v22.pt
V25D=$HOME/rv25runs/ck/v25d256.pt; V25F=$HOME/rv25runs/ck/v25f3x.pt
mkdir -p $CK $L $D
[ -f $D/val.npz ] || BLACKOUT_P11=$P16 $PY examples/rollout_net_train.py pack --shards ~/p14runs/p16/shards --out $D \
    --p11 --p11-device cuda --workers 6 > $L/pack.log 2>&1
TR="examples/rollout_net_train.py train --data $D --amp fp16 --compile --eval-every 10000"
s1() { [ -f $CK/${1}_s1.pt.done ] || { $PY $TR --out $CK/${1}_s1.pt --init $2 --steps ${S1:-30000} --lr 2e-3 --p11-kl 1.0 \
         > $L/${1}_s1.log 2>&1 && touch $CK/${1}_s1.pt.done; }; }
POOL="{\"self\": 0.25, \"teacher\": 0.2, \"net:$P14B\": 0.1, \"net:$P11\": 0.05, \"small:$V25D\": 0.05, \
\"heur:strategic_v27:reserve_stage=1.5\": 0.15, \"heur:strategic_v25\": 0.1, \"heur:strategic_v23\": 0.1}"
dag() { [ -f $R/dagger_$1/train.npz ] || $PY examples/rollout_net_collect.py --net $CK/${1}_s1.pt --teacher net:$P16 --pool "$POOL" \
         --games ${GAMES:-1536} --per-job 128 --workers $3 --device cuda --teacher-frac 0.1 --seed0 $2 \
         --out $R/dagger_$1 > $L/${1}_dagger.log 2>&1; }
s2() { [ -f $CK/${1}.pt.done ] || { $PY $TR --out $CK/${1}.pt --init $CK/${1}_s1.pt --steps ${S2:-20000} --lr 1e-3 --p11-kl 0.5 \
         --extra $R/dagger_$1/train.npz --extra-frac 0.5 > $L/${1}_s2.log 2>&1 && touch $CK/${1}.pt.done; }; }
s1 v27f3x $V25F
echo "$(date +%T) f3x s1 done"
s1 v27d256 $V25D & P1=$!
dag v27f3x 5300000 4
wait $P1
echo "$(date +%T) d256 s1 + f3x dagger done"
s2 v27f3x & P2=$!
dag v27d256 5310000 3
wait $P2
echo "$(date +%T) f3x s2 + d256 dagger done"
s2 v27d256
echo "$(date +%T) CHAIN DONE"
