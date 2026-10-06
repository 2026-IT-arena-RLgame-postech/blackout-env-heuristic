#!/bin/bash
# docs §49 R3 redo after the NaN chain: DAgger with the (standalone) stage-1 net in a <= 20 min unguarded window
# (Q12's gpu2 gate pgid $Q12 SIGSTOPped for the window, SIGCONT right after and in an EXIT trap), then stage 2.
# usage: setsid nohup nice -n 39 bash reports/r3_rollout/scripts/dagger_s2.sh > ~/r3runs/logs/dagger_s2.log 2>&1 < /dev/null &
set -u
Q12=${Q12:-2739512}
cd ~/blackout-env-r3
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
R=~/r3runs; CK=$R/ck; L=$R/logs; D=$R/data
P20=$PWD/checkpoints/distill/frozen/p20_v30.pt; P17=$PWD/checkpoints/distill/frozen/p17_v28.pt
P16=$PWD/checkpoints/distill/frozen/p16_v27.pt; P14B=$PWD/checkpoints/distill/frozen/p14_v25.pt
V30="heur:strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28="heur:strategic_v28:reserve_stage=0.5,trade_hold=advance"
HEUR="\"$V30\": 0.15, \"$V28\": 0.1, \"heur:strategic_v27:reserve_stage=1.5\": 0.05, \"heur:strategic_v25\": 0.05, \"heur:strategic_v23\": 0.05"
POOL2="{\"self\": 0.25, \"teacher\": 0.2, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
cont() { kill -CONT -$Q12 2>/dev/null && echo "$(date +%T) CONT Q12 $Q12"; }
trap cont EXIT
[ -f $CK/r20d256_s1.pt.done ] || { echo "no stage-1 .done"; exit 1; }
if [ ! -f $R/dagger/train.npz ]; then
  kill -STOP -$Q12 2>/dev/null && echo "$(date +%T) STOP Q12 $Q12"
  timeout 1200 $PY examples/rollout_net_collect.py --net $CK/r20d256_s1.pt --teacher net:$P20 --pool "$POOL2" --games 1536 \
      --per-job 128 --workers ${WORKERS:-12} --device cuda --teacher-frac 0.1 --seed0 5610000 --out $R/dagger > $L/dagger.log 2>&1
  rc=$?
  cont
  [ $rc = 0 ] || { echo "$(date +%T) DAgger failed rc=$rc"; exit 1; }
fi
echo "$(date +%T) dagger done: $(tail -1 $L/dagger.log)"
$PY examples/rollout_net_train.py train --data $D --amp fp16 --compile --eval-every 10000 --out $CK/r20d256.pt \
    --init $CK/r20d256_s1.pt --steps ${S2:-20000} --lr 1e-3 --p11-kl 0.5 --extra $R/dagger/train.npz --extra-frac 0.5 \
    > $L/r20d256_s2.log 2>&1 || { echo "stage 2 failed"; exit 1; }
grep -q '"loss": NaN' $L/r20d256_s2.log && { echo "stage 2 NaN"; exit 1; }
touch $CK/r20d256.pt.done
echo "$(date +%T) S2 DONE"
