#!/bin/bash
# docs §54 R4 on poscat_gpu2: R3's §49 rollout-net recipe (reports/r3_rollout/scripts/run.sh) re-run with P21e as the
# teacher, d256, warm-started from r20d256 (the P20-distilled net).  Only the teacher, the init and the seed ranges change.
#   data   : P21e-driven sim games (--net none: team A = P21e, team B = the pool), labelled by P21e,
#            1536 x 150 s every 3rd decision (train) + 512 x every-8 (val), seeds 5700000+ / 5790000+
#   stage 1: 30k steps lr 2e-3 OneCycle, KL to P21e 1.0, init = r20d256
#   DAgger : the stage-1 net drives (teacher-frac 0.1), P21e labels both teams' views (GPU), seeds 5710000+
#   stage 2: 20k steps from stage 1, 50 % DAgger rows, lr 1e-3, KL 0.5
# Each stage aborts the chain on a NaN loss (fp16 loss scale, §39/§49); rerun the stage standalone then.
# usage (gpu2, clone ~/blackout-env-r4): GPU=2 setsid nohup nice -n 39 bash reports/r4_rollout/scripts/run.sh \
#            > ~/r4runs/logs/run.log 2>&1 < /dev/null &
set -e
cd ~/blackout-env-r4
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-2}
R=~/r4runs; CK=$R/ck; L=$R/logs; D=$R/data
T=$PWD/checkpoints/distill/frozen/p21e_v30.pt; P17=$PWD/checkpoints/distill/frozen/p17_v28.pt
P16=$PWD/checkpoints/distill/frozen/p16_v27.pt; P14B=$PWD/checkpoints/distill/frozen/p14_v25.pt
INIT=$PWD/checkpoints/rollout_net/r3/r20d256.pt
W=${WORKERS:-8}
mkdir -p $CK $L $D
echo $$ > $R/run.pid
V30="heur:strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28="heur:strategic_v28:reserve_stage=0.5,trade_hold=advance"
HEUR="\"$V30\": 0.15, \"$V28\": 0.1, \"heur:strategic_v27:reserve_stage=1.5\": 0.05, \"heur:strategic_v25\": 0.05, \"heur:strategic_v23\": 0.05"
POOL1="{\"teacher\": 0.35, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
POOL2="{\"self\": 0.25, \"teacher\": 0.2, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
collect() {  # out-dir log args...
  local out=$1 log=$2; shift 2
  $PY examples/rollout_net_collect.py "$@" --teacher net:$T --per-job ${PERJOB:-128} --workers $W --device cuda \
      --out $out > $L/$log.log 2>&1
}
# NaN check + divergence check: R4's first stage 1 blew up at step 16k without a NaN loss (fp16 loss scale -> 1e-4,
# loss 1e4, acc 0.15), so the last logged train acc must also stay >= 0.8
ok() { ! grep -q '"loss": NaN' $L/$1.log && grep -o '"acc": [0-9.]*' $L/$1.log | tail -1 | awk '{exit !($2 >= 0.8)}'; }
TR="examples/rollout_net_train.py train --data $D --amp fp16 --compile --eval-every 10000 --seed ${SEED:-0}"
[ -f $D/val.npz ] || { PERJOB=64 collect $R/s1val s1val --net none --pool "$POOL1" --games 512 --every 8 --seed0 5790000 \
                       && mv $R/s1val/train.npz $D/val.npz; }
echo "$(date +%T) val done"
[ -f $D/train.npz ] || { collect $R/s1train s1train --net none --pool "$POOL1" --games 1536 --every 3 --seed0 5700000 \
                         && mv $R/s1train/train.npz $D/train.npz; }
echo "$(date +%T) s1 data done"
[ -f $CK/r21d256_s1.pt.done ] || { $PY $TR --out $CK/r21d256_s1.pt --init $INIT --steps ${S1:-30000} --lr 2e-3 --p11-kl 1.0 \
     > $L/r21d256_s1.log 2>&1 && ok r21d256_s1 && touch $CK/r21d256_s1.pt.done; } || { echo "stage 1 failed (NaN?)"; exit 1; }
echo "$(date +%T) s1 done"
[ -f $R/dagger/train.npz ] || collect $R/dagger dagger --net $CK/r21d256_s1.pt --pool "$POOL2" --games 1536 \
     --teacher-frac 0.1 --seed0 5710000
echo "$(date +%T) dagger done"
[ -f $CK/r21d256.pt.done ] || { $PY $TR --out $CK/r21d256.pt --init $CK/r21d256_s1.pt --steps ${S2:-20000} --lr 1e-3 --p11-kl 0.5 \
     --extra $R/dagger/train.npz --extra-frac 0.5 > $L/r21d256_s2.log 2>&1 && ok r21d256_s2 && touch $CK/r21d256.pt.done; } || { echo "stage 2 failed (NaN?)"; exit 1; }
echo "$(date +%T) CHAIN DONE"
