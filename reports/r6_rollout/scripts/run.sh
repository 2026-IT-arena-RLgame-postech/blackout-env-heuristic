#!/bin/bash
# docs §61 R6 on poscat_gpu2: R5's §59 rollout-net recipe (reports/r5_rollout/scripts/run.sh, fp32) re-run with P24c (v9, V32.1 teacher) as the
# teacher, d256, warm-started from r22d256 (the P22-distilled net).  Teacher, init, seeds change; V32 B, V32.1, h4 join the pools.
#   data   : P24c-driven sim games (--net none: team A = P24c, team B = the pool), labelled by P24c,
#            1536 x 150 s every 3rd decision (train) + 512 x every-8 (val), seeds 5900000+ / 5990000+
#   stage 1: 30k steps lr 2e-3 OneCycle, KL to P24c 1.0, init = r22d256
#   DAgger : the stage-1 net drives (teacher-frac 0.1), P24c labels both teams' views (GPU), seeds 5910000+
#   stage 2: 20k steps from stage 1, 50 % DAgger rows, lr 1e-3, KL 0.5
# Each stage aborts the chain on a NaN loss (fp16 loss scale, §39/§49); rerun the stage standalone then.
# fp32 by default (AMP=none): R5's fp16 stage 1 diverged on seeds 0 and 1 (§59).
# usage (gpu2, clone ~/blackout-env-r5): GPU=2 setsid nohup nice -n 39 bash reports/r6_rollout/scripts/run.sh \
#            > ~/r6runs/logs/run.log 2>&1 < /dev/null &
set -e
cd ~/blackout-env-r5  # the R5 clone (has checkpoints/rollout_net/r5)
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-2}
R=~/r6runs; CK=$R/ck; L=$R/logs; D=$R/data
T=$PWD/checkpoints/distill/frozen/p24c_v321.pt; P17=$PWD/checkpoints/distill/frozen/p17_v28.pt
P16=$PWD/checkpoints/distill/frozen/p16_v27.pt; P14B=$PWD/checkpoints/distill/frozen/p14_v25.pt
INIT=$PWD/checkpoints/rollout_net/r5/r22d256.pt
W=${WORKERS:-8}
mkdir -p $CK $L $D
echo $$ > $R/run.pid
V30="heur:strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28="heur:strategic_v28:reserve_stage=0.5,trade_hold=advance"
# R6: the V32 family (4 Hunters + sweep) and h4 join both pools so the opponent model sees them (§59: search
# fell back to the student against V32 B / h4, whose behaviour r22d256 had never seen)
V32B="heur:strategic_v32:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5,hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,sweep=true,cargo_transform=true"
H4="heur:strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,hunter_quota=4"
NEW="\"$V32B\": 0.05, \"$V32B,adapt_k=3\": 0.1, \"$H4\": 0.05"
HEUR="$NEW, \"$V30\": 0.15, \"$V28\": 0.1, \"heur:strategic_v27:reserve_stage=1.5\": 0.05, \"heur:strategic_v25\": 0.05, \"heur:strategic_v23\": 0.05"
POOL1="{\"teacher\": 0.35, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
POOL2="{\"self\": 0.25, \"teacher\": 0.2, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
collect() {  # out-dir log args...
  local out=$1 log=$2; shift 2
  $PY examples/rollout_net_collect.py "$@" --teacher net:$T --per-job ${PERJOB:-128} --workers $W --device cuda \
      --out $out > $L/$log.log 2>&1
}
# NaN check + divergence check: R4's first stage 1 blew up at step 16k without a NaN loss (fp16 loss scale -> 1e-4,
# loss 1e4, acc 0.15), so the last logged train acc must also stay >= 0.8.  R5: seed 1's stage 1 kept acc 0.92 but its
# KL term blew up at step 23.5k (loss 0.65 -> 9-17, loss scale 128 -> 2e-4, val agreement 0.80 -> 0.74), so the last
# logged loss must also stay <= 2x the lowest one.
ok() { ! grep -q '"loss": NaN' $L/$1.log && grep -o '"acc": [0-9.]*' $L/$1.log | tail -1 | awk '{exit !($2 >= 0.8)}' &&
       grep -o '"loss": [0-9.e+-]*' $L/$1.log | awk '{v=$2+0; if (NR==1||v<m) m=v; l=v} END {exit !(l <= 2*m)}'; }
TR="examples/rollout_net_train.py train --data $D --amp ${AMP:-none} --compile --eval-every 10000 --seed ${SEED:-0}"
[ -f $D/val.npz ] || { PERJOB=64 collect $R/s1val s1val --net none --pool "$POOL1" --games 512 --every 8 --seed0 5990000 \
                       && mv $R/s1val/train.npz $D/val.npz; }
echo "$(date +%T) val done"
[ -f $D/train.npz ] || { collect $R/s1train s1train --net none --pool "$POOL1" --games 1536 --every 3 --seed0 5900000 \
                         && mv $R/s1train/train.npz $D/train.npz; }
echo "$(date +%T) s1 data done"
[ -f $CK/r24cd256_s1.pt.done ] || { $PY $TR --out $CK/r24cd256_s1.pt --init $INIT --steps ${S1:-30000} --lr 2e-3 --p11-kl 1.0 \
     > $L/r24cd256_s1.log 2>&1 && ok r24cd256_s1 && touch $CK/r24cd256_s1.pt.done; } || { echo "stage 1 failed (NaN?)"; exit 1; }
echo "$(date +%T) s1 done"
[ -f $R/dagger/train.npz ] || collect $R/dagger dagger --net $CK/r24cd256_s1.pt --pool "$POOL2" --games 1536 \
     --teacher-frac 0.1 --seed0 5910000
echo "$(date +%T) dagger done"
[ -f $CK/r24cd256.pt.done ] || { $PY $TR --out $CK/r24cd256.pt --init $CK/r24cd256_s1.pt --steps ${S2:-20000} --lr 1e-3 --p11-kl 0.5 \
     --extra $R/dagger/train.npz --extra-frac 0.5 > $L/r24cd256_s2.log 2>&1 && ok r24cd256_s2 && touch $CK/r24cd256.pt.done; } || { echo "stage 2 failed (NaN?)"; exit 1; }
echo "$(date +%T) CHAIN DONE"
