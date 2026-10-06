#!/bin/bash
# docs §49 R3 on poscat_gpu2: the §39 (Q9) rollout-net recipe re-run with P20 (v_0005) as the teacher, d256 only,
# warm-started from v28d256.  Q9's stage-1 data were P17's Unity collection shards; P20's shards live on wsl
# (reserved), so stage 1 uses simulator games that P20 drives (rollout_net_collect.py --net none: team A = P20,
# team B = a pool), labelled by P20 (arc = arg max +- 1, KL targets = its logits / offsets).
#   data   : P20-driven sim games, 1536 x 150 s (train) + 512 x every-8 (val), seeds 5600000+ / 5690000+
#   stage 1: 30k steps lr 2e-3 OneCycle, KL to P20 1.0, init = v28d256
#   DAgger : the stage-1 net drives (teacher-frac 0.1), P20 labels both teams' views (GPU), seeds 5610000+
#   stage 2: 20k steps from stage 1, 50 % DAgger rows, lr 1e-3, KL 0.5
# collection workers are CPU (nice 19); register their process group in ~/cpu_extra.pgids while Q12 runs (GUARD=1).
# usage: setsid nohup nice -n 39 bash reports/r3_rollout/scripts/run.sh > ~/r3runs/logs/run.log 2>&1 < /dev/null &
set -e
cd ~/blackout-env-r3
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
R=~/r3runs; CK=$R/ck; L=$R/logs; D=$R/data
P20=$PWD/checkpoints/distill/frozen/p20_v30.pt; P17=$PWD/checkpoints/distill/frozen/p17_v28.pt
P16=$PWD/checkpoints/distill/frozen/p16_v27.pt; P14B=$PWD/checkpoints/distill/frozen/p14_v25.pt
V28D=$PWD/checkpoints/rollout_net/v28/v28d256.pt
W=${WORKERS:-8}
mkdir -p $CK $L $D
echo $$ > $R/run.pid
V30="heur:strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28="heur:strategic_v28:reserve_stage=0.5,trade_hold=advance"
HEUR="\"$V30\": 0.15, \"$V28\": 0.1, \"heur:strategic_v27:reserve_stage=1.5\": 0.05, \"heur:strategic_v25\": 0.05, \"heur:strategic_v23\": 0.05"
POOL1="{\"teacher\": 0.35, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
POOL2="{\"self\": 0.25, \"teacher\": 0.2, \"net:$P17\": 0.1, \"net:$P16\": 0.05, \"net:$P14B\": 0.05, $HEUR}"
# collection in its own process group, registered with the gpu2 CPU guard while GUARD=1
collect() {  # out-dir log args...
  local out=$1 log=$2; shift 2
  setsid $PY examples/rollout_net_collect.py "$@" --teacher net:$P20 --per-job ${PERJOB:-128} --workers $W --device cuda \
      --out $out > $L/$log.log 2>&1 &
  local p=$!; sleep 2
  local g=$(ps -o pgid= -p $p | tr -d ' ')
  [ "${GUARD:-1}" = 1 ] && [ -n "$g" ] && echo "$g" >> ~/cpu_extra.pgids
  wait $p; local rc=$?
  [ -n "$g" ] && sed -i "/^$g\$/d" ~/cpu_extra.pgids
  return $rc
}
ok() { ! grep -q '"loss": NaN' $L/$1.log; }
TR="examples/rollout_net_train.py train --data $D --amp fp16 --compile --eval-every 10000"
[ -f $D/val.npz ] || { PERJOB=64 collect $R/s1val s1val --net none --pool "$POOL1" --games 512 --every 8 --seed0 5690000 \
                       && mv $R/s1val/train.npz $D/val.npz; }
echo "$(date +%T) val done"
[ -f $D/train.npz ] || { collect $R/s1train s1train --net none --pool "$POOL1" --games 1536 --seed0 5600000 \
                         && mv $R/s1train/train.npz $D/train.npz; }
echo "$(date +%T) s1 data done"
[ -f $CK/r20d256_s1.pt.done ] || { $PY $TR --out $CK/r20d256_s1.pt --init $V28D --steps ${S1:-30000} --lr 2e-3 --p11-kl 1.0 \
     > $L/r20d256_s1.log 2>&1 && ok r20d256_s1 && touch $CK/r20d256_s1.pt.done; }
echo "$(date +%T) s1 done"
[ -f $R/dagger/train.npz ] || collect $R/dagger dagger --net $CK/r20d256_s1.pt --pool "$POOL2" --games 1536 \
     --teacher-frac 0.1 --seed0 5610000
echo "$(date +%T) dagger done"
[ -f $CK/r20d256.pt.done ] || { $PY $TR --out $CK/r20d256.pt --init $CK/r20d256_s1.pt --steps ${S2:-20000} --lr 1e-3 --p11-kl 0.5 \
     --extra $R/dagger/train.npz --extra-frac 0.5 > $L/r20d256_s2.log 2>&1 && ok r20d256_s2 && touch $CK/r20d256.pt.done; }
echo "$(date +%T) CHAIN DONE"
