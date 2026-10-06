#!/bin/bash
# P20 eval (docs §47) on wsl_gpu: net NAME=PATH vs V28 / P17 / V30 / V29 (default), §42 settings, seeds 1100000+, base = P17 rows
# (reports/p19_sim/matches.jsonl copied in).  Usage: eval_net.sh <out> <name> <path.pt> [opponents]; waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p18b; OUT=$R/$1; N=$2; P=$3
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=$N=$P,p19=$HOME/p14frozen/p19.pt,p18bp2=$HOME/p14frozen/p18bp2.pt,p17=$HOME/p14frozen/p17.pt,p14b=$HOME/p14frozen/p14b.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
[ -e $OUT/matches.jsonl ] || cp reports/p19_sim/matches.jsonl $OUT/matches.jsonl
.venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 ${SEED0:-1100000} --cap-s 425 --stop-exhausted --device cuda --chunk 16 \
  --workers ${EW:-11} --out $OUT --ours $N --opponent "${4:-$V28;p17;$V30;$V29}"
.venv/bin/python examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
echo "EVAL DONE" >> $OUT/stdout.log
