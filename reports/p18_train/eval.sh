#!/bin/bash
R=$HOME/blackout-env-p18; B=$HOME/p14runs/p18; OUT=$R/reports/p18_sim
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 BLACKOUT_NETS=p18=$HOME/p14frozen/p18.pt,p17=$HOME/p14frozen/p17.pt,p16=$HOME/p14frozen/p16.pt,p14b=$HOME/p14frozen/p14b.pt
while kill -0 $(cat $B/run.pid) 2>/dev/null; do sleep 30; done
grep -q "done:" $B/stdout.log && [ ! -e $HOME/p14frozen/p18.pt ] && cp $B/latest.pt $HOME/p14frozen/p18.pt
[ -e $HOME/p14frozen/p18.pt ] || { echo "NO P18 CHECKPOINT" >> $OUT/stdout.log; exit 1; }
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
.venv/bin/python examples/v27_eval.py play --ours "p18;p17;$V29" \
  --opponent "$V29;p17;$V28;strategic_v27:reserve_stage=1.5;strategic_v25;p14b;strategic_v23" --seeds 64 --seed0 1100000 \
  --cap-s 425 --stop-exhausted --device cuda --chunk 32 --workers 9 --out $OUT >> $OUT/stdout.log 2>&1
.venv/bin/python examples/v27_eval.py report --base p17 --out $OUT >> $OUT/stdout.log 2>&1
echo "P18 EVAL DONE" >> $OUT/stdout.log
