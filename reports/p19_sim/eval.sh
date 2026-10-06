#!/bin/bash
# P19 eval (docs §45) on wsl_gpu: waits for the run, freezes ~/p14frozen/p19.pt, 64 seeds (1100000+) as §42, base p17.
R=$HOME/blackout-env-p18b; B=$HOME/p14runs/p19; OUT=$R/reports/p19_sim
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=p19=$HOME/p14frozen/p19.pt,p18bp2=$HOME/p14frozen/p18bp2.pt,p17=$HOME/p14frozen/p17.pt,p14b=$HOME/p14frozen/p14b.pt
while kill -0 $(cat $B/run.pid) 2>/dev/null; do sleep 30; done
grep -q "done:" $B/stdout.log && [ ! -e $HOME/p14frozen/p19.pt ] && cp $B/latest.pt $HOME/p14frozen/p19.pt
[ -e $HOME/p14frozen/p19.pt ] || { echo "NO P19 CHECKPOINT" >> $OUT/stdout.log; exit 1; }
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
PLAY=".venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 1100000 --cap-s 425 --stop-exhausted --device cuda --chunk 16 --workers 11 --out $OUT"
[ -e $OUT/matches.jsonl ] || cat reports/p18b_pilot2/matches.jsonl > $OUT/matches.jsonl  # P17 / P18 / V29 / P18bp2 rows
$PLAY --ours p19 --opponent "${OPPS:-$V30;p17;$V29;p18bp2;$V28;strategic_v27:reserve_stage=1.5;strategic_v25;p14b;strategic_v23}"
$PLAY --ours "p17;$V30" --opponent "$V30;p18bp2;p17"
.venv/bin/python examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
echo "P19 EVAL DONE" >> $OUT/stdout.log
