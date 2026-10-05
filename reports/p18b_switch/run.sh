#!/bin/bash
# P18b: which part of pilot2 (opening <6 s / later ops) loses to V28?  Sim, 32 seeds (1100000+), switch policies (v27_eval.py).
R=$HOME/blackout-env-p18b; OUT=$R/reports/p18b_switch
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=p18bp2=$HOME/p14runs/p18b_pilot2/latest.pt,p17=$HOME/p14frozen/p17.pt
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
[ -e $OUT/matches.jsonl ] || grep -E '"ours": "(p17|p18bp2)"' reports/p18b_pilot2/matches.jsonl > $OUT/matches.jsonl
.venv/bin/python examples/v27_eval.py play --ours "${OURS:-switch:6:p18bp2|p17;switch:6:p17|p18bp2}" --opponent "${OPPS:-$V28}" \
  --seeds 32 --seed0 1100000 --cap-s 425 --stop-exhausted --device cuda --chunk 4 --workers 11 --out $OUT
.venv/bin/python examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
echo SWITCH DONE
