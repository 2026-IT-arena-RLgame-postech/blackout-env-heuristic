#!/bin/bash
# P18b pilot (5k steps) check on wsl_gpu: Hunters@99 (does it stop joining vs V23/V25/V27?) then 64 seeds vs V28/P17/V29/V23.
NAME=${NAME:-p18bp}; CK=${CK:-$HOME/p14runs/p18b_pilot/latest.pt}
R=$HOME/blackout-env-p18b; OUT=$R/reports/${OUTDIR:-p18b_pilot}
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=$NAME=$CK,p18=$HOME/p14frozen/p18.pt,p17=$HOME/p14frozen/p17.pt,p16=$HOME/p14frozen/p16.pt,p14b=$HOME/p14frozen/p14b.pt
PY=.venv/bin/python
$PY reports/p18b_diag/scripts/diag.py play --ours "$NAME" --opponent "V28;p17;V29;V23;V25;V27" --seeds 32 --cap-s 5 --workers 11 --chunk 16 --device cuda --out $OUT/h99
$PY reports/p18b_diag/scripts/diag.py summary --out $OUT/h99 > $OUT/h99/summary.txt 2>/dev/null
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
[ -e $OUT/matches.jsonl ] || grep -E '"ours": "(p17|p18)"' reports/p18_sim/matches.jsonl > $OUT/matches.jsonl
$PY examples/v27_eval.py play --ours "$NAME" --opponent "${OPPS:-$V28;p17;$V29;strategic_v23}" --seeds ${SEEDS:-64} --seed0 1100000 \
  --cap-s 425 --stop-exhausted --device cuda --chunk 16 --workers 11 --out $OUT
$PY examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
echo EVAL DONE
