#!/bin/bash
# P21 (docs §50) snapshot screen on wsl_gpu: (1) Hunters at decision 99 (cap 5 s) vs V28 / P17 / V30, 32 seeds from 1700000;
# (2) a short full-match check vs V28 and P17, 16 seeds from 1800000 (neither = the gate seeds).  P20 is screened as the base.
# Usage: screen.sh <run name>; out reports/p21_train/screen_<name>; waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p21; B=$HOME/p14runs/p21_$1; OUT=$R/reports/p21_train/screen_$1
cd $R; mkdir -p $OUT; export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
NETS=p17=$HOME/p14frozen/p17.pt,p20=$HOME/p14frozen/p20.pt; OURS="p20"
for f in $B/v_*.pt; do n=$1$(basename $f .pt | sed 's/v_0*/v/'); NETS=$NETS,$n=$f; OURS="$OURS;$n"; done
export BLACKOUT_NETS=$NETS
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
.venv/bin/python reports/p20_diag/scripts/diag.py play --ours "$OURS" --opponent "V28;p17;V30" --seeds 32 --seed0 1700000 --cap-s 5 \
  --workers ${EW:-11} --chunk 16 --device cuda --out $OUT/h99
.venv/bin/python reports/p20_diag/scripts/diag.py summary --out $OUT/h99 2>/dev/null | grep -v "^ " > $OUT/h99.txt
.venv/bin/python examples/v27_eval.py play --seeds ${SS:-16} --seed0 1800000 --cap-s 425 --stop-exhausted --device cuda --chunk 16 \
  --workers ${EW:-11} --out $OUT/full --ours "$OURS" --opponent "${SOPP:-$V28;p17}"
.venv/bin/python examples/v27_eval.py report --base p20 --out $OUT/full > $OUT/full.txt
echo "SCREEN DONE" >> $OUT/stdout.log
