#!/bin/bash
# P20: Hunters at decision 99 (cap 5 s) for every kept snapshot of a pilot run, vs V28 / P17 / V30, 32 seeds from 1700000
# (not the eval seeds).  Usage: screen_h99.sh <run name, e.g. a>; out reports/p20_train/screen_<name>; waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p18b; B=$HOME/p14runs/p20_$1; OUT=$R/reports/p20_train/screen_$1
cd $R; export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
NETS=p17=$HOME/p14frozen/p17.pt; OURS=""
for f in $B/v_*.pt; do n=$1$(basename $f .pt | sed 's/v_0*/v/'); NETS=$NETS,$n=$f; OURS="$OURS;$n"; done
export BLACKOUT_NETS=$NETS
.venv/bin/python reports/p20_diag/scripts/diag.py play --ours "${OURS#;}" --opponent "V28;p17;V30" --seeds 32 --seed0 1700000 --cap-s 5 \
  --workers ${EW:-11} --chunk 16 --device cuda --out $OUT
.venv/bin/python reports/p20_diag/scripts/diag.py summary --out $OUT 2>/dev/null | grep -v "^ " > $OUT/summary.txt
