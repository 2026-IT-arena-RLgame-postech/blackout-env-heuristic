#!/bin/bash
# P22 (docs §55) snapshot screen on wsl_gpu (= reports/p21_train/screen.sh with base P21e and V31 added):
# (1) Hunters at decision 99 (cap 5 s) vs V28 / P17 / V30 / V31, 32 seeds from 1700000 x both sides;
# (2) short full matches vs V28 / P17 / V25 / V31, 16 seeds from 1800000 (not the gate seeds), paired with P21e.
# Usage: screen.sh <run name>; out reports/p22_train/screen_<name>; waits for $WAIT_PID.  Snapshot v_0001 = the init (skipped).
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p22; B=$HOME/p14runs/p22_$1; OUT=$R/reports/p22_train/screen_$1
cd $R; mkdir -p $OUT; export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
NETS=p17=$HOME/p14frozen/p17.pt,p21e=$HOME/p14frozen/p21e.pt; OURS="p21e"
for f in $B/v_000[2-5].pt; do n=$1$(basename $f .pt | sed 's/v_0*/v/'); NETS=$NETS,$n=$f; OURS="$OURS;$n"; done
export BLACKOUT_NETS=$NETS
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V31="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5"
.venv/bin/python reports/p20_diag/scripts/diag.py play --ours "$OURS" --opponent "V28;p17;V30;$V31" --seeds 32 --seed0 1700000 --cap-s 5 \
  --workers ${EW:-11} --chunk 16 --device cuda --out $OUT/h99
.venv/bin/python reports/p20_diag/scripts/diag.py summary --out $OUT/h99 2>/dev/null | grep -v "^ " > $OUT/h99.txt
.venv/bin/python examples/v27_eval.py play --seeds ${SS:-16} --seed0 1800000 --cap-s 425 --stop-exhausted --device cuda --chunk 16 \
  --workers ${EW:-11} --out $OUT/full --ours "$OURS" --opponent "${SOPP:-$V28;p17;strategic_v25;$V31}"
.venv/bin/python examples/v27_eval.py report --base p21e --out $OUT/full > $OUT/full.txt
echo "SCREEN DONE" >> $OUT/stdout.log
