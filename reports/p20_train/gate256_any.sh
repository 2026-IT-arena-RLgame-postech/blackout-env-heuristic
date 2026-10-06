#!/bin/bash
# P20 256-seed gate, any host: R (clone), DEV (cpu/cuda), CHUNK, EW, OPPS; same as gate256_gpu1.sh otherwise.
# (= P19 v_0005) AND the base P17, 8 opponents.  The 64-seed rows (reports/p20_p19v5 on wsl) are merged on the Mac afterwards.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
R=${R:-$HOME/blackout-env-p20}; OUT=$R/reports/p20_gate256; N=$HOME/p20nets
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLACKOUT_NETS=p20=$N/p20.pt,p17=$N/p17.pt,p14b=$N/p14b.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
OPP="${OPPS:-$V28;p17;$V30;strategic_v25;p14b;strategic_v23;$V29;strategic_v27:reserve_stage=1.5}"
for o in $(echo "$OPP" | tr ";" " "); do
  for us in p20 p17; do
    .venv/bin/python examples/v27_eval.py play --seeds 192 --seed0 1100064 --cap-s 425 --stop-exhausted --device ${DEV:-cpu} --chunk ${CHUNK:-8} \
      --workers ${EW:-8} --out $OUT --ours $us --opponent "$o"
  done
done
echo "GATE DONE" >> $OUT/stdout.log
