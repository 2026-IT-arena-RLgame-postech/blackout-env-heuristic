#!/bin/bash
# P21 (docs §50) gate on wsl_gpu (or gpu1 CPU with DEV=cpu): candidate NAME=PATH and the base P20 on fresh seeds
# (default 64 from 1900000) vs the pool V30, V29, P17, P18bp2, V27(1.5), V28, P14b, V25, V23; report --base p20.
# Usage: eval.sh <out dir under reports/> <name> <path.pt>; env SEEDS, SEED0, EW, DEV, OPPS, NETDIR (frozen nets dir); waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=${R:-$HOME/blackout-env-p21}; OUT=$R/reports/$1; N=$2; P=$3; F=${NETDIR:-$HOME/p14frozen}; DEV=${DEV:-cuda}
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=$N=$P,p20=$F/p20.pt,p18bp2=$F/p18bp2.pt,p17=$F/p17.pt,p14b=$F/p14b.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
OPP="${OPPS:-$V28;p17;$V30;$V29;p18bp2;strategic_v27:reserve_stage=1.5;p14b;strategic_v25;strategic_v23}"
for o in $(echo "$OPP" | tr ";" " "); do
  .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 ${SEED0:-1900000} --cap-s 425 --stop-exhausted --device $DEV \
    --chunk ${CHUNK:-16} --workers ${EW:-11} --out $OUT --ours "$N;p20" --opponent "$o"
  .venv/bin/python examples/v27_eval.py report --base p20 --out $OUT > $OUT/report.txt
done
echo "EVAL DONE" >> $OUT/stdout.log
