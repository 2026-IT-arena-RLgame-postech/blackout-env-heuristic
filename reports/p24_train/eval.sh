#!/bin/bash
# P24 (docs §60) gate on wsl_gpu: candidates (NAME=PATH,...) and the base P22 on fresh seeds (default 64 from 2200000)
# vs V32, V31, V30, P22 (= h2h), P21e, P20, P17, V28(0.5, advance), V29, V27(1.5), P14b, V25, V23 + OOD1 h4, h4hunt; report --base p22.
# Usage: eval.sh <out dir under reports/> "<n1>=<p1>,<n2>=<p2>"; env SEEDS, SEED0, EW, DEV, OPPS, NETDIR (dir with p22/p21e/p20/p17/p14b.pt); waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=${R:-$HOME/blackout-env-p24}; OUT=$R/reports/$1; C=$2; F=${NETDIR:-$HOME/p14frozen}; DEV=${DEV:-cuda}
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=$C,p22=$F/p22.pt,p21e=$F/p21e.pt,p20=$F/p20.pt,p17=$F/p17.pt,p14b=$F/p14b.pt
OURS="$(echo $C | tr ',' '\n' | cut -d= -f1 | tr '\n' ';')p22"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"; H4="$V30,hunter_quota=4"; H4H="$H4,hunter_mode=hunt"
V32=$(.venv/bin/python -c "from blackout_env.heuristics.v32_four_hunters import V32; print(V32)")
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
OPP="${OPPS:-$V32;$V31;$V30;p22;$H4;$H4H;p21e;p20;p17;$V28;$V29;strategic_v27:reserve_stage=1.5;p14b;strategic_v25;strategic_v23}"
for o in $(echo "$OPP" | tr ";" " "); do
  .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 ${SEED0:-2200000} --cap-s 425 --stop-exhausted --device $DEV \
    --chunk ${CHUNK:-16} --workers ${EW:-11} --out $OUT --ours "$OURS" --opponent "$o"
  .venv/bin/python examples/v27_eval.py report --base p22 --out $OUT > $OUT/report.txt
done
echo "EVAL DONE" >> $OUT/stdout.log
