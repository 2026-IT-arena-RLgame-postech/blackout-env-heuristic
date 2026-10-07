#!/bin/bash
# P24 (docs §60) snapshot screen on wsl_gpu (= reports/p21_train/screen.sh with base P22, V31, V32 and OOD1 h4 added):
# (1) Hunters at decision 99 (cap 5 s) vs V28 / P17 / V30 / V31 / V32 / h4, 32 seeds from 1700000 x both sides;
# (2) short full matches vs V28 / P17 / V25 / V31 / V32 / h4, 16 seeds from 1800000 (not the gate seeds), paired with P22.
# Usage: screen.sh <run name>; env DEV (cuda | cpu: gpu1), NETDIR (p17/p22.pt), RUNDIR (dir holding p24_<name>/v_*.pt), EW, SS, SOPP, SKIP_H99; out reports/p24_train/screen_<name>; waits for $WAIT_PID.  Snapshot v_0001 = the init (skipped); SNAPS = glob of snapshot digits (default [2-9]).
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p24; B=${RUNDIR:-$HOME/p14runs}/p24_$1; F=${NETDIR:-$HOME/p14frozen}; DEV=${DEV:-cuda}; OUT=$R/reports/p24_train/screen_$1
cd $R; mkdir -p $OUT; export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; [ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=
NETS=p17=$F/p17.pt,p22=$F/p22.pt; OURS="p22"
ls $B/v_000${SNAPS:-[2-9]}.pt >/dev/null || { echo "no snapshots in $B" >> $OUT/stdout.log; exit 1; }
for f in $B/v_000${SNAPS:-[2-9]}.pt; do n=$1$(basename $f .pt | sed 's/v_0*/v/'); NETS=$NETS,$n=$f; OURS="$OURS;$n"; done
export BLACKOUT_NETS=$NETS
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V31="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"; H4="$V30,hunter_quota=4"; V32=$(.venv/bin/python -c "from blackout_env.heuristics.v32_four_hunters import V32; print(V32)")
[ -z "$SKIP_H99" ] && .venv/bin/python reports/p20_diag/scripts/diag.py play --ours "$OURS" --opponent "V28;p17;V30;$V31;$V32;$H4" --seeds 32 --seed0 1700000 --cap-s 5 \
  --workers ${EW:-11} --chunk 16 --device $DEV --out $OUT/h99
[ -z "$SKIP_H99" ] && .venv/bin/python reports/p20_diag/scripts/diag.py summary --out $OUT/h99 2>/dev/null | grep -v "^ " > $OUT/h99.txt
.venv/bin/python examples/v27_eval.py play --seeds ${SS:-16} --seed0 1800000 --cap-s 425 --stop-exhausted --device $DEV --chunk 16 \
  --workers ${EW:-11} --out $OUT/full --ours "$OURS" --opponent "${SOPP:-$V28;p17;strategic_v25;$V31;$V32;$H4}"
.venv/bin/python examples/v27_eval.py report --base p22 --out $OUT/full > $OUT/full.txt
echo "SCREEN DONE" >> $OUT/stdout.log
