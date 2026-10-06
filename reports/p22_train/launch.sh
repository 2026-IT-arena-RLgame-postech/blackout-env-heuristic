#!/bin/bash
# P22 (docs §55) on wsl_gpu: one more short fine-tune round on top of P21e with the §50 E recipe; TEACHER_V = v30 | v31.
# Usage: launch.sh <name> [extra run.py flags...]; env: TEACHER_V (v30), INIT (P21e), STEPS (2000), WORKERS (4, as P21e), WAIT_PID.
# = reports/p21_train/launch.sh with E's pool (EXTRA3 = V25 x2 + V23, mixture 0.3 / checkpoints 0.1 / extra 0.6) and reserve x50 at 3.4-4.04 s.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
NAME=$1; shift
R=$HOME/blackout-env-p22
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"
case ${TEACHER_V:-v30} in v30) T=$V30;; v31) T=$V31;; *) echo "bad TEACHER_V"; exit 1;; esac
V28S="$V28 $V28 $V28 $V28 "
P17=ckpt:$HOME/p14frozen/p17.pt
export R INIT=${INIT:-$HOME/p14frozen/p21e.pt} TEACHER=act:$T STEPS=${STEPS:-2000} WORKERS=${WORKERS:-4}
export OPPONENTS="mixture=0.3 checkpoints=0.1 extra=0.6"
export EXTRA2="$V28S$V29 $V30 $V30 strategic_v27:reserve_stage=1.0 $P17 $P17 ckpt:$HOME/p14frozen/p16.pt self strategic_v25 strategic_v25 strategic_v23"
$R/scripts/p18b_wsl_launch.sh $HOME/p14runs/p22_$NAME --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight 50 --reserve-seconds 3.4,4.04 --lr 1e-4 --lr-schedule cosine --lr-final 0.02 \
  --publish-every 500 --snapshot-every 1 --eval-every 1000 "$@"
