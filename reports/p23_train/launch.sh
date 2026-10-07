#!/bin/bash
# P23 (docs §57) on wsl_gpu: one more short fine-tune round on top of P22 with the §50/§55 E recipe and teacher V31 (TEACHER_V = v31 | v30); OOD_ADD=1 adds OOD1's V30 hunter_quota=4 (h4), +hunter_mode=hunt (h4hunt), carrier_quota=0 (c0) to extra (P23b).
# Usage: launch.sh <name> [extra run.py flags...]; env: TEACHER_V (v31), OOD_ADD (1 = add OOD1's h4, h4hunt, c0 to extra = P23b), INIT (P22), STEPS (2000), WORKERS (4, as P21e), WAIT_PID.
# = reports/p21_train/launch.sh with E's pool (EXTRA3 = V25 x2 + V23, mixture 0.3 / checkpoints 0.1 / extra 0.6) and reserve x50 at 3.4-4.04 s.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
NAME=$1; shift
R=$HOME/blackout-env-p23
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V31="$V30,threat_radius=0.13,camp_engage_radius=6.5"
case ${TEACHER_V:-v31} in v30) T=$V30;; v31) T=$V31;; *) echo "bad TEACHER_V"; exit 1;; esac
[ "${OOD_ADD:-0}" = 1 ] && EXTRA_ADD="$V30,hunter_quota=4 $V30,hunter_quota=4,hunter_mode=hunt $V30,carrier_quota=0"
V28S="$V28 $V28 $V28 $V28 "
P17=ckpt:$HOME/p14frozen/p17.pt
export R INIT=${INIT:-$HOME/p14frozen/p22.pt} TEACHER=act:$T STEPS=${STEPS:-2000} WORKERS=${WORKERS:-4}
export OPPONENTS="mixture=0.3 checkpoints=0.1 extra=0.6"
export EXTRA2="$V28S$V29 $V30 $V30 strategic_v27:reserve_stage=1.0 $P17 $P17 ckpt:$HOME/p14frozen/p16.pt self strategic_v25 strategic_v25 strategic_v23 $EXTRA_ADD"
$R/scripts/p18b_wsl_launch.sh $HOME/p14runs/p23_$NAME --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight 50 --reserve-seconds 3.4,4.04 --lr 1e-4 --lr-schedule cosine --lr-final 0.02 \
  --publish-every 500 --snapshot-every 1 --eval-every 1000 "$@"
