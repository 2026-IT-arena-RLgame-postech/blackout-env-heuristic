#!/bin/bash
# P20 pilots (docs §47) on wsl_gpu: §43 recipe, teacher V30, 5k steps.  Usage: launch_pilot.sh <name> <init.pt> <V28 copies in extra>
# Extra draws uniformly from its list, so K copies of V28 raise its share of 'extra' games (P19: 1 of 19).  Waits for $WAIT_PID.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
NAME=$1; INIT=$2; K=${3:-1}
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28S=$(for i in $(seq $K); do echo -n "$V28 "; done)
export INIT TEACHER=act:$V30 STEPS=${STEPS:-5000} WORKERS=9 OPPONENTS="${OPPONENTS:-mixture=0.35 checkpoints=0.15 extra=0.5}"
export EXTRA2="$V28S$V29 $V30 strategic_v27:reserve_stage=1.0 ckpt:$HOME/p14frozen/p17.pt ckpt:$HOME/p14frozen/p16.pt self"
$HOME/blackout-env-p18b/scripts/p18b_wsl_launch.sh $HOME/p14runs/p20_$NAME --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 "${@:4}"
