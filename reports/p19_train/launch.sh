#!/bin/bash
# P19 (docs §45): V30 distilled with the P18b pilot2 recipe, init = P18bp2, 20k steps, wsl_gpu.  Waits for $WAIT_PID first.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
export INIT=$HOME/p14frozen/p18bp2.pt TEACHER=act:$V30 STEPS=20000 WORKERS=9 OPPONENTS="mixture=0.35 checkpoints=0.15 extra=0.5"
export EXTRA2="strategic_v28:reserve_stage=0.5,trade_hold=advance $V29 $V30 strategic_v27:reserve_stage=1.0 ckpt:$HOME/p14frozen/p17.pt ckpt:$HOME/p14frozen/p16.pt self"
$HOME/blackout-env-p18b/scripts/p18b_wsl_launch.sh $HOME/p14runs/p19 --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5
