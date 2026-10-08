#!/bin/bash
# P18b runs on wsl_gpu (docs §43), all from ~/blackout-env-p18b with scripts/p18b_wsl_launch.sh.  Not meant to be run as a whole.
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V29M="$V29,join_margin=10"
CLOSE="strategic_v28:reserve_stage=0.5,trade_hold=advance $V29 strategic_v27:reserve_stage=1.0 ckpt:$HOME/p14frozen/p17.pt ckpt:$HOME/p14frozen/p16.pt self"
STANDW="--stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5"
# pilot1 (5k): STAND only -> never joins
INIT=$HOME/p14frozen/p17.pt TEACHER=act:$V29 STEPS=5000 WORKERS=9 scripts/p18b_wsl_launch.sh ~/p14runs/p18b_pilot $STANDW
# pilot2 (5k) = P18bp2 (frozen ~/p14frozen/p18bp2.pt): + close-reserve opponents  <- the recipe for the next distil
INIT=$HOME/p14frozen/p17.pt TEACHER=act:$V29 STEPS=5000 WORKERS=9 OPPONENTS="mixture=0.35 checkpoints=0.15 extra=0.5" \
  EXTRA2="$CLOSE" scripts/p18b_wsl_launch.sh ~/p14runs/p18b_pilot2 $STANDW
# full 20k with the joinswitch teacher: killed at step ~1000 after the teacher check failed vs V28
INIT=$HOME/p14frozen/p17.pt TEACHER="act:joinswitch:3.6,4.4,4.4:$V29M|ckpt:$HOME/p14frozen/p17.pt" STEPS=20000 WORKERS=8 \
  OPPONENTS="mixture=0.35 checkpoints=0.15 extra=0.5" EXTRA2="$CLOSE" scripts/p18b_wsl_launch.sh ~/p14runs/p18b $STANDW
