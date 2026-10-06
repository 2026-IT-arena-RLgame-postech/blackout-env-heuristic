#!/bin/bash
# P21 (docs §50) on wsl_gpu: short V30 fine-tune of P20 (= P19 v_0005) with the staged-reserve upweight.
# Usage: launch.sh <name> [extra run.py flags...]; env: INIT (default P20), STEPS (2000), LR (1e-4), RW (--reserve-weight, 50),
# K28 (V28 copies in extra, 4), EXTRA3 (more extra entries), OPPONENTS.  Cosine lr to ~0, a snapshot every 500 steps (v_0001..), no in-run eval batches.
# Base recipe = §43 (scripts/p18b_wsl_launch.sh: --stand-action 2.5-6 s, window x3 3-5 s) + a larger close-reserve share:
# extra = 0.7 of matches, its list = the old 10 + V28 x K28, V29, V30 x2, V27(1.0), P17 x2, P16, self.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
NAME=$1; shift
R=$HOME/blackout-env-p21
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28S=$(for i in $(seq ${K28:-4}); do echo -n "$V28 "; done)
P17=ckpt:$HOME/p14frozen/p17.pt
export R INIT=${INIT:-$HOME/p14frozen/p20.pt} TEACHER=act:$V30 STEPS=${STEPS:-2000} WORKERS=${WORKERS:-9}
export OPPONENTS="${OPPONENTS:-mixture=0.2 checkpoints=0.1 extra=0.7}"
export EXTRA2="$V28S$V29 $V30 $V30 strategic_v27:reserve_stage=1.0 $P17 $P17 ckpt:$HOME/p14frozen/p16.pt self $EXTRA3"
$R/scripts/p18b_wsl_launch.sh $HOME/p14runs/p21_$NAME --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight ${RW:-50} --reserve-seconds 3.4,4.04 --lr ${LR:-1e-4} --lr-schedule cosine --lr-final 0.02 \
  --publish-every 500 --snapshot-every 1 --eval-every 1000 "$@"
