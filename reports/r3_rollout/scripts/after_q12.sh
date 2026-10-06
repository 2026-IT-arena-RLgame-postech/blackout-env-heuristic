#!/bin/bash
# R3 (docs §49) on poscat_gpu2: wait for Q12's 16-worker gate (PID $Q12, default 2739512) to exit, then drop the
# guard for R3 (16-worker budget at nice 19): restart the rollout chain with WORKERS=12 GUARD=0 (finished
# collection parts are kept) and start the 4 baseline search lanes (GUARD=0).
# launch: setsid nohup nice -n 39 bash after_q12.sh </dev/null >~/r3runs/logs/after_q12.log 2>&1 &
Q12=${Q12:-2739512}
cd ~/blackout-env-r3
while kill -0 $Q12 2>/dev/null; do sleep 60; done
echo "$(date +%T) Q12 gone"
C=$(cat ~/r3runs/run.pid)
for p in $(pgrep -P $C); do g=$(ps -o pgid= -p $p | tr -d ' '); [ "$g" != "$C" ] && { sed -i "/^$g\$/d" ~/cpu_extra.pgids; kill -TERM -$g; kill -CONT -$g; }; done
kill $C; sleep 20
GPU=0 WORKERS=12 GUARD=0 setsid nohup nice -n 39 bash reports/r3_rollout/scripts/run.sh >> ~/r3runs/logs/run.log 2>&1 < /dev/null &
L=reports/r3_search/scripts/lane.sh; mkdir -p ~/r3runs/logs
A="p20,p17,v30,v28"; B="p16,v27,v25,p14b"
GUARD=0 setsid nohup nice -n 39 bash $L 1 a p20 64 plain,o4f_v28r $A < /dev/null > ~/r3runs/logs/lane_a.log 2>&1 &
GUARD=0 setsid nohup nice -n 39 bash $L 2 b p20 64 plain,o4f_v28r $B < /dev/null > ~/r3runs/logs/lane_b.log 2>&1 &
GUARD=0 setsid nohup nice -n 39 bash $L 3 c p17 64 plain,o4f_v28r $A < /dev/null > ~/r3runs/logs/lane_c.log 2>&1 &
GUARD=0 setsid nohup nice -n 39 bash $L 3 d p17 64 plain,o4f_v28r $B < /dev/null > ~/r3runs/logs/lane_d.log 2>&1 &
echo "$(date +%T) relaunched"
