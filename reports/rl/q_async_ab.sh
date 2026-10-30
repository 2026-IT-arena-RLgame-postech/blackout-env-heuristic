#!/bin/bash
# Learning A/B: the HPO lr6e-6 trial (P24c start, beta 0.01, pool 9 + V33, 10M, ckpt every 50 its) with async_collect on
# 2 GPUs (learner GPU2, actor GPU0), compared to the two sync replicates hpo_lr6e-6 / hpo_lr6e-6_r2 via hpo_eval.sh + hpo_analysis.py.
# Launches only when GPU2 and GPU0 have no other user's process at two checks 10 min apart. Log reports/rl/q_async_ab.log.
cd ~/blackout-env-abl; L=reports/rl/q_async_ab.log; T=${T:-lr6e-6_async}
foreign() { local u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" '$1==u{print $2}'); do
    [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || c=$((c+1)); done; echo $c; }
free2() { [ "$(foreign 2)" = 0 ] && [ "$(foreign 0)" = 0 ]; }
while true; do free2 || { sleep 600; continue; }; sleep 600; free2 && break; echo "$(date) busy, waiting" >> $L; done
echo "$(date) launching $T" >> $L
TRIAL=$T HP="lr=6e-6 async_collect=true actor_device=cuda:1" REPO=~/blackout-env-abl GPU=2,0 bash reports/rl/ppo_hpo.sh >> $L 2>&1
pid=$(sed -n 's/launched pid \([0-9]*\).*/\1/p' $L | tail -1)
while kill -0 $pid 2>/dev/null; do sleep 60; done
pkill -f "[h]ealth_logger.sh checkpoints/rl/hpo_$T" 2>/dev/null
[ -s checkpoints/rl/hpo_$T/final.pt ] || { echo "$(date) died without final.pt" >> $L; exit 1; }
echo "$(date) trained" >> $L
TRIAL=$T REPO=~/blackout-env-abl GPU=2 bash reports/rl/hpo_eval.sh >> $L 2>&1
echo "$(date) eval done" >> $L
