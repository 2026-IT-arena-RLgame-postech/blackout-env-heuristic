#!/bin/bash
# F2: continue from F1 with lr 6e-6 (HPO winner); launch on GPU2 only when no other user has a process there at two checks 10 min apart.
cd ~/blackout-env-abl; GPU=2; L=reports/rl/q_f2.log
foreign() { local u=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" "\$1==u{print \$2}"); do
    [ "$(ps -o user= -p $p | tr -d " ")" = "$USER" ] || c=$((c+1)); done; echo $c; }
while true; do
  [ "$(foreign)" = 0 ] || { sleep 600; continue; }; sleep 600; [ "$(foreign)" = 0 ] && break
  echo "$(date) GPU busy, waiting" >> $L; done
echo "$(date) launching F2" >> $L
E4=checkpoints/rl/f1_v33 EXTRA="lr=6e-6" OUT=checkpoints/rl/f2_lr6 ALLOW_OWN=1 REPO=~/blackout-env-abl GPU=$GPU bash reports/rl/ppo_f1.sh >> $L 2>&1
