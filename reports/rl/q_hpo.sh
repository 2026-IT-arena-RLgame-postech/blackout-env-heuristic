#!/bin/bash
# HPO queue on one GPU: runs the trials of reports/rl/hpo_trials.txt in order (first line whose checkpoints/rl/hpo_<name>
# dir does not exist), each followed by its learning-curve eval in the background (overlaps the next trial's training).
# Stop after the current trial: touch reports/rl/q_hpo.stop.   Usage: REPO=~/blackout-env-abl GPU=2 nohup bash reports/rl/q_hpo.sh
cd ${REPO:?}; GPU=${GPU:?}; [ "$GPU" != 3 ] || exit 1
L=reports/rl/q_hpo.log
foreign() {   # number of other users' processes on the GPU
  local u=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) c=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2}'); do
    [ "$(ps -o user= -p $p | tr -d ' ')" = "$USER" ] || c=$((c+1)); done; echo $c; }
while [ ! -e reports/rl/q_hpo.stop ]; do
  next=$(grep -v '^#' reports/rl/hpo_trials.txt | while IFS='|' read -r n s; do n=$(echo $n); [ -n "$n" ] && [ ! -e checkpoints/rl/hpo_$n ] && { echo "$n|$s"; break; }; done)
  [ -n "$next" ] || { echo "$(date) Q_HPO queue empty" >> $L; break; }
  n=${next%%|*}; s=$(echo ${next#*|})
  # conservative: no foreign process at two checks 10 min apart (user 10-27: only a completely free GPU)
  [ "$(foreign)" = 0 ] || { sleep 600; continue; }
  sleep 600
  [ "$(foreign)" = 0 ] || { echo "$(date) GPU $GPU taken by another user, waiting" >> $L; continue; }
  out=$(TRIAL=$n HP="$s" ALLOW_OWN=1 REPO=$REPO GPU=$GPU bash reports/rl/ppo_hpo.sh 2>&1)
  pid=$(echo "$out" | sed -n 's/launched pid \([0-9]*\).*/\1/p')
  if [ -z "$pid" ]; then echo "$(date) launch $n failed: $out (retry in 10 min)" >> $L; rmdir checkpoints/rl/hpo_$n 2>/dev/null; rm -rf checkpoints/rl/hpo_$n; sleep 600; continue; fi
  echo "$(date) start $n [$s] pid $pid" >> $L
  while kill -0 $pid 2>/dev/null; do sleep 60; done
  pkill -f "[h]ealth_logger.sh checkpoints/rl/hpo_$n" 2>/dev/null
  if [ -s checkpoints/rl/hpo_$n/final.pt ]; then
    echo "$(date) trained $n" >> $L
    (TRIAL=$n REPO=$REPO GPU=$GPU bash reports/rl/hpo_eval.sh >> $L 2>&1 &)
  else echo "$(date) $n died without final.pt (see run.log)" >> $L; fi
done
echo "$(date) Q_HPO stopped" >> $L
