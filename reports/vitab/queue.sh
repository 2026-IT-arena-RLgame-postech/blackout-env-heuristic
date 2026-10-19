#!/bin/bash
# Sequential queue of the ViT A/B runs on GPU 3 (ours; shares it with the GRPO job).  Before every start it re-checks nvidia-smi:
# every process on the GPU must be ours (GRPO pid GRPO_PID or this queue's own children) and used + REQ MiB must stay under 11264 - 1024.
# Usage: reports/vitab/queue.sh "<arm>:<enc>:<seed> ..."   (detached by the caller; log: checkpoints/distill/vitab/queue.log)
R=$HOME/blackout-env-vitab; cd $R
GPU=3; GRPO_PID=${GRPO_PID:-193230}; REQ=${REQ:-1800}
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
gate() {
  local others used
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$UUID -v p=$GRPO_PID '$1==u && $2!=p{print $2}')
  [ -z "$others" ] || { echo "$(date) gate: foreign pid(s) on GPU $GPU: $others"; return 1; }
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')
  [ $((used + REQ)) -lt $((11264 - 1024)) ] || { echo "$(date) gate: used $used MiB + $REQ > limit"; return 1; }
}
for job in $1; do
  IFS=: read arm enc seed <<< "$job"; out=checkpoints/distill/vitab/${arm}_s${seed}
  [ -f $out/result.json ] && continue
  until gate; do sleep 120; done
  echo "$(date) start $job"
  mkdir -p $out
  CUDA_VISIBLE_DEVICES=$GPU nice -n 30 env PYTHONPATH=. reports/vitab/train.sh $enc $seed $out > $out/run.log 2>&1
  echo "$(date) done $job rc=$?"
done
