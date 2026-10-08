#!/bin/bash
# wait for each variant's training, then evaluate it (one at a time per chain)
cd ~/blackout-env-arch
GPU=$1; shift
for v in "$@"; do
  until grep -q "TRAIN DONE" reports/rollout_net_arch/train/run_$v.log 2>/dev/null; do
    grep -q . reports/rollout_net_arch/train/run_$v.log 2>/dev/null && ! pgrep -f "train_variant.sh $v" > /dev/null && { echo "train $v failed"; continue 2; }
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=$GPU WORKERS=${WORKERS:-6} bash eval_variant.sh $v > reports/rollout_net_arch/eval_$v.log 2>&1
done
echo CHAIN DONE "$@"
