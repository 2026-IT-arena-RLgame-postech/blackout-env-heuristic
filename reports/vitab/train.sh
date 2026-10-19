#!/bin/bash
# One arm of the ViT A/B (docs/rl_finetune_plan_ko.md): offline, from scratch, on the shared V32.1-labelled shards
# (reports/vitab/collect.sh), the P24c loss weights (reports/p24_train/launch.sh), lr 3e-4 cosine warm-up 500 as the §25 from-scratch runs.
# Usage: CUDA_VISIBLE_DEVICES=<n> reports/vitab/train.sh <cnn|patch4|patch6> <seed> <out-dir>   (foreground; the queue runs it detached)
set -e
R=${R:-$HOME/blackout-env-vitab}; cd $R; export PYTHONPATH=$R
ENC=$1; SEED=$2; OUT=$3; mkdir -p $OUT
exec .venv/bin/python examples/distill_offline_ab.py ${SHARDS:-checkpoints/distill/vitab/data/shards} --rows 800000 --steps ${STEPS:-30000} \
  --seed $SEED --eval-every 5000 --log-every 500 --heldout-rows 60000 --out $OUT/result.json --save $OUT/final.pt -- \
  --map-encoder $ENC --policy-head logits --action-bins 32 --act-arc-bins 1 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 \
  --opening-weight 1.5 --opening-seconds 10 --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight 50 --reserve-seconds 3.4,4.04 --join-weight 10 --join-seconds 3.4,4.04 \
  --lr 3e-4 --lr-schedule cosine --lr-warmup 500 --lr-final 0.02 --batch-size 256 --amp fp16 --compile
