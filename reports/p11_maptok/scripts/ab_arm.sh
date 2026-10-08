#!/bin/bash
# One offline A/B arm (docs §25): ab_arm.sh <name> <gpu> <steps> <rows> <init|none> -- <learner flags...>
# Runs in ~/blackout-env-p11m on poscat_gpu2; output runs/<name>/{stdout.log,ab.json,model.pt}
set -e
cd ~/blackout-env-p11m
export PYTHONPATH=$PWD
NAME=$1 GPU=$2 STEPS=$3 ROWS=$4 INIT=$5; shift 5; [[ $1 == -- ]] && shift
OUT=runs/$NAME; mkdir -p $OUT
INITARG=(); [[ $INIT != none ]] && INITARG=(--init $INIT)
CUDA_VISIBLE_DEVICES=$GPU exec ~/blackout-env/.venv/bin/python examples/distill_offline_ab.py checkpoints/distill/p11/shards \
  --steps $STEPS --rows $ROWS --heldout-rows 0 --seed ${SEED:-0} --log-every 500 --eval-every 5000 --clean-heldout data/clean "${INITARG[@]}" \
  --out $OUT/ab.json --save $OUT/model.pt -- \
  --policy-head logits --action-bins 32 --act-arc-bins 1 --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 \
  --opening-weight 1.5 --opening-seconds 10 --amp fp16 --compile "$@" > $OUT/stdout.log 2>&1
