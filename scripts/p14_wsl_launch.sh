#!/bin/bash
# P14 = P11 recipe with the V25 teacher (docs/policy_v2_design_ko.md §29), on wsl_gpu (RTX 3060 Laptop 6 GB,
# 12 threads).  Init frozen P11, lr 1e-4, logits K=32 + angle head (arc 1), opening weight 1.5 / 10 s, D6-style
# pool (the 160 loadable offline checkpoints P11 used) + EXTRA, beta 0.3, ring 1.2M, fp16 + compile
# (bench: fp16/bf16 + compile 44 ms/step at batch 256, eager fp32 103, TF32 89 -- reports/p14_bench).
# Usage: scripts/p14_wsl_launch.sh <out-dir> [extra run.py flags...]   (detached; pid in <out>/run.pid)
set -e
R=$HOME/blackout-env-p14
A=$HOME/p14assets
OUT=$1; shift
WORKERS=${WORKERS:-10}
STEPS=${STEPS:-20000}
INIT=${INIT:-$A/checkpoints/distill/frozen/p11_v22.pt}
EXTRA="strategic_raider strategic_cargo_hunter strategic_early_rush strategic_fortress strategic_duelist
 perturb:eps=0.05:strategic_v19 perturb:lag=2:strategic_v17 perturb:sticky=3:strategic_v21
 perturb:dither=0.1:strategic_v19 perturb:eps=0.05,sticky=2,lag=2:teacher:v21-teach
 ckpt:$A/checkpoints/distill/d3b/latest.pt ckpt:$A/checkpoints/distill/d4/latest.pt"
mkdir -p $OUT
cd $R
export PYTHONPATH=$R
setsid nohup .venv/bin/python -m blackout_env.train.distill.run --build $A/build/linux_server/BlackOut.x86_64 \
  --init $INIT --beta-start 0.3 --lr 1e-4 --workers $WORKERS --steps $STEPS --max-rows 1200000 --amp fp16 --compile \
  --policy-head logits --action-bins 32 --teacher act:strategic_v25 --act-arc-bins 1 \
  --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --opening-seconds 10 \
  --opponents mixture=0.45 checkpoints=0.2 extra=0.35 --opponent-extra $EXTRA \
  --opponent-checkpoints "$A/checkpoints/offline/*/*.pt" --out-dir $OUT "$@" >> $OUT/stdout.log 2>&1 < /dev/null &
echo $! > $OUT/run.pid
echo "started pid $(cat $OUT/run.pid)"
