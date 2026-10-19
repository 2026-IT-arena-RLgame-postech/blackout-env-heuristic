#!/bin/bash
# ViT A/B data (docs/rl_finetune_plan_ko.md): one shared V32.1-labelled DAgger dataset, collected with the P24c recipe
# (reports/p24_train/launch.sh: E pool, V32.1 act teacher, beta 0.3 decaying 0.9 per version, same loss weights) by P24c itself
# (INIT = p24c_v321.pt, lr ~0 so the collecting student stays P24c).  The P24c run's own shards no longer exist (wsl_gpu cleaned up).
# Usage: CUDA_VISIBLE_DEVICES=3 reports/vitab/collect.sh <out-dir> [extra run.py flags]    (run from a checkout with .venv and build/)
set -e
R=${R:-$HOME/blackout-env-vitab}; cd $R; export PYTHONPATH=$R
F=$R/checkpoints/distill/frozen; mkdir -p $(dirname $0) 2>/dev/null || true
OUT=$1; shift; mkdir -p $OUT
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V32=$(.venv/bin/python -c "from blackout_env.heuristics.v32_four_hunters import V32; print(V32)")
V28S="$V28 $V28 $V28 $V28 "
P17=ckpt:$F/p17_v28.pt
EXTRA="strategic_raider strategic_cargo_hunter strategic_early_rush strategic_fortress strategic_duelist
 perturb:eps=0.05:strategic_v19 perturb:lag=2:strategic_v17 perturb:sticky=3:strategic_v21
 perturb:dither=0.1:strategic_v19 perturb:eps=0.05,sticky=2,lag=2:teacher:v21-teach
 ckpt:$F/d6_v42.pt ckpt:$F/p5a_v22.pt $V28S$V29 $V30 $V30 strategic_v27:reserve_stage=1.0 $P17 $P17 ckpt:$F/p16_v27.pt self strategic_v25 strategic_v25 strategic_v23"
setsid nohup nice -n 30 .venv/bin/python -m blackout_env.train.distill.run --build $R/build/linux_server/BlackOut.x86_64 \
  --init $F/p24c_v321.pt --beta-start 0.3 --lr 1e-8 --workers ${WORKERS:-10} --steps ${STEPS:-100000} --max-rows 800000 --max-replay-ratio 2.3 \
  --amp fp16 --compile --policy-head logits --action-bins 32 --teacher act:$V32,adapt_k=3 --act-arc-bins 1 \
  --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --opening-seconds 10 \
  --opponents mixture=0.3 checkpoints=0.1 extra=0.6 --opponent-extra $EXTRA \
  --opponent-checkpoints "$F/*.pt" --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight 50 --reserve-seconds 3.4,4.04 --join-weight 10 --join-seconds 3.4,4.04 --publish-every 500 --snapshot-every 1000 --eval-every 100000 \
  --out-dir $OUT "$@" >> $OUT/stdout.log 2>&1 < /dev/null &
echo $! > $OUT/run.pid; echo "started pid $(cat $OUT/run.pid)"
