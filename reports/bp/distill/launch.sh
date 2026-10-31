#!/bin/bash
# Post-patch distillation on wsl_gpu (RTX 3060 6 GB): P24c recipe (Unity DAgger, logits K=32 + angle head, fp16+compile,
# --max-replay-ratio 2.3) with the patched build and the post-patch champion heuristic as act: teacher.
# Usage (on wsl, from ~/blackout-env-bp): TEACHER_DESC=<descriptor> reports/bp/distill/launch.sh <out-dir> [extra run.py flags]
# env: INIT (p24c_v321.pt), STEPS (20000), WORKERS (8), POOL_FILE (post-patch heuristics, one descriptor per line),
#      OPP_CKPTS (glob of checkpoint opponents), NICE (5), CUDA_VISIBLE_DEVICES (gpu2: one free GPU, never 3)
set -e
R=$HOME/blackout-env-bp; cd $R; export PYTHONPATH=$R
OUT=$1; shift; mkdir -p $OUT
F=$HOME/blackout-env-abl/checkpoints/distill/frozen
INIT=${INIT:-$F/p24c_v321.pt}
T=${TEACHER_DESC:?set TEACHER_DESC}
# opponents: post-patch heuristic pool (x2 weight for the champion family) + its perturbations + old students + self
POOL=$(tr '\n' ' ' < ${POOL_FILE:-reports/bp/distill/pool.txt})
EXTRA="$POOL perturb:eps=0.05:$T perturb:lag=2:$T perturb:sticky=3:$T perturb:dither=0.1:$T
 strategic_raider strategic_cargo_hunter strategic_early_rush strategic_fortress strategic_duelist
 ckpt:$F/p24c_v321.pt ckpt:$F/p22_v31.pt ckpt:$F/p16_v27.pt ckpt:$F/p5a_v22.pt self self"
echo "$T" > $OUT/teacher.txt
setsid nohup nice -n ${NICE:-5} .venv/bin/python -m blackout_env.train.distill.run --build $R/build/linux_server/BlackOut.x86_64 \
  --init $INIT --beta-start 0.3 --lr 1e-4 --lr-schedule cosine --lr-final 0.05 --workers ${WORKERS:-8} --steps ${STEPS:-20000} \
  --max-rows 800000 --max-replay-ratio 2.3 \
  --amp fp16 --compile --policy-head logits --action-bins 32 --teacher act:$T --act-arc-bins 1 \
  --angle-head --angle-arc-bins 1 --angle-loss-weight 1.0 --opening-weight 1.5 --opening-seconds 10 \
  --opponents mixture=0.1 checkpoints=0.1 extra=0.8 --opponent-extra $EXTRA \
  --opponent-checkpoints "${OPP_CKPTS:-$HOME/p14assets/checkpoints/offline/*/*.pt}" \
  --stand-action --stand-seconds 2.5,6 --window-weight 3 --window-seconds 3,5 \
  --reserve-weight 50 --reserve-seconds 3.4,4.04 --join-weight 10 --join-seconds 3.4,4.04 \
  --publish-every 500 --snapshot-every 1000 --eval-every 2000 \
  --out-dir $OUT "$@" >> $OUT/stdout.log 2>&1 < /dev/null &
echo $! > $OUT/run.pid; echo "started pid $(cat $OUT/run.pid)"
