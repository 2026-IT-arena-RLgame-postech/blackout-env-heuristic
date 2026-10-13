#!/bin/bash
# Strength of the official-format submission vs V32.1 (seeds 4000000+, both sides, full 420 s matches).
#   launch.sh ARM(plain|search) WORKERS SEEDS_PER_WORKER [GPUS]
# rows -> reports/official_submission/strength_v321/<arm>_w<k>.jsonl
set -u
ARM=$1; NW=$2; PER=$3; GPUS=${4:-}
cd ~/blackout-env
OUT=reports/official_submission/strength_v321
mkdir -p $OUT/logs
V321="strategic_v32:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5,hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,sweep=true,cargo_transform=true,adapt_k=3"
for k in $(seq 0 $((NW - 1))); do
  if [ -n "$GPUS" ]; then IFS=, read -ra G <<< "$GPUS"; export CUDA_VISIBLE_DEVICES=${G[$((k % ${#G[@]}))]}; else export CUDA_VISIBLE_DEVICES=; fi
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=1 PYTHONPATH=.:$HOME/ort_pkgs nohup nice -n 10 \
    .venv/bin/python examples/official_submission_check.py --dir ~/sub_$ARM --official-env ~/official-env --reference "" \
    --opp "$V321" --seed0 $((4000000 + k * PER)) --seeds $PER --out $OUT/${ARM}_w$k.jsonl > $OUT/logs/${ARM}_w$k.log 2>&1 &
done
echo launched $ARM x $NW
