#!/usr/bin/env zsh
# Run 11 end to end: heuristic data -> offline pretraining -> evaluation.
# Run from the blackout-env root:  models/run11_step80k/run11_pipeline.sh <stage> [args]
#
#   collect            1M live rows per stream (matches end when no battery is left), 10% uniform noise
#   train              Run 11's exact offline_pretrain command (200k steps, reward v2 fitted)
#   gui [ckpt]         6 matches vs V4 with the Unity window (seeds 404 505 606, 3x speed)
#   selfplay [ckpt]    the checkpoint against itself with the Unity window
#   vs <ckpt_b> [ckpt] two checkpoints against each other with the Unity window
#   measure [ckpt]     headless: reversal by wall distance, storage split, per-5 s steals/deliveries vs V4
#   elo <run_dir>      relative Elo of a run's 20k..180k + final checkpoints (final = 0, SE 50)
#
# [ckpt] defaults to the committed Run 11 80k checkpoint. Unity build: blackout repo 94fabbd
# (build/mac/BlackOut.app, see docs/offline_pretrain_runs.md for the build command).
set -euo pipefail

PY=.venv/bin/python
BUILD=build/mac/BlackOut.app
DATASET=datasets/heuristic_mixv6_live_20260917
CKPT_DEFAULT=models/run11_step80k/step_80000.pt
stage=${1:-help}
shift || true

case $stage in
  collect)
    # Run 11 collected with the collector's then-default --noise-frac 0.1; the default is 0 now.
    $PY -m blackout_env.train.collect_heuristic_dataset_parallel --build $BUILD \
      --steps 1000000 --workers 18 --noise-frac 0.1 --out $DATASET
    ;;
  train)
    $PY -m blackout_env.train.offline_pretrain --dataset-dir $DATASET \
      --reward v2-fitted --steps 200000 --device mps --compile --spr-loss-weight 5.0 --encoder-weight-decay 1e-4 \
      --bc-loss-alpha 1.0 --blocked-penalty 0.02 --onpolicy-self-vs-heuristic-frac 0.3 \
      --onpolicy-self-play-frac 0 --reset-warmup-steps 2000 --eval-interval 10000
    ;;
  gui)
    $PY examples/evaluate_checkpoint_vs_heuristic.py --checkpoint ${1:-$CKPT_DEFAULT} --time-scale 3 --seeds 404 505 606
    ;;
  selfplay)
    ck=${1:-$CKPT_DEFAULT}
    $PY examples/evaluate_checkpoint_vs_checkpoint.py --a a=$ck --b b=$ck --seeds 404 505 606 --time-scale 3
    ;;
  vs)
    $PY examples/evaluate_checkpoint_vs_checkpoint.py --a a=${2:-$CKPT_DEFAULT} --b b=$1 --seeds 404 505 606 --time-scale 3
    ;;
  measure)
    ck=${1:-$CKPT_DEFAULT}
    $PY examples/measure_movement_and_storage.py --checkpoint $ck
    $PY examples/measure_match_phases.py --checkpoint $ck
    ;;
  elo)
    run=$1
    cks=()
    for s in 20000 40000 60000 80000 100000 120000 140000 160000 180000; do cks+=("s$((s / 1000))k=$run/step_$s.pt"); done
    cks+=("final=$run/final.pt")
    $PY examples/elo_checkpoints.py --checkpoints $cks --anchors --reference final --target-se 50 \
      --output reports/elo_$(basename $run)
    ;;
  *)
    sed -n 2,15p $0
    ;;
esac
