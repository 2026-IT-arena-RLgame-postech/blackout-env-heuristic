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
#   build              Unity player from ../blackout (run11-80k branch) into build/mac/BlackOut.app
#
# [ckpt] defaults to the committed Run 11 80k checkpoint. Everything below the Run 11 values can be
# overridden from the environment, e.g. a smoke test:
#   COLLECT_STEPS=4000 COLLECT_WORKERS=2 DATASET=/tmp/ds TRAIN_STEPS=300 EVAL_INTERVAL=150 run11_pipeline.sh train
set -euo pipefail

PY=${PY:-.venv/bin/python}
BUILD=${BUILD:-build/mac/BlackOut.app}
UNITY=${UNITY:-/Applications/Unity/Hub/Editor/6000.4.11f1/Unity.app/Contents/MacOS/Unity}
DATASET=${DATASET:-datasets/heuristic_mixv6_live_20260917}
COLLECT_STEPS=${COLLECT_STEPS:-1000000}
COLLECT_WORKERS=${COLLECT_WORKERS:-18}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
DEVICE=${DEVICE:-mps}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
SEEDS=(${=SEEDS:-404 505 606})
ELO_TARGET_SE=${ELO_TARGET_SE:-50}
ELO_MAX_GAMES=${ELO_MAX_GAMES:-600}
CKPT_DEFAULT=models/run11_step80k/step_80000.pt
stage=${1:-help}
shift || true

case $stage in
  build)
    $UNITY -batchmode -quit -projectPath ../blackout -executeMethod CIBuild.BuildBlackOutMac \
      -buildPath $BUILD -logFile build/mac_build.log
    ;;
  collect)
    # Run 11 collected with the collector's then-default --noise-frac 0.1; the default is 0 now.
    $PY -m blackout_env.train.collect_heuristic_dataset_parallel --build $BUILD \
      --steps $COLLECT_STEPS --workers $COLLECT_WORKERS --noise-frac 0.1 --out $DATASET
    ;;
  train)
    extra=()
    [[ -n $CHECKPOINT_DIR ]] && extra=(--checkpoint-dir $CHECKPOINT_DIR)
    [[ $DEVICE == mps || $DEVICE == cuda* ]] && extra+=(--compile)
    $PY -m blackout_env.train.offline_pretrain --dataset-dir $DATASET --eval-build $BUILD \
      --reward v2-fitted --steps $TRAIN_STEPS --device $DEVICE --spr-loss-weight 5.0 --encoder-weight-decay 1e-4 \
      --bc-loss-alpha 1.0 --blocked-penalty 0.02 --onpolicy-self-vs-heuristic-frac 0.3 \
      --onpolicy-self-play-frac 0 --reset-warmup-steps 2000 --eval-interval $EVAL_INTERVAL $extra
    ;;
  gui)
    $PY examples/evaluate_checkpoint_vs_heuristic.py --build $BUILD --checkpoint ${1:-$CKPT_DEFAULT} --time-scale 3 --seeds $SEEDS
    ;;
  selfplay)
    ck=${1:-$CKPT_DEFAULT}
    $PY examples/evaluate_checkpoint_vs_checkpoint.py --build $BUILD --a a=$ck --b b=$ck --seeds $SEEDS --time-scale 3
    ;;
  vs)
    $PY examples/evaluate_checkpoint_vs_checkpoint.py --build $BUILD --a a=${2:-$CKPT_DEFAULT} --b b=$1 --seeds $SEEDS --time-scale 3
    ;;
  measure)
    ck=${1:-$CKPT_DEFAULT}
    $PY examples/measure_movement_and_storage.py --build $BUILD --checkpoint $ck
    $PY examples/measure_match_phases.py --build $BUILD --checkpoint $ck
    ;;
  elo)
    run=$1
    cks=()
    for s in 20000 40000 60000 80000 100000 120000 140000 160000 180000; do cks+=("s$((s / 1000))k=$run/step_$s.pt"); done
    cks+=("final=$run/final.pt")
    $PY examples/elo_checkpoints.py --build $BUILD --checkpoints $cks --anchors --reference final \
      --target-se $ELO_TARGET_SE --max-games $ELO_MAX_GAMES --output reports/elo_$(basename $run)
    ;;
  *)
    sed -n 2,20p $0
    ;;
esac
