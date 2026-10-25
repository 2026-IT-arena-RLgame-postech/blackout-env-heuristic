#!/bin/bash
# Phase-A eval of an ablation arm / control with the batched fast eval (examples/eval_fast.py, fp16 + CUDA graphs):
# 256 seeds (970000..970255) x 2 sides vs p24c, p22, V32.1 (+ head-to-head vs CTRL if given).
# Fast eval is not bit-identical to the old eval on GPU (batch-size numerics) -> controls are re-evaluated with this script.
# Usage: [REPO=..] [GPU=n] [CHUNK=128] CAND=<ckpt> TAG=<name> [CTRL=<ckpt>] reports/rl/abl_eval.sh   (never GPU 3)
set -e
cd ${REPO:-.}
CAND=${CAND:?}; TAG=${TAG:?}
if [ -n "$GPU" ]; then [ "$GPU" != 3 ] || { echo "GPU 3 is off limits"; exit 1; }; export CUDA_VISIBLE_DEVICES=$GPU; fi
F=checkpoints/distill/frozen
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;V32.1"; [ -z "$CTRL" ] || OPPS="$OPPS;net:$CTRL"
mkdir -p reports/rl/abl_eval
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/eval_fast.py --cand $CAND --opps "$OPPS" \
  --lo 970000 --hi 970256 --chunk ${CHUNK:-128} --out reports/rl/abl_eval --tag $TAG --device cuda:0 --heur-workers ${HW:-8} \
  > reports/rl/abl_eval/$TAG.log 2>&1
echo "$(date) eval done $TAG"
