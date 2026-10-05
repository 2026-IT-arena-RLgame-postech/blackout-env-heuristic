#!/bin/bash
# Q11 V30 gate (docs/policy_v2_design_ko.md §44): paired sim eval (examples/v27_eval.py, full matches, cap 150 s,
# stop when exhausted), base = V29 on the same (seed, side).  Nets on CPU.
#   reports/q11_v30/scripts/gate.sh <out> <seed0> <seeds> <workers> <ours ';'-list> <opponents ';'-list>
# env: PY (python), NETDIR (dir with p17_v28.pt / p16_v27.pt / p14_v25.pt)
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=${1:?out}; SEED0=${2:?seed0}; SEEDS=${3:?seeds}; WORKERS=${4:?workers}; OURS=${5:?ours}; OPPS=${6:?opponents}
NETDIR=${NETDIR:-checkpoints/distill/frozen}
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLACKOUT_NETS="p17=$NETDIR/p17_v28.pt,p16=$NETDIR/p16_v27.pt,p14b=$NETDIR/p14_v25.pt"
PY=${PY:-.venv/bin/python}
nice -n 10 "$PY" examples/v27_eval.py play --ours "$OURS" --opponent "$OPPS" --seed0 "$SEED0" --seeds "$SEEDS" \
  --chunk 16 --workers "$WORKERS" --stop-exhausted --out "$OUT"
