#!/bin/bash
# Q7 V28 gate on poscat_gpu1 (CPU only, nice 10, 8 workers; docs/policy_v2_design_ko.md §36):
# paired sim eval (examples/v27_eval.py, full matches, stop when exhausted), base = V27(1.5) on the same (seed, side).
#   scripts/q7_gate.sh <out> <seed0> <seeds> [ours] [opponents]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out}; SEED0=${2:?seed0}; SEEDS=${3:?seeds}
OURS=${4:-"strategic_v27:reserve_stage=1.5;strategic_v28:reserve_stage=0.5;strategic_v28:reserve_stage=0.5,trade_hold=advance;strategic_v28:reserve_stage=0.5,trade_hold=hold;strategic_v28:reserve_stage=1.5,trade_hold=advance"}
OPPS=${5:-"strategic_v27:reserve_stage=1.5;strategic_v25;strategic_v23;p14b;p16;p11;strategic_v27:reserve_stage=1.0;perturb:eps=0.1:strategic_v27:reserve_stage=1.5;strategic_v27:reserve_stage=0.5"}
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=. BLACKOUT_NETS="p16=checkpoints/distill/frozen/p16_v27.pt"
PY=${PY:-$HOME/blackout-env/.venv/bin/python}
nice -n 10 "$PY" examples/v27_eval.py play --ours "$OURS" --opponent "$OPPS" --seed0 "$SEED0" --seeds "$SEEDS" \
  --chunk 16 --workers 8 --stop-exhausted --out "$OUT"
"$PY" examples/v27_eval.py report --base "strategic_v27:reserve_stage=1.5" --out "$OUT"
