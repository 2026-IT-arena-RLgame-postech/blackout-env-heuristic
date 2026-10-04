#!/usr/bin/env bash
# Replay check failed bit-identity (38/40 equal; seed 1200005 differs both sides vs §22's rows), so
# P11 vs V23 / V19 / P5a is re-played on the same seeds after run.sh finishes (same 10 workers, GPU0).
set -u
cd ~/blackout-env-q1
OUT=reports/p14_unity/unity
until grep -q "RUN DONE" $OUT/stdout.log; do sleep 30; done
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export BLACKOUT_CUDA_NETS=p11_v22.pt,p5a_v22.pt,p14_v25.pt
PY=~/blackout-env/.venv/bin/python
P11=checkpoints/distill/frozen/p11_v22.pt
P5A=checkpoints/distill/frozen/p5a_v22.pt
echo "run2 start $(date)" >> $OUT/stdout.log
$PY examples/distill_heldout_eval.py $P11 --opponents strategic_v23 strategic_v19 ckpt:$P5A --teacher act:strategic_v23 \
    --no-teacher-baseline --seed-base 1200000 --workers ${WORKERS:-10} --build build/linux_server/BlackOut.x86_64 \
    --seeds 200 --out $OUT >> $OUT/stdout.log 2>&1
echo "RUN2 DONE $(date)" >> $OUT/stdout.log
