#!/usr/bin/env bash
# P14b (docs §29) in Unity: P14b vs V23 / V19 / P5a / V25 / P11, plus P11 vs V25 (no §22 row), seeds
# 1200000-199 x both sides (= §19/§22 P11 Unity rows, reused if the replay check matches bit for bit).
# 10 Unity workers on poscat_gpu2, P11 / P14b / P5a on GPU0 (BLACKOUT_CUDA_NETS).
# Run with setsid nohup from ~/blackout-env-q1 (checkpoints/ and build/ symlinked to the mirror);
# a rerun skips matches already in matches.jsonl.
set -u
cd ~/blackout-env-q1
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export BLACKOUT_CUDA_NETS=p11_v22.pt,p5a_v22.pt,p14_v25.pt
PY=~/blackout-env/.venv/bin/python
OUT=reports/p14_unity/unity
P11=checkpoints/distill/frozen/p11_v22.pt
P5A=checkpoints/distill/frozen/p5a_v22.pt
P14B=checkpoints/distill/frozen/p14_v25.pt
COMMON="--teacher act:strategic_v23 --no-teacher-baseline --seed-base 1200000 --workers ${WORKERS:-10}
        --build build/linux_server/BlackOut.x86_64"
mkdir -p $OUT reports/p14_unity/replay
echo "start $(date)" >> $OUT/stdout.log
# replay check: P11 vs V23 / P5a, first 10 seeds, against reports/unity_newnet/unity (§22, same build)
$PY examples/distill_heldout_eval.py $P11 --opponents strategic_v23 ckpt:$P5A $COMMON --seeds 10 \
    --out reports/p14_unity/replay >> $OUT/stdout.log 2>&1
echo "replay done $(date)" >> $OUT/stdout.log
$PY examples/distill_heldout_eval.py $P14B --opponents strategic_v23 strategic_v25 ckpt:$P11 strategic_v19 ckpt:$P5A \
    $COMMON --seeds 200 --out $OUT >> $OUT/stdout.log 2>&1
echo "part1 done $(date)" >> $OUT/stdout.log
$PY examples/distill_heldout_eval.py $P11 --opponents strategic_v25 $COMMON --seeds 200 --out $OUT >> $OUT/stdout.log 2>&1
echo "RUN DONE $(date)" >> $OUT/stdout.log
