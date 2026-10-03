#!/usr/bin/env bash
# H1 in Unity (docs §22 "Unity 확인"): P11 and the two map-token nets vs V23 / V19 / P5a, then the
# small nets head-to-head vs P11.  Seeds 1200000-199 x both sides (= §19's r5 Unity run), 10 Unity
# workers on poscat_gpu2 (moved from gpu1 at 1055 matches; resumed), P11 / P5a on GPU0 (BLACKOUT_CUDA_NETS), the small nets on the CPU.
# Run with setsid nohup from ~/blackout-env-newnet; a rerun skips matches already in matches.jsonl.
set -u
cd ~/blackout-env-newnet
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export BLACKOUT_CUDA_NETS=${BLACKOUT_CUDA_NETS-p11_v22.pt,p5a_v22.pt}
PY=~/blackout-env/.venv/bin/python
OUT=reports/unity_newnet/unity
P11=checkpoints/distill/frozen/p11_v22.pt
P5A=checkpoints/distill/frozen/p5a_v22.pt
D256=checkpoints/rollout_net/arch/d256_f3_x32_d2.pt
F3X=checkpoints/rollout_net/arch/f3_x_d2.pt
COMMON="--teacher act:strategic_v23 --no-teacher-baseline --seeds ${SEEDS:-200} --seed-base 1200000 --workers ${WORKERS:-10}
        --build build/linux_server/BlackOut.x86_64"
mkdir -p $OUT
echo "start $(date)" >> $OUT/stdout.log
$PY examples/distill_heldout_eval.py $P11 $D256 $F3X --opponents strategic_v23 strategic_v19 ckpt:$P5A $COMMON --out $OUT >> $OUT/stdout.log 2>&1
echo "part1 done $(date)" >> $OUT/stdout.log
$PY examples/distill_heldout_eval.py $D256 $F3X --opponents ckpt:$P11 $COMMON --out $OUT >> $OUT/stdout.log 2>&1
echo "RUN DONE $(date)" >> $OUT/stdout.log
