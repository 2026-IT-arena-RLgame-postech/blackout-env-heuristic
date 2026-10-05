#!/bin/bash
# P17 Unity milestone (docs §40) on poscat_gpu2: P17 and P16 (base) vs V28(0.5, advance), V25, P16; 300 matches/pair, cap 425 s.
# setsid nohup bash reports/p17_unity/scripts/run.sh </dev/null > reports/p17_unity/play.log 2>&1 &
set -u
cd ~/blackout-env-r2
OUT=reports/p17_unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CK=$HOME/blackout-env/checkpoints/distill/frozen
P17=ckpt:$CK/p17_v28.pt; P16=ckpt:$CK/p16_v27.pt
echo "start $(date)"
~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "$P17;$P16" \
    --opponent "strategic_v28:reserve_stage=0.5,trade_hold=advance;strategic_v25;$P16" \
    --seeds 150 --seed0 4000000 --cap-s 425 --workers 8 --out $OUT
echo "done $(date)"
rm -f $OUT/run.pid
