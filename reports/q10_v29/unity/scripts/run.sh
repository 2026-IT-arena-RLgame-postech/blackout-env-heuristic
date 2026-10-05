#!/bin/bash
# V29 Unity milestone (docs §41) on poscat_gpu2: V29 and V28 (base) vs V28(0.5, advance), P17, V25; 300 matches/pair, cap 425 s.
# setsid nohup bash reports/q10_v29/unity/scripts/run.sh </dev/null > reports/q10_v29/unity/play.log 2>&1 &
set -u
cd ~/blackout-env-q10
OUT=reports/q10_v29/unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CK=$HOME/blackout-env/checkpoints/distill/frozen
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
echo "start $(date)"
~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "$V29;$V28;" \
    --opponent "$V28;ckpt:$CK/p17_v28.pt;strategic_v25;" --build $HOME/blackout-env/build/linux_server/BlackOut.x86_64 \
    --seeds 150 --seed0 4000000 --cap-s 425 --workers 8 --out $OUT
echo "done $(date)"
rm -f $OUT/run.pid
