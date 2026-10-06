#!/bin/bash
# P20 Unity milestone on poscat_gpu1 (CPU): P20 and P17 (base) vs V30, V28(0.5, advance), P17, V25; 300 matches/pair, cap 425 s.
# setsid nohup nice -n 10 bash reports/p20_unity/scripts/run.sh </dev/null > reports/p20_unity/play.log 2>&1 &
set -u
cd ~/blackout-env-u20
OUT=reports/p20_unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
CK=$HOME/blackout-env/checkpoints/distill/frozen
P20=ckpt:$CK/p20_v30.pt; P17=ckpt:$CK/p17_v28.pt
echo "start $(date)"
~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "$P20;$P17" \
    --opponent "strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true;strategic_v28:reserve_stage=0.5,trade_hold=advance;$P17;strategic_v25" \
    --seeds 150 --seed0 4000000 --cap-s 425 --workers 8 --out $OUT
echo "done $(date)"
rm -f $OUT/run.pid
