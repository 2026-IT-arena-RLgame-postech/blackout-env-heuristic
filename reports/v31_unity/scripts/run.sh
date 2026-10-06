#!/bin/bash
# V31 Unity check on poscat_gpu1 (CPU): V31 and V30 (base) vs V30, P17, V25, P20; 300 matches/pair, cap 425 s.
set -u
cd ~/blackout-env-u31
OUT=reports/v31_unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
CK=$HOME/blackout-env/checkpoints/distill/frozen
V30=strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true
V31=$V30,threat_radius=0.13,camp_engage_radius=6.5
~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "$V31;$V30" \
    --opponent "$V30;ckpt:$CK/p17_v28.pt;strategic_v25;ckpt:$CK/p20_v30.pt" \
    --seeds 150 --seed0 4100000 --cap-s 425 --workers 8 --out $OUT
rm -f $OUT/run.pid
