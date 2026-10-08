#!/bin/bash
# P22 Unity milestone on poscat_gpu1 (CPU): P22 (P22b) and P21e (base) vs V31, V30, P21e, P20, V28(0.5, advance); 300 matches/pair, cap 425 s.
set -u
cd ~/blackout-env-u22
OUT=reports/p22_unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
CK=$HOME/blackout-env/checkpoints/distill/frozen
V30=strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true
V31=$V30,threat_radius=0.13,camp_engage_radius=6.5
~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "ckpt:$CK/p22_v31.pt;ckpt:$CK/p21e_v30.pt" \
    --opponent "$V31;$V30;ckpt:$CK/p21e_v30.pt;ckpt:$CK/p20_v30.pt;strategic_v28:reserve_stage=0.5,trade_hold=advance" \
    --seeds 150 --seed0 4300000 --cap-s 425 --workers 8 --out $OUT > $OUT/play.log 2>&1
rm -f $OUT/run.pid
