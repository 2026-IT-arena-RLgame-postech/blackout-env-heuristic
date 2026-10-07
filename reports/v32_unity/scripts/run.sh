#!/bin/bash
# V32 (docs §58) early Unity spot-check on poscat_gpu1 (CPU, nice 10): B and V31 (base) vs V31 and P22, 50 seeds x 2 sides
# from 4300000, cap 425 s, 8 Unity workers.  Uses the Linux server build of ~/blackout-env-u22 read-only.
set -u
cd ~/blackout-env-v32
OUT=reports/v32_unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
CK=$PWD/checkpoints/distill/frozen
P=reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5
V31=strategic_v30:$P
B=strategic_v32:$P,hunter_quota=4,solo_sentry=false,sentry_min_hunters=3,exit_guards=1,guard_min_hunters=2,sweep=true,cargo_transform=true
nice -n 10 .venv/bin/python examples/unity_race_rule_eval.py play --build $HOME/blackout-env-u22/build/linux_server/BlackOut.x86_64 \
    --ours "$B;$V31" --opponent "$V31;ckpt:$CK/p22_v31.pt" \
    --seeds 50 --seed0 4300000 --cap-s 425 --workers 8 --out $OUT > $OUT/play.log 2>&1
.venv/bin/python examples/unity_race_rule_eval.py report --base "$V31" --out $OUT > $OUT/report.txt 2>&1
rm -f $OUT/run.pid
echo UNITY DONE >> $OUT/play.log
