#!/bin/bash
# V30 Unity milestone (docs §44) on poscat_gpu2: V30 and V29 (base) vs V29, P17, V25; 300 matches/pair, cap 425 s.
# V29 vs P17 / V25 rows are reused from reports/q10_v29/unity (same build, seeds, cap); copy them in first.
# setsid nohup bash reports/q11_v30/unity/scripts/run.sh </dev/null > reports/q11_v30/unity/play.log 2>&1 &
set -u
cd ~/blackout-env-q11
OUT=reports/q11_v30/unity
mkdir -p $OUT; echo $$ > $OUT/run.pid
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CK=$HOME/blackout-env/checkpoints/distill/frozen
V29="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve"
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
echo "start $(date)"
nice -n 10 ~/blackout-env/.venv/bin/python examples/unity_race_rule_eval.py play --ours "$V30;$V29;" \
    --opponent "$V29;ckpt:$CK/p17_v28.pt;strategic_v25;" --build $HOME/blackout-env/build/linux_server/BlackOut.x86_64 \
    --seeds 150 --seed0 4000000 --cap-s 425 --workers ${WORKERS:-8} --out $OUT
echo "done $(date)"
rm -f $OUT/run.pid
