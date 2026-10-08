#!/bin/bash
# P24 Unity milestone (docs §60) on poscat_gpu1 (CPU), clone ~/blackout-env-u24 (build -> ~/blackout-env/build).
# part 1: P24c (V32.1 student, p24c_v321.pt) and P22 (base) vs V32.1, V32, V31, P22, P21e; 150 seeds x 2 sides from 4400000, cap 425 s.
# part 2: V32.1 and V31 (base) vs V31, P22, V23; 100 seeds from 4500000 (the V32.1 latch in Unity).
# Usage: run.sh <1|2>; env W (workers, 5), WAIT_PID.
set -u
while [ -n "${WAIT_PID:-}" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
cd ~/blackout-env-u24
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
CK=$HOME/blackout-env/checkpoints/distill/frozen; N=$HOME/p24nets
V32=$(.venv/bin/python -c "from blackout_env.heuristics.v32_four_hunters import V32; print(V32)")
V321="$V32,adapt_k=3"
V31=strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true,threat_radius=0.13,camp_engage_radius=6.5
if [ "$1" = 1 ]; then
  OUT=reports/p24_unity; mkdir -p $OUT
  nice -n 10 .venv/bin/python examples/unity_race_rule_eval.py play --ours "ckpt:$N/p24c9.pt;ckpt:$CK/p22_v31.pt" \
    --opponent "$V321;$V32;$V31;ckpt:$CK/p22_v31.pt;ckpt:$CK/p21e_v30.pt" \
    --seeds 150 --seed0 4400000 --cap-s 425 --workers ${W:-5} --out $OUT >> $OUT/play.log 2>&1
else
  OUT=reports/p24_unity/v321; mkdir -p $OUT
  nice -n 10 .venv/bin/python examples/unity_race_rule_eval.py play --ours "$V321;$V31" \
    --opponent "$V31;ckpt:$CK/p22_v31.pt;strategic_v23" \
    --seeds 100 --seed0 4500000 --cap-s 425 --workers ${W:-5} --out $OUT >> $OUT/play.log 2>&1
fi
echo "UNITY $1 DONE" >> $OUT/play.log
