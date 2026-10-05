#!/bin/bash
# P18b teacher check (sim, 16 seeds 1100000+, cap 425, --stop-exhausted): V29m (= V29 with join_margin=10: join whenever the
# enemy stages a close reserve) and the training teacher joinswitch(V29m | P17) vs V28 / P17; base rows P17 / V29 from reports/p18_sim.
R=$HOME/blackout-env-p18b; OUT=$R/reports/p18b_teacher
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 BLACKOUT_NETS=p17=$HOME/p14frozen/p17.pt
V29M="strategic_v29:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,join_margin=10"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
T="joinswitch:3.6,4.4,4.4:$V29M|ckpt:$HOME/p14frozen/p17.pt"
[ -e $OUT/matches.jsonl ] || grep -E '"ours": "(p17|strategic_v29[^"]*)"' reports/p18_sim/matches.jsonl > $OUT/matches.jsonl
.venv/bin/python examples/v27_eval.py play --ours "$V29M;$T" --opponent "$V28;p17" --seeds ${SEEDS:-16} --seed0 1100000 \
  --cap-s 425 --stop-exhausted --device cpu --chunk 2 --workers ${WORKERS:-4} --out $OUT
.venv/bin/python examples/v27_eval.py report --base p17 --out $OUT > $OUT/report.txt
echo TEACHER DONE
