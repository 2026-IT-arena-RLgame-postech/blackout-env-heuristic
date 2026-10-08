#!/bin/bash
# QZ plain-policy strength check (docs "QZ"): P20 fp32 vs P20 fp16c (fast checkpoint), 64 fresh seeds x 2 sides,
# vs V30 / V28 / P17 on gpu2 (CUDA nets, guarded CPU workers).  R = clone, OUT, EW workers, SEEDS, SEED0.
R=${R:-$HOME/blackout-env-qz}; OUT=${OUT:-$R/reports/qz_quant/plain64}; F=$R/checkpoints/distill/frozen
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
[ -e $F/p20_v30_fp16c.pt ] || $HOME/blackout-env/.venv/bin/python -c "
from blackout_env.sim.fastnet import fast_checkpoint; fast_checkpoint('$F/p20_v30.pt', '$F/p20_v30_fp16c.pt', 'fp16c')"
export BLACKOUT_NETS=p20=$F/p20_v30.pt,p20c=$F/p20_v30_fp16c.pt,p17=$F/p17_v28.pt
V30="strategic_v30:reserve_stage=0.5,trade_hold=advance,reserve_join=reserve,solo_sentry=true"
V28="strategic_v28:reserve_stage=0.5,trade_hold=advance"
for o in "$V30" "$V28" p17; do
  $HOME/blackout-env/.venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-64} --seed0 ${SEED0:-2000000} --cap-s 425 \
    --stop-exhausted --device cuda --chunk 16 --workers ${EW:-8} --out $OUT --ours "p20;p20c" --opponent "$o"
  $HOME/blackout-env/.venv/bin/python examples/v27_eval.py report --base p20 --out $OUT > $OUT/report.txt
done
echo "PLAIN DONE" >> $OUT/stdout.log
