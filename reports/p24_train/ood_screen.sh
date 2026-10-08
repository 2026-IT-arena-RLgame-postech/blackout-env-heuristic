#!/bin/bash
# P24 (docs §60) OOD-26 screen on wsl CPU (ood_suite merged in feat/policy-v2):
# OURS (default "p22") vs ood26 + v30_h4, SEEDS (32) x 2 sides from 2600000 = OOD1's screen seeds (§56), cap 425 s, --stop-exhausted.
# P22's rows are pre-seeded from reports/p23_ood/p22 (same seeds, CPU) so only missing cells are played.  nice 10, 1 thread each.
# Usage: ood_screen.sh <out name>; env OURS, SEEDS, EW (6), WAIT_PID; out reports/p24_ood/<name>.
while [ -n "$WAIT_PID" ] && kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
R=$HOME/blackout-env-p24; OUT=$R/reports/p24_ood/$1; F=${NETDIR:-$HOME/p14frozen}
cd $R; mkdir -p $OUT
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export BLACKOUT_NETS=$EXTRA_NETS${EXTRA_NETS:+,}p22=$F/p22.pt,p21e=$F/p21e.pt,p20=$F/p20.pt,p17=$F/p17.pt,p14b=$F/p14b.pt
nice -n 10 .venv/bin/python examples/v27_eval.py play --seeds ${SEEDS:-32} --seed0 2600000 --cap-s 425 --stop-exhausted \
  --device cpu --chunk 16 --workers ${EW:-6} --out $OUT --ours "${OURS:-p22}" --opponent "ood26;ood26=v30_h4"
.venv/bin/python examples/ood_report.py --out $OUT --base ${BASE:-p22} > $OUT/ood_report.txt
echo "OOD DONE" >> $OUT/stdout.log
