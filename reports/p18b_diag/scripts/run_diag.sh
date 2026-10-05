#!/bin/bash
# P18b diagnosis on wsl_gpu: (1) Hunters at decision 99 (cap 5 s), 32 seeds; (2) full matches P18/P17 vs V28/V23, 16 seeds.
R=$HOME/blackout-env-p18b; OUT=$R/reports/p18b_diag
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=p18=$HOME/p14frozen/p18.pt,p17=$HOME/p14frozen/p17.pt,p16=$HOME/p14frozen/p16.pt,p18v10=$HOME/p14runs/p18/v_0010.pt
PY=.venv/bin/python; D=reports/p18b_diag/scripts/diag.py
$PY $D play --ours "p18;p17;V29;p16;p18v10" --opponent "V28;p17;V29;V23;V25;V27" --seeds 32 --cap-s 5 --workers 11 --chunk 16 --device cuda --out $OUT/h99
$PY $D summary --out $OUT/h99 > $OUT/h99/summary.txt
$PY $D play --ours "p18;p17" --opponent "V28;V23" --seeds 16 --workers 11 --chunk 4 --device cuda --out $OUT/full
$PY $D summary --out $OUT/full > $OUT/full/summary.txt
echo DIAG DONE
