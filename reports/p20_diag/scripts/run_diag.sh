#!/bin/bash
# P20 diagnosis on wsl_gpu: (1) Hunters at decision 99 (cap 5 s) for P19/V30, 32 seeds; (2) full matches vs V28 / P17, 32 seeds
# (seeds 1100000+ = the same (seed, side) as reports/p19_sim/matches.jsonl).
R=$HOME/blackout-env-p18b; OUT=$R/reports/p20_diag
cd $R
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
export BLACKOUT_NETS=p19=$HOME/p14frozen/p19.pt,p18bp2=$HOME/p14frozen/p18bp2.pt,p17=$HOME/p14frozen/p17.pt
PY=.venv/bin/python; D=reports/p20_diag/scripts/diag.py
$PY $D play --ours "p19;V30" --opponent "V28;p17;V30;V23" --seeds 32 --cap-s 5 --workers 11 --chunk 16 --device cuda --out $OUT/h99
$PY $D summary --out $OUT/h99 > $OUT/h99/summary.txt
$PY $D play --ours "p17;p18bp2;p19;V30" --opponent "V28" --seeds 32 --workers 11 --chunk 4 --device cuda --out $OUT/full
$PY $D summary --out $OUT/full > $OUT/full/summary.txt
echo DIAG DONE
