#!/bin/bash
# V32 (docs §58): per-phase diag (reports/ood1/scripts/diag2.py) of A and B vs H4 / P22 / V31, SEEDS x 2 from 5200000.
# Usage: diag.sh OUT ; env R (repo), EW (workers, 4), NICE (15).
R=${R:-$(cd "$(dirname "$0")/../../.." && pwd)}; cd $R
. reports/v32/scripts/names.sh
export PYTHONPATH=$R NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
PY=${PY:-.venv/bin/python}
nice -n ${NICE:-15} $PY reports/ood1/scripts/diag2.py play --ours "$(expand "A;B")" --opponent "$(expand "H4;p22;V31")" \
  --seeds ${SEEDS:-32} --seed0 5200000 --workers ${EW:-4} --chunk 8 --out $1 > $1.log 2>&1
$PY reports/ood1/scripts/diag2.py summary --out $1 > $1/summary.txt
echo DIAG DONE >> $1.log
