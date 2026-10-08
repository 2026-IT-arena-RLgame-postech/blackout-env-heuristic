#!/bin/bash
# Q-head data: search-r5 (B8, 1 s, margin 3) self-play roots with every branch's leaf value
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. NUMBA_NUM_THREADS=6 OMP_NUM_THREADS=2
G=$1; S0=$2; N=$3; OPP=${4:-r5}
$PY examples/s3_search_eval.py play --opponent $OPP --seeds $N --seed0 $S0 --margin 3 --tag q8m3 --record-roots --no-plain --cap-s ${CAP:-150} --chunk ${CHUNK:-128} \
   --device cuda --out reports/s3_qdata/${PREFIX:-g}${G}_$OPP
echo QDATA DONE
