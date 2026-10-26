#!/bin/bash
# Angle-precision probe: is 32 bins + deterministic offset a bottleneck?  The candidate (default E4) plays with its move
# angle (a) unchanged (q0), (b) at bin centres, offset ignored (q-1 = 32 dirs, 11.25 deg), (c) rounded to 32/64/128 dirs
# (examples/eval_fast.py --angle-quant; opponents untouched).  256 seeds x 2 sides vs p24c/p22/V32.1/V33 + hunt-style OOD
# (v30_h4hunt, v30_hunt, v30_reckless, early_rush, rush_all).  Small loss from q0 -> q32: precision is not the bottleneck.
# Usage: [REPO=..] GPU=n QS="0 -1 32" [CAND=..] reports/rl/angle_probe.sh   (never GPU 3)
set -e
cd ${REPO:-.}
[ "$GPU" != 3 ] || { echo "GPU 3 is off limits"; exit 1; }; export CUDA_VISIBLE_DEVICES=${GPU:?}
CAND=${CAND:-checkpoints/rl/abl_beta01/final.pt}
F=checkpoints/distill/frozen
OOD=$(PYTHONPATH=. .venv/bin/python -c "
from blackout_env.train.distill.ood_suite import OOD26, REFERENCE, FOLLOWUP
d={n:x for n,x,_,_ in OOD26+REFERENCE+FOLLOWUP}
print(';'.join(d[n] for n in ['v30_h4hunt','v30_hunt','v30_reckless','early_rush','rush_all']))")
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;V32.1;V33;$OOD"
mkdir -p reports/rl/angle_probe
for Q in ${QS:?}; do
  PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/eval_fast.py --cand $CAND --opps "$OPPS" \
    --lo 970000 --hi 970256 --chunk ${CHUNK:-128} --out reports/rl/angle_probe --tag q$Q --device cuda:0 --heur-workers ${HW:-8} \
    --angle-quant $Q > reports/rl/angle_probe/q$Q.log 2>&1
  echo "$(date) probe done q$Q"
done
