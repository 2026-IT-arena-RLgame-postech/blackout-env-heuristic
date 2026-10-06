#!/bin/bash
# Q12 counterfactual lane (docs §46), CPU only, base = V30: cf_lane.sh OPP PART NROOTS [extra args]
set -u
OPP=$1; PART=$2; N=$3; shift 3
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
nice -n 10 ~/blackout-env/.venv/bin/python examples/q11_why.py cf --base v30 --opponent $OPP --n-roots $N --part $PART \
    --suffix _cp${PART%%/*} --device cpu --out reports/q12_v31 "$@" && echo "CF DONE $OPP $PART" || echo "CF FAILED $OPP $PART"
