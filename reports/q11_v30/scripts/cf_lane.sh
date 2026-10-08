#!/bin/bash
# Q11 counterfactual lane (docs §44), CPU only: cf_lane.sh OPP PART NROOTS SUFFIX [extra args]
set -u
OPP=$1; PART=$2; N=$3; SUF=$4; shift 4
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
nice -n 10 ~/blackout-env/.venv/bin/python examples/q11_why.py cf --opponent $OPP --n-roots $N --part $PART \
    --suffix ${SUF}p${PART%%/*} --device cpu "$@" && echo "CF DONE $OPP $PART" || echo "CF FAILED $OPP $PART"
