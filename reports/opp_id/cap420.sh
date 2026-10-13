#!/bin/bash
# Does the §61 search gain (+16.7 vs V32.1 at the 150 s cap) survive full 420 s matches?  plain vs o4f_r24c, lead traced at 150 / 300 s.
cd ~/blackout-env
export CUDA_VISIBLE_DEVICES=${1:-3} PYTHONPATH=. OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
nice -n 5 .venv/bin/python examples/blind_search_eval.py play --base p24c --opponent v321 --tags plain,o4f_r24c \
    --seeds 32 --seed0 6300000 --cap-s 420 --out reports/opp_id/cap420 --device cuda --chunk 64
