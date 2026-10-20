#!/bin/bash
# usage: run1.sh gpu name cand opp lo hi
cd ~/blackout-env-vitabc
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$1
o=~/vitabc_eval/$2.json
[ -f $o ] && exit 0
nice -n 30 .venv/bin/python examples/eval_grpo_ckpt.py --cand $3 --opp "$4" --lo $5 --hi $6 --out $o --device cuda:0 >> ~/vitabc_eval/logs/$2.log 2>&1
echo "$(date +%T) done $2 rc=$?" >> ~/vitabc_eval/progress.log
