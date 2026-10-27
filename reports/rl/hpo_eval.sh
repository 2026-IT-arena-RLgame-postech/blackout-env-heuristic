#!/bin/bash
# Learning-curve eval of one HPO trial: iter_50/100/200 (128 seeds) + final (256 seeds) x 2 sides vs p24c, p22, V32.1, V33
# (fast eval) -> reports/rl/hpo_eval/hpo_<trial>_<ck>__<opp>__<lo>.json.  Paired against the base trial by seed.
# Usage: TRIAL=<name> [REPO=.. GPU=..] reports/rl/hpo_eval.sh   (never GPU 3)
set -e
cd ${REPO:-.}
[ "$GPU" != 3 ] || exit 1; [ -z "$GPU" ] || export CUDA_VISIBLE_DEVICES=$GPU
F=checkpoints/distill/frozen; C=checkpoints/rl/hpo_${TRIAL:?}
OPPS="net:$F/p24c_v321.pt;net:$F/p22_v31.pt;V32.1;V33"
mkdir -p reports/rl/hpo_eval
for ck in iter_50 iter_100 iter_200 final; do
  hi=970128; [ $ck = final ] && hi=970256
  PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python examples/eval_fast.py --cand $C/$ck.pt --opps "$OPPS" \
    --lo 970000 --hi $hi --chunk 128 --out reports/rl/hpo_eval --tag hpo_${TRIAL}_$ck --device cuda:0 --heur-workers ${HW:-8} \
    > reports/rl/hpo_eval/hpo_${TRIAL}_$ck.log 2>&1
done
echo "$(date) hpo eval done $TRIAL"
