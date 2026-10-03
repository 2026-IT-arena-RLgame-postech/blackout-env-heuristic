#!/bin/bash
# docs §27: after run_variants.sh -- bias diagnosis + cached r5 x r5 leaf scoring of every variant's iter5
set -u
cd ~/blackout-env-vmse
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4
R=reports/value_mse; CK=checkpoints/value_mse
until grep -q "VARIANTS DONE" $R/variants.log; do sleep 60; done
specs="iter5=checkpoints/value/frozen/s2v_iter5.pt"
for d in $CK/*/; do n=$(basename $d); [ -f $d/iter5.pt ] && specs="$specs $n=$d/iter5.pt"; done
echo "specs: $specs"
nice -n 10 $PY examples/value_branch_train.py diag --device cuda --out $R/diag_variants.json --ckpt $specs > $R/diag_variants.log 2>&1
nice -n 10 $PY examples/leaf_rollout_eval.py --from-cache $CK/leaf_cache.pkl --device cuda --values $specs \
  --paired-ref iter5_sym --tag variants --out $R/leaf > $R/leaf_variants.log 2>&1
echo EVAL DONE $(date)
