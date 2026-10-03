#!/bin/bash
# docs §27: S1.6 recipe (no shard rows: the S1 cache is not on the servers) with final-head variants.
# usage: run_variants.sh name:"flags" ...   (run from ~/blackout-env-vmse, one GPU via CUDA_VISIBLE_DEVICES)
set -u
cd ~/blackout-env-vmse
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4
CK=checkpoints/value_mse; R=reports/value_mse
mkdir -p $CK $R
T="nice -n 10 $PY examples/value_branch_train.py train --shard-frac 0 --compile --device cuda --log-every 500"
RANK="--group-roots 16 --rank-weight 2"
for spec in "$@"; do
  name=${spec%%:*}; opts=${spec#*:}; last=${LAST_OPTS:-}
  D=$CK/$name; mkdir -p $D
  [ "$name" = ctl ] && last="--final-beta 10"   # = s2v_d iter5 (Huber beta 10 in the last round only)
  echo "== $name: $opts ($(date))"
  [ -f $D/iter1.pt ] || $T --init checkpoints/value/frozen/s1b_value.pt --out-dir $D --iters 1 --steps 4000 $opts > $D/train1.log 2>&1
  [ -f $D/iter3.pt ] || $T --init $D/iter1.pt --out-dir $D --start-iter 2 --iters 2 --steps 3000 $RANK $opts > $D/train23.log 2>&1
  [ -f $D/iter4.pt ] || $T --init $D/iter3.pt --out-dir $D --start-iter 4 --iters 1 --steps 2000 $RANK $opts > $D/train4.log 2>&1
  [ -f $D/iter5.pt ] || $T --init $D/iter4.pt --out-dir $D --start-iter 5 --iters 1 --steps 1500 $RANK $opts $last > $D/train5.log 2>&1
  ls $D/iter5.pt || { echo "FAILED $name"; tail -20 $D/train*.log; }
done
echo VARIANTS DONE $(date)
