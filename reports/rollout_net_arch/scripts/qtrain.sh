#!/bin/bash
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2
CK=checkpoints/rollout_net_arch; R=reports/rollout_net_arch/qhead
mkdir -p $R
D=$CK/qdata_${QD:-r5}
[ -f $D/val.npz ] || $PY examples/q_head_train.py pack --roots "${ROOTS:-reports/s3_qdata/g[12]_r5/roots/*.npz}" --out $D > $R/pack_${QD:-r5}.log 2>&1
for spec in "$@"; do
  name=${spec%%:*}; opts=${spec#*:}
  $PY examples/q_head_train.py train --data $D --init ${INIT:-checkpoints/rollout_net/r5.pt} --out $CK/$name.pt $opts > $R/train_$name.log 2>&1
done
echo QTRAIN DONE
