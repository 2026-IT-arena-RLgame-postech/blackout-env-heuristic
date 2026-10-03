#!/bin/bash
# DAgger round 2: the net's own sim DAgger rows (rollout_net_collect.py, V23 labels) + P11 targets, fine-tune, eval
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2 BLACKOUT_P11=$HOME/blackout-env/checkpoints/distill/frozen/p11_v22.pt
V=$1; CK=checkpoints/rollout_net_arch; R=reports/rollout_net_arch/train; D=$CK/dagger_$V
[ -f $D/train_p11.npz ] || $PY - <<PYEOF
import sys; sys.path.insert(0, "examples")
import numpy as np
from rollout_net_train import p11_targets
d = dict(np.load("$D/train.npz"))
d.update(p11_targets(d, device="cuda"))
np.savez("$D/train_p11.npz", **d)
print("rows", len(d["tick"]))
PYEOF
$PY examples/rollout_net_train.py train --data checkpoints/rollout_net/data --out $CK/${V}_d2.pt --init $CK/$V.pt --steps 10000 \
   --lr 3e-4 --p11-kl 0.5 --extra $D/train_p11.npz --extra-frac 0.5 --amp fp16 --eval-every 5000 --in-memory 0 --gpu-data 0 > $R/${V}_d2.log 2>&1 || exit 1
$PY examples/rollout_net_train.py eval --data checkpoints/rollout_net/data --net $CK/${V}_d2.pt --val2 checkpoints/rollout_net/simval/val.npz \
   --json reports/rollout_net_arch/heldout/${V}_d2.json > $R/${V}_d2_eval.log 2>&1
echo TRAIN DONE ${V}_d2 > $R/run_${V}_d2.log
WORKERS=${WORKERS:-6} bash eval_variant.sh ${V}_d2 > reports/rollout_net_arch/eval_${V}_d2.log 2>&1
rm -f $D/part_*.npz
echo D2 DONE $V
