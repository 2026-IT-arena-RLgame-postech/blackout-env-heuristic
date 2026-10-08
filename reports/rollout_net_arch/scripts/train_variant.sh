#!/bin/bash
# r5's two-stage recipe for one RolloutNet variant (examples/rollout_net_arch.py recipe), then held-out eval
cd ~/blackout-env-arch
PY=~/blackout-env/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=2
V=$1; AMP=${AMP:-fp16}; N=${NAME:-$V}; SEED=${SEED:-0}
CK=checkpoints/rollout_net_arch; R=reports/rollout_net_arch/train
D=checkpoints/rollout_net/data; E=checkpoints/rollout_net/simdata/train_ex23.npz
mkdir -p $R $CK
CFG=$($PY -c "import sys; sys.path.insert(0,'examples'); import json, rollout_net_arch as a; print(json.dumps(a.config('$V')))")
[ -f $CK/${N}_s1.pt.done ] || { $PY examples/rollout_net_train.py train --data $D --out $CK/${N}_s1.pt --config "$CFG" --seed $SEED \
   --steps 30000 --lr 2e-3 --p11-kl 1.0 --amp $AMP --eval-every 10000 ${TRAIN_OPTS:---in-memory 0 --gpu-data 0} > $R/${N}_s1.log 2>&1 && touch $CK/${N}_s1.pt.done; } || exit 1
$PY examples/rollout_net_train.py train --data $D --out $CK/${N}.pt --init $CK/${N}_s1.pt --seed $SEED --steps 20000 --lr 1e-3 \
   --p11-kl 0.5 --extra $E --extra-frac 0.5 --amp $AMP --eval-every 10000 ${TRAIN_OPTS:---in-memory 0 --gpu-data 0} > $R/${N}_s2.log 2>&1 || exit 1
$PY examples/rollout_net_train.py eval --data $D --net $CK/${N}.pt --val2 checkpoints/rollout_net/simval/val.npz \
   --json reports/rollout_net_arch/heldout/${N}.json > $R/${N}_eval.log 2>&1
echo TRAIN DONE $N
